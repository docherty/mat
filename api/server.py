from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from connectors.dotenv import load_env
from connectors.loader import load_connectors_dir
from connectors.paths import default_pool_dir
from loop.runner import OrchestrationLoop, live_enabled

QUALITY_TIERS = ("fast", "balanced", "max")


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str = "balanced"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None


def _map_model(name: str) -> str:
    if name in QUALITY_TIERS:
        return name
    return "balanced"


def _check_auth(authorization: str | None) -> None:
    expected = os.environ.get("MAT_GATEWAY_KEY", "local-dev-key")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid gateway key")


def create_app(connectors_dir: str | None = None) -> FastAPI:
    load_env()
    app = FastAPI(title="mat", version="0.1.0")
    pool_dir = connectors_dir or os.environ.get("MAT_POOL_DIR") or str(default_pool_dir())
    try:
        pool = load_connectors_dir(pool_dir)
    except (OSError, ValueError) as exc:
        pool = []
        app.state.pool_error = str(exc)
    else:
        app.state.pool_error = None
    app.state.pool_dir = pool_dir

    def _loop(tier: str) -> OrchestrationLoop:
        return OrchestrationLoop.from_env(pool, quality_tier=tier)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "live": live_enabled(),
            "pool_size": len(pool),
            "pool_dir": str(pool_dir),
            "pool_error": app.state.pool_error,
        }

    @app.get("/v1/models")
    def list_models() -> dict:
        data = [
            {"id": tier, "object": "model", "owned_by": "mat"}
            for tier in QUALITY_TIERS
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(
        body: ChatRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        _check_auth(authorization)
        if not pool:
            raise HTTPException(
                status_code=503,
                detail=f"no connectors in {pool_dir}; run mat-discover-lmstudio",
            )
        tier = _map_model(body.model)
        messages = [m.model_dump() for m in body.messages]
        loop = _loop(tier)

        if body.stream:
            return StreamingResponse(
                _stream_response(loop, messages, tier, body.seed),
                media_type="text/event-stream",
            )

        result = loop.run(messages)
        return _completion_payload(result, tier)

    return app


def _completion_payload(result, tier: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": tier,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
            "x_mat": {
                "stages": result.steps,
                "connector_id": result.connector_id,
                "passed": result.passed,
                "cost_usd": result.estimated_cost_usd,
            },
        },
    }


def _stream_response(loop: OrchestrationLoop, messages: list[dict], tier: str, seed: int | None):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    narrations = (
        []
        if tier == "fast"
        else [
            "Assessing task difficulty…",
            "Selecting specialists from pool…",
            "Synthesizing response…",
        ]
    )
    for line in narrations:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": line},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json_dumps(payload)}\n\n"
        time.sleep(0.05)

    result = loop.run(messages)
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": result.answer},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json_dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)


def main() -> None:
    import uvicorn

    load_env()
    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = int(os.environ.get("MAT_PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port)
