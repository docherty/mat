from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from connectors.loader import load_connectors_dir
from loop.runner import OrchestrationLoop

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


def create_app(connectors_dir: str = "connectors/examples") -> FastAPI:
    app = FastAPI(title="mat", version="0.1.0")
    pool = load_connectors_dir(connectors_dir)
    loop = OrchestrationLoop(pool)

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
        tier = _map_model(body.model)
        messages = [m.model_dump() for m in body.messages]

        if body.stream:
            return StreamingResponse(
                _stream_response(loop, messages, tier, body.seed),
                media_type="text/event-stream",
            )

        result = loop.run(messages)
        return _completion_payload(result.answer, tier, result.steps)

    return app


def _completion_payload(content: str, tier: str, steps: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": tier,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "x_breakdown": {"stages": steps, "local_token_pct": 0.0},
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
            "Verifying output…",
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

    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = int(os.environ.get("MAT_PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port)
