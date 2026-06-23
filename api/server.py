from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.pool_health import build_pool_health
from api.telemetry import RequestRecord, telemetry
from connectors.dotenv import load_env
from connectors.paths import default_pool_dir
from connectors.pool_resolver import resolve_pool
from loop.runner import OrchestrationLoop, live_enabled

QUALITY_TIERS = ("fast", "balanced", "max")


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None


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


def _record_request(
    *,
    tier: str,
    result,
    latency_ms: float,
    stream: bool,
) -> None:
    telemetry.record(
        RequestRecord(
            ts=time.time(),
            tier=tier,
            connector_id=result.connector_id,
            model_id=result.model_id,
            stages=list(result.stages),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.estimated_cost_usd,
            latency_ms=latency_ms,
            passed=result.passed,
            stream=stream,
        )
    )


def create_app(connectors_dir: str | None = None) -> FastAPI:
    load_env()
    app = FastAPI(title="mat", version="0.1.0")
    explicit_pool_dir = connectors_dir if connectors_dir else None

    def _load_pool() -> list:
        try:
            res = resolve_pool(pool_dir=explicit_pool_dir)
            pool = res.pool
        except (OSError, ValueError) as exc:
            app.state.pool_error = str(exc)
            return []
        app.state.pool_error = None
        app.state.pool_source = res.source
        app.state.pool_dir = (
            str(explicit_pool_dir)
            if explicit_pool_dir
            else str(res.library_dir or os.environ.get("MAT_POOL_DIR") or default_pool_dir())
        )
        app.state.active_manifest = str(res.active_manifest) if res.active_manifest else None
        return pool

    pool = _load_pool()

    def _loop(tier: str) -> OrchestrationLoop:
        return OrchestrationLoop.from_env(pool, quality_tier=tier)

    def _health_payload(*, reload: bool = False) -> dict:
        nonlocal pool
        if reload or app.state.pool_error:
            pool = _load_pool()
        pool_health = build_pool_health(pool) if pool else {"status": "empty", "connectors": []}
        overall = "ok"
        if app.state.pool_error or not pool:
            overall = "error"
        elif pool_health.get("status") == "degraded":
            overall = "degraded"
        return {
            "status": overall,
            "live": live_enabled(),
            "pool_size": len(pool),
            "pool_dir": str(getattr(app.state, "pool_dir", "")),
            "pool_source": getattr(app.state, "pool_source", None),
            "active_manifest": getattr(app.state, "active_manifest", None),
            "pool_error": app.state.pool_error,
            "pool_health": pool_health,
            "metrics": telemetry.snapshot(),
        }

    @app.get("/health")
    def health() -> dict:
        return _health_payload(reload=bool(app.state.pool_error))

    @app.get("/v1/mat/status")
    def mat_status(authorization: str | None = Header(default=None)) -> dict:
        _check_auth(authorization)
        return _health_payload(reload=True)

    @app.get("/v1/mat/metrics")
    def mat_metrics(authorization: str | None = Header(default=None)) -> dict:
        _check_auth(authorization)
        return telemetry.snapshot()

    @app.get("/v1/mat/recent")
    def mat_recent(
        authorization: str | None = Header(default=None),
        limit: int = 20,
    ) -> dict:
        _check_auth(authorization)
        return {"recent": telemetry.recent(limit=min(limit, 50))}

    @app.get("/v1/models")
    def list_models() -> dict:
        data = [{"id": tier, "object": "model", "owned_by": "mat"} for tier in QUALITY_TIERS]
        for c in pool:
            data.append({"id": c.id, "object": "model", "owned_by": "mat"})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(
        body: ChatRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        _check_auth(authorization)
        nonlocal pool
        if app.state.pool_error:
            pool = _load_pool()
        if not pool:
            raise HTTPException(
                status_code=503,
                detail=f"no connectors (pool_source={getattr(app.state,'pool_source',None)}); "
                f"check active.yaml or MAT_POOL_DIR ({getattr(app.state,'pool_dir',None)})",
            )
        tier = _map_model(body.model)
        messages = [m.model_dump() for m in body.messages]
        loop = _loop(tier)

        if body.stream:
            return StreamingResponse(
                _stream_response(loop, messages, tier),
                media_type="text/event-stream",
            )

        t0 = time.perf_counter()
        result = loop.run(messages)
        _record_request(
            tier=tier,
            result=result,
            latency_ms=(time.perf_counter() - t0) * 1000,
            stream=False,
        )
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
                "stages": result.stages,
                "connector_id": result.connector_id,
                "model_id": result.model_id,
                "passed": result.passed,
                "cost_usd": result.estimated_cost_usd,
            },
        },
    }


def _stream_response(loop: OrchestrationLoop, messages: list[dict], tier: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    connector_id = ""
    model_id: str | None = None
    stages: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    passed = True
    answer_parts: list[str] = []

    for chunk in loop.run_stream(messages):
        if chunk.text:
            answer_parts.append(chunk.text)
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json_dumps(payload)}\n\n"
        if chunk.done:
            connector_id = chunk.connector_id or connector_id
            stages = chunk.stages or stages
            passed = chunk.passed
            cost_usd = chunk.cost_usd
            if chunk.result:
                model_id = chunk.result.model
                input_tokens = chunk.result.input_tokens
                output_tokens = chunk.result.output_tokens

    latency_ms = (time.perf_counter() - t0) * 1000
    from loop.runner import LoopResult

    pseudo = LoopResult(
        answer="".join(answer_parts),
        passed=passed,
        steps=len(stages) or 1,
        connector_id=connector_id or "(unknown)",
        model_id=model_id,
        stages=stages or ["stream"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost_usd,
    )
    _record_request(tier=tier, result=pseudo, latency_ms=latency_ms, stream=True)

    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "x_mat": {
                "stages": stages,
                "connector_id": connector_id,
                "model_id": model_id,
                "passed": passed,
                "cost_usd": cost_usd,
                "latency_ms": round(latency_ms, 1),
            },
        },
    }
    yield f"data: {_json_dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def _json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)


def main() -> None:
    import uvicorn

    load_env()
    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = int(os.environ.get("MAT_PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port)
