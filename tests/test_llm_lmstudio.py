from __future__ import annotations

from datetime import UTC, datetime

import pytest

from connectors.lmstudio_api import ModelNotServedError
from connectors.schema import Connector
from workers.llm import resolve_model_name


def _connector(model_name: str) -> Connector:
    return Connector.model_validate(
        {
            "connector_version": "1.1",
            "id": "test@lmstudio",
            "display_name": "test",
            "endpoint": {
                "type": "openai",
                "base_url": "http://127.0.0.1:1234/v1",
                "model_name": model_name,
                "auth_env": "LMSTUDIO_API_KEY",
            },
            "context_window": 8192,
            "max_output_tokens": 1024,
            "locality": "local",
            "capabilities": {
                tag: {"score": 0.5, "tier": "mid"}
                for tag in (
                    "reasoning",
                    "coding",
                    "long_context",
                    "instruction_following",
                    "verification",
                    "tool_use",
                )
            },
            "speed": {"tier": "medium"},
            "profile": {
                "profile_method": "benchmark_import",
                "catalog": "test",
                "catalog_id": "test",
                "profiled_at": datetime.now(UTC),
            },
        }
    )


def test_resolve_model_name_uses_connector_exactly(monkeypatch):
    monkeypatch.setattr(
        "connectors.lmstudio_api.fetch_served_model_ids",
        lambda _url: ("qwen3.6-35b-a3b", "gemma-4-31b-it"),
    )
    conn = _connector("qwen3.6-35b-a3b")
    assert resolve_model_name(conn) == "qwen3.6-35b-a3b"


def test_resolve_model_name_never_substitutes_other_model(monkeypatch):
    monkeypatch.setattr(
        "connectors.lmstudio_api.fetch_served_model_ids",
        lambda _url: ("qwen3.6-35b-a3b",),
    )
    conn = _connector("gemma-4-31b-it")
    with pytest.raises(ModelNotServedError):
        resolve_model_name(conn)


def test_resolve_model_name_non_lmstudio_skips_validation():
    conn = _connector("anything")
    conn.endpoint.base_url = "https://api.openrouter.ai/v1"
    assert resolve_model_name(conn) == "anything"
