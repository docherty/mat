from __future__ import annotations

from datetime import UTC, datetime

from connectors.schema import Connector
from workers.llm import _pick_lmstudio_model, resolve_model_name


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


def test_resolve_model_name_prefers_exact_match(monkeypatch):
    monkeypatch.setattr(
        "workers.llm._lmstudio_model_ids",
        lambda _url: ("qwen3.6-35b-a3b", "gemma-4-31b-it"),
    )
    conn = _connector("qwen3.6-35b-a3b")
    assert resolve_model_name(conn) == "qwen3.6-35b-a3b"


def test_pick_lmstudio_model_env_override(monkeypatch):
    monkeypatch.setenv("MAT_LMSTUDIO_MODEL", "custom-model")
    assert _pick_lmstudio_model("http://127.0.0.1:1234/v1", "ignored") == "custom-model"
