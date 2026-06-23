from __future__ import annotations

import pytest

from connectors.lmstudio_api import (
    ModelNotServedError,
    match_served_model_id,
    resolve_connector_model,
)

SERVED = (
    "qwen3.6-35b-a3b",
    "qwen3.6-27b",
    "gemma-4-26b-a4b-it-mlx",
    "gemma-4-31b-it-qat-optiq",
    "qwen/qwen3.5-9b",
    "liquid/lfm2-24b-a2b",
    "qwen3.6-27b-mtplx-optimized-speed",
)


@pytest.mark.parametrize(
    ("folder", "catalog", "expected"),
    [
        (
            "mlx-community-qwen3-6-35b-a3b-8bit",
            "mlx-community/qwen3-6-35b-a3b-8bit",
            "qwen3.6-35b-a3b",
        ),
        (
            "mlx-community-qwen3-6-27b-8bit",
            "mlx-community/qwen3-6-27b-8bit",
            "qwen3.6-27b",
        ),
        (
            "lmstudio-community-gemma-4-26b-a4b-it-mlx-8bit",
            "lmstudio-community/gemma-4-26b-a4b-it-mlx-8bit",
            "gemma-4-26b-a4b-it-mlx",
        ),
        (
            "lmstudio-community-qwen3-5-9b-gguf",
            "lmstudio-community/qwen3-5-9b-gguf",
            "qwen/qwen3.5-9b",
        ),
        (
            "lmstudio-community-lfm2-24b-a2b-mlx-8bit",
            "lmstudio-community/lfm2-24b-a2b-mlx-8bit",
            "liquid/lfm2-24b-a2b",
        ),
    ],
)
def test_match_served_model_id(folder: str, catalog: str, expected: str):
    assert match_served_model_id(folder, catalog, SERVED) == expected


def test_resolve_connector_model_strict():
    assert (
        resolve_connector_model(
            "a@lmstudio",
            "http://127.0.0.1:1234/v1",
            "qwen3.6-35b-a3b",
            validate=False,
        )
        == "qwen3.6-35b-a3b"
    )


def test_resolve_connector_model_rejects_unknown(monkeypatch):
    monkeypatch.setattr(
        "connectors.lmstudio_api.fetch_served_model_ids",
        lambda _url: SERVED,
    )
    with pytest.raises(ModelNotServedError):
        resolve_connector_model(
            "bad@lmstudio",
            "http://127.0.0.1:1234/v1",
            "nonexistent-model-xyz",
            validate=True,
        )
