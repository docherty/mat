"""Build a mat connector skeleton from the OpenRouter model catalog."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.request import urlopen

from connectors.schema import (
    CapabilityDim,
    Connector,
    Endpoint,
    Pricing,
    Profile,
    Speed,
    Supports,
)

OPENROUTER_MODEL_URL = "https://openrouter.ai/api/v1/model/{model_id}"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _per_1k(token_price: str) -> float:
    return float(token_price) * 1000


def fetch_openrouter_model(model_id: str) -> dict:
    with urlopen(OPENROUTER_MODEL_URL.format(model_id=model_id)) as resp:
        payload = json.load(resp)
    return payload["data"]


def connector_from_openrouter(
    model_id: str,
    *,
    connector_id: str | None = None,
    capabilities: dict[str, CapabilityDim] | None = None,
    contributor: str = "mat:import-openrouter",
) -> Connector:
    """Operational fields from OpenRouter; capabilities must be supplied separately."""
    data = fetch_openrouter_model(model_id)
    params = set(data.get("supported_parameters") or [])
    top = data.get("top_provider") or {}
    max_out = top.get("max_completion_tokens") or 8192

    if capabilities is None:
        tags = (
            "reasoning",
            "coding",
            "long_context",
            "instruction_following",
            "verification",
            "tool_use",
        )
        capabilities = {tag: CapabilityDim.from_score(0.5) for tag in tags}

    pricing_raw = data.get("pricing") or {}
    pricing = None
    if pricing_raw.get("prompt") and pricing_raw.get("completion"):
        pricing = Pricing(
            input_per_1k=_per_1k(pricing_raw["prompt"]),
            output_per_1k=_per_1k(pricing_raw["completion"]),
        )

    return Connector(
        connector_version="1.1",
        id=connector_id or model_id.replace("/", "-") + "@openrouter",
        display_name=data.get("name", model_id),
        endpoint=Endpoint(
            type="openai",
            base_url=OPENROUTER_BASE,
            model_name=model_id,
            auth_env="OPENROUTER_API_KEY",
        ),
        context_window=int(data.get("context_length") or 32768),
        max_output_tokens=int(max_out),
        modalities=list((data.get("architecture") or {}).get("input_modalities") or ["text"]),
        pricing=pricing,
        locality="api",
        supports=Supports(
            tools="tools" in params,
            json_mode="structured_outputs" in params or "response_format" in params,
            streaming=True,
            system_prompt=True,
            reasoning="reasoning" in params,
        ),
        tool_format="openai",
        capabilities=capabilities,
        speed=Speed(tier="medium"),
        benchmarks=[],
        profile=Profile(
            profile_method="hand",
            catalog="openrouter",
            catalog_id=model_id,
            profiled_at=datetime.now(UTC),
            contributor=contributor,
            notes="Operational fields imported from OpenRouter; set capabilities before use.",
        ),
    )


def default_capability_from_index(index_score: float, *, scale_max: float = 60.0) -> CapabilityDim:
    """Rough single-number import when only a composite index is available."""
    return CapabilityDim.from_score(min(1.0, index_score / scale_max))
