from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from connectors.schema import Pricing


@dataclass(frozen=True)
class PricingUpdate:
    pricing: Pricing
    source: str


def _base(origin: str) -> str:
    return origin.rstrip("/")


def _per_1k_from_per_1m(usd_per_1m: float) -> float:
    return float(usd_per_1m) / 1000.0


def _per_1k_from_per_token(usd_per_token: str | float) -> float:
    return float(usd_per_token) * 1000.0


def fetch_venice_pricing(base_url: str, *, api_key: str) -> dict[str, PricingUpdate]:
    """Return {model_id: PricingUpdate} for Venice models.

    Venice returns pricing under `model_spec.pricing.{input,output}.usd` (USD per 1M tokens).
    """
    url = f"{_base(base_url)}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(url, headers=headers, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    out: dict[str, PricingUpdate] = {}
    for row in (data.get("data") or []):
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        spec = row.get("model_spec") or {}
        pricing = spec.get("pricing") or {}
        in_usd = ((pricing.get("input") or {}) if isinstance(pricing, dict) else {}).get("usd")
        out_usd = ((pricing.get("output") or {}) if isinstance(pricing, dict) else {}).get("usd")
        if not model_id or in_usd is None or out_usd is None:
            continue
        out[str(model_id)] = PricingUpdate(
            pricing=Pricing(
                input_per_1k=_per_1k_from_per_1m(float(in_usd)),
                output_per_1k=_per_1k_from_per_1m(float(out_usd)),
                currency="USD",
            ),
            source="venice:/models",
        )
    return out


def fetch_openrouter_pricing(base_url: str) -> dict[str, PricingUpdate]:
    """Return {model_id: PricingUpdate} for OpenRouter models.

    OpenRouter returns pricing under `pricing.{prompt,completion}` (USD per token).
    """
    url = f"{_base(base_url)}/models"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data") or []
    out: dict[str, PricingUpdate] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        pricing = row.get("pricing") or {}
        if not isinstance(pricing, dict):
            continue
        prompt = pricing.get("prompt")
        completion = pricing.get("completion")
        if not model_id or prompt is None or completion is None:
            continue
        out[str(model_id)] = PricingUpdate(
            pricing=Pricing(
                input_per_1k=_per_1k_from_per_token(prompt),
                output_per_1k=_per_1k_from_per_token(completion),
                currency="USD",
            ),
            source="openrouter:/models",
        )
    return out


def sync_pricing_for_endpoint(
    *,
    base_url: str,
    auth_env: str,
) -> dict[str, PricingUpdate]:
    """Fetch pricing map for a specific provider base_url."""
    b = _base(base_url)
    if "venice.ai/api" in b:
        key = os.environ.get(auth_env, "")
        if not key:
            raise RuntimeError(f"missing {auth_env} for Venice pricing sync")
        return fetch_venice_pricing(b, api_key=key)
    if "openrouter.ai/api" in b:
        return fetch_openrouter_pricing(b)
    raise RuntimeError(f"unsupported pricing provider for base_url={base_url!r}")

