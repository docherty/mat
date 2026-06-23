"""Pool health summaries for /health and mat-dashboard."""

from __future__ import annotations

from connectors.lmstudio_api import fetch_served_model_ids, is_lmstudio_url
from connectors.schema import Connector


def connector_summary(c: Connector, *, served: bool | None = None) -> dict:
    pricing = c.pricing
    return {
        "id": c.id,
        "display_name": c.display_name,
        "locality": c.locality,
        "model_name": c.endpoint.model_name,
        "base_url": c.endpoint.base_url,
        "modalities": list(c.modalities or ["text"]),
        "coding_score": round(c.capabilities["coding"].score, 3),
        "token_efficiency": c.speed.token_efficiency,
        "input_per_1k": pricing.input_per_1k if pricing else None,
        "output_per_1k": pricing.output_per_1k if pricing else None,
        "served": served,
    }


def build_pool_health(pool: list[Connector]) -> dict:
    """Return per-connector health including LM Studio served checks."""
    lm_bases: dict[str, set[str]] = {}
    for c in pool:
        base = c.endpoint.base_url.rstrip("/")
        if is_lmstudio_url(base):
            lm_bases.setdefault(base, set())

    served_cache: dict[str, set[str]] = {}
    for base in lm_bases:
        try:
            served_cache[base] = set(fetch_served_model_ids(base))
        except OSError:
            served_cache[base] = set()

    connectors: list[dict] = []
    issues: list[str] = []
    for c in pool:
        served: bool | None = None
        base = c.endpoint.base_url.rstrip("/")
        if is_lmstudio_url(base):
            served_set = served_cache.get(base, set())
            served = c.endpoint.model_name in served_set
            if not served:
                issues.append(f"{c.id}: model {c.endpoint.model_name!r} not served at {base}")
        if c.locality == "api" and not c.pricing:
            issues.append(f"{c.id}: missing pricing (run mat-pool sync-pricing)")
        connectors.append(connector_summary(c, served=served))

    status = "ok" if not issues else "degraded"
    return {
        "status": status,
        "connector_count": len(pool),
        "connectors": connectors,
        "issues": issues,
    }
