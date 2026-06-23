"""Keep one local model per LM Studio endpoint to avoid load/unload churn."""

from __future__ import annotations

import os
from collections import defaultdict

from connectors.schema import Connector

LocalEndpointKey = tuple[str, str]


def local_endpoint_key(connector: Connector) -> LocalEndpointKey:
    return (connector.endpoint.base_url.rstrip("/"), connector.endpoint.auth_env or "")


def locals_by_endpoint(pool: list[Connector]) -> dict[LocalEndpointKey, list[Connector]]:
    groups: dict[LocalEndpointKey, list[Connector]] = defaultdict(list)
    for connector in pool:
        if connector.locality == "local":
            groups[local_endpoint_key(connector)].append(connector)
    return dict(groups)


def local_primary_from_env() -> str | None:
    raw = os.environ.get("MAT_LOCAL_PRIMARY", "").strip()
    return raw or None


def resolve_local_pins(
    pool: list[Connector],
    *,
    primary_id: str | None = None,
) -> dict[LocalEndpointKey, Connector]:
    """Pick one local connector per shared endpoint (e.g. one LM Studio server)."""
    primary_id = primary_id or local_primary_from_env()
    pins: dict[LocalEndpointKey, Connector] = {}
    for key, group in locals_by_endpoint(pool).items():
        if len(group) == 1:
            pins[key] = group[0]
            continue
        if primary_id:
            chosen = next((c for c in group if c.id == primary_id), None)
            if chosen is not None:
                pins[key] = chosen
                continue
        pins[key] = max(
            group,
            key=lambda c: (c.capabilities["coding"].score, c.id),
        )
    return pins


def apply_local_pin(
    connector: Connector,
    pins: dict[LocalEndpointKey, Connector],
) -> Connector:
    if connector.locality != "local":
        return connector
    return pins.get(local_endpoint_key(connector), connector)


def routing_pool(
    pool: list[Connector],
    *,
    primary_id: str | None = None,
) -> list[Connector]:
    """Drop extra locals on the same endpoint; keep all API connectors."""
    pins = resolve_local_pins(pool, primary_id=primary_id)
    if not pins:
        return list(pool)
    pinned_ids = {c.id for c in pins.values()}
    out: list[Connector] = []
    for connector in pool:
        if connector.locality != "local":
            out.append(connector)
        elif connector.id in pinned_ids:
            out.append(connector)
    return out
