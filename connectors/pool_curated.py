"""Curated connector pool manifests."""

from __future__ import annotations

from pathlib import Path

import yaml

from connectors.loader import load_connector


def load_curated_ids(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        raw = data.get("connectors") or data.get("ids") or []
        return [str(x) for x in raw]
    raise ValueError(f"invalid curated pool file: {path}")


def apply_curated_pool(
    pool_dir: Path, curated_path: Path
) -> tuple[list[str], list[str], list[str]]:
    """Keep only connectors listed in curated_path. Returns (kept, removed, missing)."""
    keep = load_curated_ids(curated_path)
    keep_set = set(keep)
    removed: list[str] = []
    kept: list[str] = []
    for path in sorted(pool_dir.glob("*.yaml")):
        connector = load_connector(path)
        if connector.id in keep_set:
            kept.append(connector.id)
        else:
            path.unlink()
            removed.append(connector.id)
    missing = [cid for cid in keep if cid not in kept]
    return kept, removed, missing
