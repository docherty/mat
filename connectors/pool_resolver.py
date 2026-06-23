from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from connectors.loader import load_connector, load_connectors_dir
from connectors.paths import EXAMPLES_DIR, INSTALLED_REPO_DIR, REPO_ROOT, default_pool_dir
from connectors.schema import Connector


@dataclass(frozen=True)
class PoolResolution:
    pool: list[Connector]
    source: str
    library_dir: Path | None = None
    active_manifest: Path | None = None
    library_index: dict[str, Path] | None = None


def default_library_dir() -> Path:
    # Keep YAMLs out of the Python package root; store them in a subdir.
    return REPO_ROOT / "connectors" / "library"


def default_active_manifest() -> Path:
    return REPO_ROOT / "active.yaml"


def load_active_ids(path: Path) -> list[str]:
    ids, _ = load_active_manifest(path)
    return ids


def load_active_manifest(path: Path) -> tuple[list[str], str | None]:
    data = yaml.safe_load(path.read_text())
    primary: str | None = None
    if isinstance(data, list):
        return [str(x) for x in data], None
    if isinstance(data, dict):
        raw = data.get("connectors") or data.get("ids") or []
        primary = data.get("local_primary")
        if primary is not None:
            primary = str(primary).strip() or None
        return [str(x) for x in raw], primary
    raise ValueError(f"invalid active manifest: {path}")


def _is_under(path: Path, parent: Path) -> bool:
    try:
        return parent in path.resolve().parents
    except OSError:
        return False


def index_library(connectors_dir: Path) -> dict[str, Path]:
    """Scan for connector YAMLs and map connector.id -> file path.

    Excludes repo examples and (optionally) the legacy installed dir.
    """
    out: dict[str, Path] = {}
    if not connectors_dir.exists():
        return out

    for p in sorted(connectors_dir.glob("**/*.yaml")):
        if _is_under(p, EXAMPLES_DIR) or _is_under(p, INSTALLED_REPO_DIR):
            continue
        c = load_connector(p)
        if c.id in out:
            raise ValueError(f"duplicate connector id {c.id!r}: {out[c.id]} and {p}")
        out[c.id] = p
    return out


def load_active_pool(*, library_dir: Path, manifest: Path) -> PoolResolution:
    ids, local_primary = load_active_manifest(manifest)
    index = index_library(library_dir)
    missing = [cid for cid in ids if cid not in index]
    if missing:
        raise ValueError(
            f"active manifest {manifest} references missing connector(s): {missing}. "
            f"Looked in library {library_dir}."
        )
    pool = [load_connector(index[cid]) for cid in ids]
    if local_primary and local_primary not in {c.id for c in pool}:
        raise ValueError(
            f"active manifest local_primary {local_primary!r} is not in connectors list"
        )
    if local_primary:
        os.environ.setdefault("MAT_LOCAL_PRIMARY", local_primary)
    return PoolResolution(
        pool=pool,
        source="active_manifest",
        library_dir=library_dir,
        active_manifest=manifest,
        library_index=index,
    )


def resolve_pool(
    *,
    pool_dir: str | Path | None = None,
    connectors_dir: str | Path | None = None,
    active_manifest: str | Path | None = None,
) -> PoolResolution:
    """Resolve which connectors are active for runtime use.

    Priority:
    - explicit pool_dir (legacy) -> load directory directly
    - MAT_ACTIVE_POOL / active manifest file -> select from library
    - fallback to legacy default_pool_dir()
    """
    if pool_dir is not None:
        d = Path(pool_dir)
        return PoolResolution(pool=load_connectors_dir(d), source=f"pool_dir:{d}")

    man = (
        Path(active_manifest)
        if active_manifest
        else Path(os.environ.get("MAT_ACTIVE_POOL", "")).expanduser()
    )
    if not active_manifest and not os.environ.get("MAT_ACTIVE_POOL"):
        man = default_active_manifest()

    lib = (
        Path(connectors_dir)
        if connectors_dir
        else Path(os.environ.get("MAT_CONNECTORS_DIR", "")).expanduser()
    )
    if not connectors_dir and not os.environ.get("MAT_CONNECTORS_DIR"):
        lib = default_library_dir()

    if man and str(man) and man.exists():
        return load_active_pool(library_dir=lib, manifest=man)

    # Legacy fallback: installed pool directory semantics
    legacy = Path(os.environ.get("MAT_POOL_DIR") or default_pool_dir())
    return PoolResolution(pool=load_connectors_dir(legacy), source=f"legacy_pool_dir:{legacy}")


def find_active_connector_path(res: PoolResolution, connector_id: str) -> Path | None:
    """Find the YAML file for a connector when using an active manifest."""
    if res.library_index is None:
        return None
    return res.library_index.get(connector_id)

