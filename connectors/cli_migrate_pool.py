from __future__ import annotations

import argparse
import re
from pathlib import Path

from connectors.dotenv import load_env
from connectors.loader import dump_connector, load_connector
from connectors.paths import REPO_ROOT, USER_POOL_DIR
from connectors.pool_resolver import default_active_manifest, default_library_dir


def _safe_filename(connector_id: str) -> str:
    # Keep filenames stable and readable.
    s = connector_id.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return f"{s}.yaml"


def migrate_pool(
    *,
    src_dir: Path,
    library_dir: Path,
    active_manifest: Path,
    select: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Copy legacy installed connectors into repo library and write active.yaml.

    Returns (copied_ids, active_ids).
    """
    library_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    ids: list[str] = []

    for p in sorted(src_dir.glob("*.yaml")):
        c = load_connector(p, check_hash=False)
        ids.append(c.id)
        out = library_dir / _safe_filename(c.id)
        dump_connector(c, out)
        copied.append(c.id)

    active_ids = select or ids
    active_manifest.write_text("connectors:\n" + "".join(f"  - {cid}\n" for cid in active_ids))
    return copied, active_ids


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        description="Migrate ~/.config/mat/connectors into repo library + active.yaml"
    )
    parser.add_argument("--src", type=Path, default=USER_POOL_DIR, help="source legacy pool dir")
    parser.add_argument(
        "--library", type=Path, default=default_library_dir(), help="destination library dir"
    )
    parser.add_argument(
        "--active", type=Path, default=default_active_manifest(), help="destination active manifest"
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="connector id to include in active.yaml (repeatable; default: all copied)",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source pool dir not found: {args.src}")

    copied, active = migrate_pool(
        src_dir=args.src,
        library_dir=args.library,
        active_manifest=args.active,
        select=list(args.select) if args.select else None,
    )
    print(f"copied {len(copied)} connector(s) into {args.library}")
    print(f"wrote active manifest {args.active} ({len(active)} connector(s))")
    print()
    print("Next:")
    print(f"  cd {REPO_ROOT}")
    print("  mat-pool list")
    print("  mat-pool verify")


if __name__ == "__main__":
    main()

