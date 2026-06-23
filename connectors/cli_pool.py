"""Inspect and validate the installed connector pool."""

from __future__ import annotations

import argparse
from pathlib import Path

from connectors.discover_lmstudio import DEFAULT_LMSTUDIO_URL, sync_pool_lmstudio_names
from connectors.lmstudio_api import (
    clear_served_model_cache,
    fetch_served_model_ids,
    is_lmstudio_url,
)
from connectors.loader import load_connector
from connectors.paths import default_pool_dir, is_example_connector


def cmd_list(pool_dir: Path) -> int:
    if not pool_dir.exists() or not any(pool_dir.glob("*.yaml")):
        print(f"no connectors in {pool_dir}")
        print("run: mat-discover-lmstudio  OR  mat-import-aa <slug> --local")
        return 1
    for path in sorted(pool_dir.glob("*.yaml")):
        c = load_connector(path)
        method = c.profile.profile_method
        coding = c.capabilities["coding"].score
        bench = len(c.benchmarks)
        model = c.endpoint.model_name
        print(
            f"{c.id:40}  {method:16}  coding={coding:.2f}  "
            f"model={model!r}  benchmarks={bench}  {path.name}"
        )
    return 0


def cmd_verify(pool_dir: Path) -> int:
    errors = 0
    lmstudio_urls: set[str] = set()
    for path in sorted(pool_dir.glob("*.yaml")):
        if is_example_connector(path):
            print(f"WARN  {path.name}: lives under examples — not for production pool")
            errors += 1
            continue
        c = load_connector(path)
        if c.profile.profile_method == "hand":
            print(f"WARN  {c.id}: hand-scored — import AA benchmarks before routing")
            errors += 1
        if not c.benchmarks and c.profile.profile_method == "benchmark_import":
            print(f"WARN  {c.id}: benchmark_import but no attestations")
            errors += 1
        if is_lmstudio_url(c.endpoint.base_url):
            lmstudio_urls.add(c.endpoint.base_url.rstrip("/"))

    for base in sorted(lmstudio_urls):
        clear_served_model_cache()
        try:
            served = set(fetch_served_model_ids(base))
        except OSError as exc:
            print(f"ERR   LM Studio unreachable at {base}: {exc}")
            errors += 1
            continue
        if not served:
            print(f"ERR   LM Studio at {base} returned no chat models")
            errors += 1
            continue
        for path in sorted(pool_dir.glob("*.yaml")):
            c = load_connector(path)
            if c.endpoint.base_url.rstrip("/") != base:
                continue
            if c.endpoint.model_name in served:
                print(f"OK    {c.id}: model {c.endpoint.model_name!r} served")
            else:
                print(
                    f"ERR   {c.id}: model {c.endpoint.model_name!r} not in /v1/models "
                    f"(load in LM Studio or mat-pool sync-lmstudio)"
                )
                errors += 1

    if errors == 0:
        print("pool OK")
    return errors


def cmd_sync_lmstudio(pool_dir: Path, *, base_url: str) -> int:
    try:
        changes = sync_pool_lmstudio_names(pool_dir, base_url=base_url)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"sync failed: {exc}")
        return 1
    if not changes:
        print("model names already match LM Studio /v1/models")
        return 0
    for connector_id, old, new in changes:
        print(f"updated {connector_id}: {old!r} -> {new!r}")
    return 0


def cmd_lmstudio_models(base_url: str) -> int:
    clear_served_model_cache()
    try:
        served = fetch_served_model_ids(base_url)
    except OSError as exc:
        print(f"cannot reach LM Studio at {base_url}: {exc}")
        return 1
    if not served:
        print("no chat models listed (load models in LM Studio)")
        return 1
    print(f"LM Studio {base_url} serves {len(served)} chat model(s):")
    for model_id in served:
        print(f"  {model_id}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="List or verify installed connector pool")
    parser.add_argument(
        "command",
        choices=("list", "verify", "sync-lmstudio", "lmstudio-models"),
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_URL)
    args = parser.parse_args()
    pool = args.pool or default_pool_dir()
    if args.command == "list":
        raise SystemExit(cmd_list(pool))
    if args.command == "verify":
        raise SystemExit(cmd_verify(pool))
    if args.command == "sync-lmstudio":
        raise SystemExit(cmd_sync_lmstudio(pool, base_url=args.base_url))
    raise SystemExit(cmd_lmstudio_models(args.base_url))


if __name__ == "__main__":
    main()
