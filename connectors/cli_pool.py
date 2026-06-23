"""Inspect and validate the installed connector pool."""

from __future__ import annotations

import argparse
from pathlib import Path

from connectors.discover_lmstudio import DEFAULT_LMSTUDIO_URL, sync_pool_lmstudio_names
from connectors.dotenv import load_env
from connectors.lmstudio_api import (
    clear_served_model_cache,
    fetch_served_model_ids,
    is_lmstudio_url,
)
from connectors.loader import load_connector
from connectors.paths import default_pool_dir, is_example_connector
from connectors.pool_curated import apply_curated_pool
from connectors.provider_pricing import sync_pricing_for_endpoint


def cmd_rehash(pool_dir: Path) -> int:
    """Rewrite integrity_sha256 for all connectors in the pool dir."""
    from connectors.loader import dump_connector

    rewritten = 0
    for path in sorted(pool_dir.glob("*.yaml")):
        # Ignore existing hash; re-save with current schema + hash.
        c = load_connector(path, check_hash=False)
        dump_connector(c, path)
        rewritten += 1
    print(f"rehash OK ({rewritten} connector(s))")
    return 0


def cmd_sync_pricing(pool_dir: Path) -> int:
    """Populate/refresh pricing for API connectors (Venice/OpenRouter)."""
    from connectors.loader import dump_connector

    # group by (base_url, auth_env) so we fetch each provider listing once
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(pool_dir.glob("*.yaml")):
        c = load_connector(path)
        if c.locality != "api":
            continue
        key = (c.endpoint.base_url.rstrip("/"), c.endpoint.auth_env)
        groups.setdefault(key, []).append(path)

    updated = 0
    for (base_url, auth_env), paths in sorted(groups.items()):
        try:
            pricing_map = sync_pricing_for_endpoint(base_url=base_url, auth_env=auth_env)
        except Exception as exc:
            print(f"WARN  pricing sync failed for {base_url}: {exc}")
            continue

        for path in paths:
            c = load_connector(path, check_hash=False)
            info = pricing_map.get(c.endpoint.model_name)
            if not info:
                print(f"WARN  {c.id}: no pricing for model_name={c.endpoint.model_name!r}")
                continue
            if c.pricing == info.pricing:
                continue
            c.pricing = info.pricing
            c.profile.notes = ((c.profile.notes or "") + f" pricing={info.source}.").strip()
            dump_connector(c, path)
            print(
                f"OK    {c.id}: pricing in={c.pricing.input_per_1k:.6g}/1k "
                f"out={c.pricing.output_per_1k:.6g}/1k ({info.source})"
            )
            updated += 1

    print(f"sync-pricing OK ({updated} connector(s) updated)")
    return 0


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
        eff = c.speed.token_efficiency
        eff_s = f"  eff={eff:.2f}" if eff is not None else ""
        print(
            f"{c.id:40}  {method:16}  coding={coding:.2f}{eff_s}  "
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


def cmd_apply(pool_dir: Path, curated_path: Path) -> int:
    if not curated_path.exists():
        print(f"curated file not found: {curated_path}")
        return 1
    pool_dir.mkdir(parents=True, exist_ok=True)
    kept, removed, missing = apply_curated_pool(pool_dir, curated_path)
    for cid in removed:
        print(f"removed {cid}")
    for cid in kept:
        print(f"kept    {cid}")
    if missing:
        print("missing (run mat-discover-lmstudio --curated … then apply again):")
        for cid in missing:
            print(f"  {cid}")
        return 1
    print(f"pool has {len(kept)} connector(s)")
    return 0


def _parse_curated_arg(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    curated = args.curated
    if curated:
        return curated
    return Path(__file__).resolve().parent / "curated" / "local-dev-pool.yaml"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="List or verify installed connector pool")
    parser.add_argument(
        "command",
        choices=(
            "list",
            "verify",
            "sync-lmstudio",
            "lmstudio-models",
            "apply",
            "rehash",
            "sync-pricing",
        ),
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument(
        "--curated",
        type=Path,
        help="curated pool yaml (for apply); default: connectors/curated/local-dev-pool.yaml",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="connector id to keep when applying curated pool (repeatable)",
    )
    args = parser.parse_args()
    pool = args.pool or default_pool_dir()
    if args.command == "list":
        raise SystemExit(cmd_list(pool))
    if args.command == "verify":
        raise SystemExit(cmd_verify(pool))
    if args.command == "sync-lmstudio":
        raise SystemExit(cmd_sync_lmstudio(pool, base_url=args.base_url))
    if args.command == "apply":
        curated = _parse_curated_arg(parser, args)
        # If you want to keep an extra connector (e.g. Venice) while applying a curated local pool,
        # append it to a temporary curated list.
        if args.keep:
            from connectors.pool_curated import load_curated_ids

            ids = load_curated_ids(curated) + list(args.keep)
            tmp = curated.parent / f".{curated.stem}.tmp.yaml"
            tmp.write_text("connectors:\n" + "".join(f"  - {cid}\n" for cid in ids))
            curated = tmp
        raise SystemExit(cmd_apply(pool, curated))
    if args.command == "rehash":
        raise SystemExit(cmd_rehash(pool))
    if args.command == "sync-pricing":
        raise SystemExit(cmd_sync_pricing(pool))
    raise SystemExit(cmd_lmstudio_models(args.base_url))


if __name__ == "__main__":
    main()
