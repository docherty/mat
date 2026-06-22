"""Inspect and validate the installed connector pool."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        print(f"{c.id:40}  {method:16}  coding={coding:.2f}  benchmarks={bench}  {path.name}")
    return 0


def cmd_verify(pool_dir: Path) -> int:
    errors = 0
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
    if errors == 0:
        print("pool OK")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="List or verify installed connector pool")
    parser.add_argument("command", choices=("list", "verify"))
    parser.add_argument("--pool", type=Path, default=None)
    args = parser.parse_args()
    pool = args.pool or default_pool_dir()
    if args.command == "list":
        raise SystemExit(cmd_list(pool))
    raise SystemExit(cmd_verify(pool))


if __name__ == "__main__":
    main()
