"""Calibrate connector capability scores from live worker-only eval."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from connectors.loader import dump_connector, load_connector, load_connectors_dir
from connectors.paths import default_pool_dir
from connectors.schema import BenchmarkAttestation, CapabilityDim, Connector
from eval.live_loop import LiveCodingLoop
from eval.oracle import load_tasks

CALIBRATION_METRICS = ("coding", "reasoning", "verification", "instruction_following")


def calibrate_connector(
    connector: Connector,
    tasks: list,
    *,
    blend: float = 0.5,
) -> Connector:
    """Blend AA benchmark scores with measured worker pass rate on coding tasks."""
    loop = LiveCodingLoop([connector])
    passed = 0
    for task in tasks:
        if loop.run_single(connector, task, reflect=False).passed:
            passed += 1
    measured = passed / len(tasks) if tasks else 0.0

    updated = connector.model_copy(deep=True)
    tag = "coding"
    prior = updated.capabilities[tag].score
    blended = blend * measured + (1.0 - blend) * prior
    updated.capabilities[tag] = CapabilityDim.from_score(blended)

    updated.benchmarks.append(
        BenchmarkAttestation(
            source="mat_calibration",
            metric="humaneval_worker_pass_rate",
            value=round(measured, 4),
            unit="ratio",
            as_of=date.today(),
            notes=f"blended into coding score at weight {blend}; n={len(tasks)} tasks",
        )
    )
    updated.profile.notes = (
        (updated.profile.notes or "")
        + f" Calibrated {datetime.now(UTC).date()}: coding {prior:.3f}→{blended:.3f} "
        f"(measured pass@1={measured:.3f}, blend={blend})."
    ).strip()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate installed connector coding score from live worker eval (train split)"
    )
    parser.add_argument("--connector", required=True, help="connector id")
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--blend", type=float, default=0.5, help="weight on measured vs AA prior")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pool_dir = args.pool or default_pool_dir()
    connector = next((c for c in load_connectors_dir(pool_dir) if c.id == args.connector), None)
    if connector is None:
        raise SystemExit(f"connector not found in {pool_dir}: {args.connector}")

    tasks_path = (
        Path(__file__).resolve().parents[1] / "eval" / "tasks" / f"humaneval_{args.split}.json"
    )
    tasks = load_tasks(tasks_path, split=args.split)[: args.limit]
    if args.split == "val":
        print("warning: calibrating on val split — prefer train to avoid leakage", flush=True)

    updated = calibrate_connector(connector, tasks, blend=args.blend)
    report = {
        "connector_id": updated.id,
        "coding_score": updated.capabilities["coding"].score,
        "tasks": len(tasks),
        "split": args.split,
    }
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        out: Path | None = None
        for p in pool_dir.glob("*.yaml"):
            if load_connector(p).id == updated.id:
                out = p
                break
        if out is None:
            out = pool_dir / f"{updated.id.split('@')[0]}.yaml"
        dump_connector(updated, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
