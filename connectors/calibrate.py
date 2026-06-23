"""Calibrate connector capability scores from live worker-only eval."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, date, datetime
from pathlib import Path

from connectors.dotenv import load_env
from connectors.efficiency import (
    DEFAULT_REFERENCE_OUTPUT_TOKENS,
    speed_tier_from_efficiency,
    token_efficiency_score,
)
from connectors.loader import dump_connector, load_connector, load_connectors_dir
from connectors.paths import default_pool_dir
from connectors.schema import BenchmarkAttestation, CapabilityDim, Connector
from eval.live_loop import LiveCodingLoop
from eval.oracle import load_tasks


def calibrate_connector(
    connector: Connector,
    tasks: list,
    *,
    blend: float = 0.5,
    reference_output_tokens: float = DEFAULT_REFERENCE_OUTPUT_TOKENS,
) -> tuple[Connector, dict]:
    """Blend AA coding score with measured pass rate and record token efficiency."""
    loop = LiveCodingLoop([connector])
    passed = 0
    output_tokens: list[int] = []
    for task in tasks:
        result = loop.run_single(connector, task, reflect=False)
        output_tokens.append(result.output_tokens)
        if result.passed:
            passed += 1
    measured = passed / len(tasks) if tasks else 0.0
    median_out = float(statistics.median(output_tokens)) if output_tokens else 0.0
    efficiency = token_efficiency_score(median_out, reference=reference_output_tokens)

    updated = connector.model_copy(deep=True)
    tag = "coding"
    prior = updated.capabilities[tag].score
    blended = blend * measured + (1.0 - blend) * prior
    updated.capabilities[tag] = CapabilityDim.from_score(blended)

    updated.speed.median_output_tokens = round(median_out, 1)
    updated.speed.token_efficiency = round(efficiency, 4)
    if updated.speed.tokens_per_sec is None:
        updated.speed.tier = speed_tier_from_efficiency(efficiency)

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
    updated.benchmarks.append(
        BenchmarkAttestation(
            source="mat_calibration",
            metric="humaneval_worker_median_output_tokens",
            value=round(median_out, 1),
            unit="tokens",
            as_of=date.today(),
            notes=f"token_efficiency={efficiency:.3f} vs ref={reference_output_tokens}",
        )
    )
    updated.profile.notes = (
        (updated.profile.notes or "")
        + f" Calibrated {datetime.now(UTC).date()}: coding {prior:.3f}→{blended:.3f} "
        f"(pass@1={measured:.3f}); median_output_tokens={median_out:.0f} "
        f"token_efficiency={efficiency:.3f}."
    ).strip()
    report = {
        "measured_pass_at_1": measured,
        "median_output_tokens": median_out,
        "token_efficiency": efficiency,
        "coding_score": blended,
    }
    return updated, report


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(
        description="Calibrate installed connector coding score from live worker eval (train split)"
    )
    parser.add_argument("--connector", required=True, help="connector id")
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--blend", type=float, default=0.5, help="weight on measured vs AA prior")
    parser.add_argument(
        "--reference-output-tokens",
        type=float,
        default=DEFAULT_REFERENCE_OUTPUT_TOKENS,
        help="reference median output tokens for efficiency score (1.0 = this concise)",
    )
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

    updated, measured = calibrate_connector(
        connector,
        tasks,
        blend=args.blend,
        reference_output_tokens=args.reference_output_tokens,
    )
    report = {
        "connector_id": updated.id,
        "tasks": len(tasks),
        "split": args.split,
        **measured,
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
