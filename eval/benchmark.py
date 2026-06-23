from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from connectors.loader import load_connectors_dir
from connectors.paths import default_pool_dir
from coordinator.checkpoint import load_checkpoint
from eval.live_loop import LiveCodingLoop, LiveLoopConfig, RoleCoordinator
from eval.oracle import Task, load_tasks


@dataclass
class BenchmarkRow:
    task_id: str
    mode: str
    passed: bool
    steps: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    connector_ids: list[str]
    error: str | None = None


@dataclass
class BenchmarkSummary:
    mode: str
    split: str
    tasks: int
    pass_at_1: float
    mean_steps: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    rows: list[BenchmarkRow] = field(default_factory=list)


def run_benchmark(
    *,
    pool_dir: str | Path,
    split: str,
    mode: str,
    limit: int | None = None,
    connector_id: str | None = None,
    reflect: bool = False,
    checkpoint: Path | None = None,
) -> BenchmarkSummary:
    pool = load_connectors_dir(pool_dir)
    if not pool:
        raise ValueError(f"no connectors in {pool_dir}")

    tasks_path = Path(__file__).parent / "tasks" / f"humaneval_{split}.json"
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"missing {tasks_path}; run: python -m eval.datasets.build_humaneval_split"
        )
    tasks = load_tasks(tasks_path, split=split)
    if limit:
        tasks = tasks[:limit]

    coordinator = None
    if checkpoint:
        coordinator = RoleCoordinator(load_checkpoint(checkpoint))
    loop = LiveCodingLoop(pool, coordinator=coordinator, config=LiveLoopConfig())
    rows: list[BenchmarkRow] = []

    for task in tasks:
        result = _run_task(loop, pool, task, mode=mode, connector_id=connector_id, reflect=reflect)
        rows.append(
            BenchmarkRow(
                task_id=result.task_id,
                mode=mode,
                passed=result.passed,
                steps=result.steps,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=result.estimated_cost_usd,
                connector_ids=result.connector_ids,
                error=result.error,
            )
        )

    n = len(rows) or 1
    passed = sum(1 for r in rows if r.passed)
    return BenchmarkSummary(
        mode=mode,
        split=split,
        tasks=len(rows),
        pass_at_1=passed / n,
        mean_steps=sum(r.steps for r in rows) / n,
        mean_input_tokens=sum(r.input_tokens for r in rows) / n,
        mean_output_tokens=sum(r.output_tokens for r in rows) / n,
        mean_cost_usd=sum(r.estimated_cost_usd for r in rows) / n,
        rows=rows,
    )


def _run_task(
    loop: LiveCodingLoop,
    pool: list,
    task: Task,
    *,
    mode: str,
    connector_id: str | None,
    reflect: bool,
):
    if mode == "orchestrated":
        return loop.run_orchestrated(task)
    if mode == "single":
        conn = _resolve_connector(pool, connector_id)
        return loop.run_single(conn, task, reflect=False)
    if mode == "single_reflect":
        conn = _resolve_connector(pool, connector_id)
        return loop.run_single(conn, task, reflect=True)
    raise ValueError(f"unknown mode: {mode}")


def _resolve_connector(pool, connector_id: str | None):
    if connector_id:
        for c in pool:
            if c.id == connector_id:
                return c
        raise ValueError(f"connector not in pool: {connector_id}")
    return pool[0]


def compare_summaries(summaries: list[BenchmarkSummary]) -> dict:
    if not summaries:
        return {}
    best_single = max(
        (s for s in summaries if s.mode in ("single", "single_reflect")),
        key=lambda s: s.pass_at_1,
        default=None,
    )
    orchestrated = next((s for s in summaries if s.mode == "orchestrated"), None)
    out: dict = {"summaries": [asdict(s) for s in summaries]}
    if best_single and orchestrated:
        out["delta_pass_at_1"] = orchestrated.pass_at_1 - best_single.pass_at_1
        out["cost_ratio_vs_best_single"] = (
            orchestrated.mean_cost_usd / best_single.mean_cost_usd
            if best_single.mean_cost_usd > 0
            else None
        )
        out["best_single_mode"] = best_single.mode
    return out


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
    parser = argparse.ArgumentParser(
        description="Honest live coding benchmark (HumanEval val — never train on this for fitness)"
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="connector directory (default: ~/.config/mat/connectors)",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--mode",
        choices=("orchestrated", "single", "single_reflect", "compare"),
        default="compare",
    )
    parser.add_argument("--connector", help="connector id for single modes")
    parser.add_argument("--limit", type=int, help="cap tasks (smoke tests)")
    parser.add_argument("--checkpoint", type=Path, help="trained coordinator for orchestrated mode")
    parser.add_argument("--out", type=Path, help="write JSON report")
    args = parser.parse_args()

    pool_dir = args.pool or default_pool_dir()
    ckpt = args.checkpoint

    if args.mode == "compare":
        summaries = []
        pool = load_connectors_dir(pool_dir)
        baseline_ids = [args.connector] if args.connector else [c.id for c in pool]
        for conn_id in baseline_ids:
            summaries.append(
                run_benchmark(
                    pool_dir=pool_dir,
                    split=args.split,
                    mode="single",
                    limit=args.limit,
                    connector_id=conn_id,
                )
            )
            summaries.append(
                run_benchmark(
                    pool_dir=pool_dir,
                    split=args.split,
                    mode="single_reflect",
                    limit=args.limit,
                    connector_id=conn_id,
                )
            )
        summaries.append(
            run_benchmark(
                pool_dir=pool_dir,
                split=args.split,
                mode="orchestrated",
                limit=args.limit,
                checkpoint=ckpt,
            )
        )
        report = compare_summaries(summaries)
    else:
        summary = run_benchmark(
            pool_dir=pool_dir,
            split=args.split,
            mode=args.mode,
            limit=args.limit,
            connector_id=args.connector,
            reflect=args.mode == "single_reflect",
            checkpoint=ckpt if args.mode == "orchestrated" else None,
        )
        report = {"summaries": [asdict(summary)]}

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
