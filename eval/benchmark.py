from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from connectors.dotenv import load_env
from connectors.pool_resolver import resolve_pool
from coordinator.checkpoint import checkpoint_type, load_checkpoint
from coordinator.factory import load_role_coordinator
from coordinator.train import _heuristic_weights
from eval.live_loop import LiveCodingLoop, LiveLoopConfig, LiveLoopResult, RoleCoordinator, trinity_loop_config
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
    model_ids: list[str] = field(default_factory=list)
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
    pool_dir: str | Path | None,
    split: str,
    mode: str,
    limit: int | None = None,
    connector_id: str | None = None,
    reflect: bool = False,
    checkpoint: Path | None = None,
    loop_config: LiveLoopConfig | None = None,
) -> BenchmarkSummary:
    res = resolve_pool(pool_dir=pool_dir) if pool_dir is not None else resolve_pool()
    pool = res.pool
    if not pool:
        raise ValueError(f"no connectors (pool_source={res.source})")

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
        kind = checkpoint_type(checkpoint)
        style = "slm" if kind == "slm_linear" else None
        coordinator = load_role_coordinator(checkpoint=checkpoint, style=style)
    config = loop_config or LiveLoopConfig()
    loop = LiveCodingLoop(pool, coordinator=coordinator, config=config)
    rows: list[BenchmarkRow] = []

    for i, task in enumerate(tasks):
        print(f"benchmark {mode}: task {i + 1}/{len(tasks)} {task.id}", flush=True)
        try:
            result = _run_task(loop, pool, task, mode=mode, connector_id=connector_id, reflect=reflect)
        except Exception as exc:
            print(f"benchmark {mode}: task {task.id} ERROR {type(exc).__name__}: {exc}", flush=True)
            result = LiveLoopResult(
                task_id=task.id,
                passed=False,
                code="",
                turns=[],
                steps=0,
                revisions=0,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                connector_ids=[connector_id] if connector_id else [],
                stages=["error"],
                error=str(exc),
            )
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
                model_ids=[t.completion.model for t in result.turns],
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
    per_question_best = _per_question_best(summaries)
    out: dict = {"summaries": [asdict(s) for s in summaries]}
    if per_question_best is not None:
        out["per_question_best_pass_at_1"] = per_question_best
    if best_single and orchestrated:
        out["delta_pass_at_1"] = orchestrated.pass_at_1 - best_single.pass_at_1
        out["cost_ratio_vs_best_single"] = (
            orchestrated.mean_cost_usd / best_single.mean_cost_usd
            if best_single.mean_cost_usd > 0
            else None
        )
        out["best_single_mode"] = best_single.mode
        if per_question_best is not None:
            out["gap_to_union"] = per_question_best - orchestrated.pass_at_1
    return out


def _per_question_best(summaries: list[BenchmarkSummary]) -> float | None:
    """Union pass rate: any single/reflect run passed the task (Trinity ceiling)."""
    singles = [s for s in summaries if s.mode in ("single", "single_reflect")]
    if not singles:
        return None
    by_task: dict[str, bool] = {}
    for summary in singles:
        for row in summary.rows:
            by_task[row.task_id] = by_task.get(row.task_id, False) or row.passed
    if not by_task:
        return None
    return sum(1 for passed in by_task.values() if passed) / len(by_task)


def _checkpoint_is_trained(path: Path) -> bool:
    """True if checkpoint came from a real training run, not heuristic warm-start."""
    raw = json.loads(path.read_text())
    if raw.get("type") == "slm_linear":
        task_limit = int(raw.get("task_limit") or 0)
        generations = int(raw.get("generations") or 0)
        return task_limit >= 5 and generations >= 3
    weights = np.array(raw.get("weights", []), dtype=float)
    if weights.size and not np.allclose(weights, _heuristic_weights()):
        return True
    task_limit = int(raw.get("task_limit") or raw.get("tasks") or 0)
    generations = int(raw.get("generations") or 0)
    return task_limit >= 8 and generations >= 6


def _require_trained_checkpoint(path: Path | None, *, force: bool) -> None:
    if not path or not path.exists():
        return
    if force or _checkpoint_is_trained(path):
        return
    raise SystemExit(
        f"checkpoint {path} is untrained (heuristic weights only). "
        "Run mat-train-live or mat-train-live-slm first, or pass --allow-untrained-checkpoint."
    )


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
    load_env()
    parser = argparse.ArgumentParser(
        description="Honest live coding benchmark (HumanEval val — never train on this for fitness)"
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="legacy connector directory override (bypasses active.yaml selection)",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--mode",
        choices=("orchestrated", "single", "single_reflect", "compare", "union"),
        default="compare",
    )
    parser.add_argument("--connector", help="connector id for single modes")
    parser.add_argument("--limit", type=int, help="cap tasks (smoke tests)")
    parser.add_argument("--checkpoint", type=Path, help="trained coordinator for orchestrated mode")
    parser.add_argument(
        "--allow-untrained-checkpoint",
        action="store_true",
        help="run orchestrated with heuristic weights (usually pointless)",
    )
    parser.add_argument("--out", type=Path, help="write JSON report")
    args = parser.parse_args()

    pool_dir = args.pool
    ckpt = args.checkpoint
    if args.mode in ("orchestrated", "compare"):
        _require_trained_checkpoint(ckpt, force=args.allow_untrained_checkpoint)

    if args.mode == "union":
        pool = resolve_pool(pool_dir=pool_dir).pool if pool_dir is not None else resolve_pool().pool
        summaries = []
        for conn in pool:
            print(f"benchmark union: single {conn.id}", flush=True)
            summaries.append(
                run_benchmark(
                    pool_dir=pool_dir,
                    split=args.split,
                    mode="single",
                    limit=args.limit,
                    connector_id=conn.id,
                )
            )
            ceiling = _per_question_best(summaries)
            partial = {
                "mode": "union",
                "per_question_best_pass_at_1": ceiling,
                "connectors_done": len(summaries),
                "summaries": [asdict(s) for s in summaries],
            }
            if args.out:
                args.out.write_text(json.dumps(partial, indent=2) + "\n")
                print(f"benchmark union: checkpoint written {args.out}", flush=True)
        report = partial
    elif args.mode == "compare":
        summaries = []
        pool = resolve_pool(pool_dir=pool_dir).pool if pool_dir is not None else resolve_pool().pool
        baseline_ids = [args.connector] if args.connector else [c.id for c in pool]
        # Trinity-fair turn budget for reflect + orchestrated paths.
        reflect_config = trinity_loop_config(use_thinker=True)
        orch_config = trinity_loop_config(use_thinker=bool(ckpt))
        for conn_id in baseline_ids:
            print(f"benchmark compare: single {conn_id}", flush=True)
            summaries.append(
                run_benchmark(
                    pool_dir=pool_dir,
                    split=args.split,
                    mode="single",
                    limit=args.limit,
                    connector_id=conn_id,
                )
            )
            print(f"benchmark compare: single_reflect {conn_id}", flush=True)
            summaries.append(
                run_benchmark(
                    pool_dir=pool_dir,
                    split=args.split,
                    mode="single_reflect",
                    limit=args.limit,
                    connector_id=conn_id,
                    loop_config=reflect_config,
                )
            )
        print("benchmark compare: orchestrated", flush=True)
        summaries.append(
            run_benchmark(
                pool_dir=pool_dir,
                split=args.split,
                mode="orchestrated",
                limit=args.limit,
                checkpoint=ckpt,
                loop_config=orch_config,
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
