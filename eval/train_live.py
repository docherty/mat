"""Train coordinator weights on live orchestration rollouts (train split only)."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"cma\.s")

import cma  # noqa: E402
import numpy as np  # noqa: E402

from connectors.dotenv import load_env  # noqa: E402
from connectors.pool_resolver import resolve_pool  # noqa: E402
from coordinator.checkpoint import save_checkpoint  # noqa: E402
from coordinator.policy import TrainedCoordinator  # noqa: E402
from coordinator.train import _heuristic_weights  # noqa: E402
from eval.live_loop import LiveCodingLoop, RoleCoordinator  # noqa: E402
from eval.oracle import load_tasks  # noqa: E402
from workers.mock import MockLLMWorker  # noqa: E402


def train_live(
    pool_dir: str | Path | None,
    *,
    task_limit: int = 10,
    generations: int = 8,
    population: int = 8,
    seed: int = 42,
    mock: bool = False,
    replicates: int = 1,
) -> dict:
    pool = resolve_pool(pool_dir=pool_dir).pool if pool_dir is not None else resolve_pool().pool
    train_path = Path(__file__).parent / "tasks" / "humaneval_train.json"
    if not train_path.exists():
        raise FileNotFoundError("run: python -m eval.datasets.build_humaneval_split")
    tasks = load_tasks(train_path, split="train")[:task_limit]
    worker = MockLLMWorker() if mock else None
    print(
        f"train-live: {len(tasks)} tasks × pop {population} × {generations} gens "
        f"({len(pool)} connectors)",
        flush=True,
    )

    eval_count = 0

    def fitness(flat: list[float]) -> float:
        nonlocal eval_count
        eval_count += 1
        coord = TrainedCoordinator(np.array(flat))
        role_coord = RoleCoordinator(coord)
        scored_loop = LiveCodingLoop(pool, role_coord, worker=worker)
        scores: list[float] = []
        for _rep in range(max(1, replicates)):
            passed = 0
            cost = 0.0
            tokens = 0
            for task in tasks:
                result = scored_loop.run_orchestrated(task)
                passed += int(result.passed)
                cost += result.estimated_cost_usd
                tokens += result.output_tokens
            n = len(tasks) or 1
            scores.append(passed / n - 0.001 * cost - 1e-6 * (tokens / n))
        score = sum(scores) / len(scores)
        print(
            f"train-live eval {eval_count} pass@{len(tasks)}={scores[0]:.2f}"
            + (f" (mean of {len(scores)} reps)" if len(scores) > 1 else ""),
            flush=True,
        )
        return score

    x0 = _heuristic_weights()
    opts = cma.CMAOptions()
    opts.set("popsize", population)
    opts.set("verbose", -9)
    opts.set("seed", seed)
    opts.set("CMA_diagonal", True)  # sep-CMA-ES — matches Trinity for high-dim heads
    es = cma.CMAEvolutionStrategy(x0, 0.3, opts)

    def _progress(_: cma.CMAEvolutionStrategy) -> None:
        gen = es.countiter
        best = float(es.result.fbest) if es.result.fbest is not None else float("nan")
        print(f"train-live gen {gen}/{generations} best_fitness={best:.4f}", flush=True)

    es.optimize(fitness, iterations=generations, callback=_progress)
    best = TrainedCoordinator(np.array(es.result.xbest))

    eval_loop = LiveCodingLoop(pool, RoleCoordinator(best), worker=worker)
    passed = sum(1 for t in tasks if eval_loop.run_orchestrated(t).passed)
    return {
        "train_pass_at_1": passed / len(tasks),
        "generations": generations,
        "population": population,
        "task_limit": task_limit,
        "weights": es.result.xbest.tolist(),
        "seed": seed,
        "coordinator": best,
    }


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Live CMA-ES on HumanEval train split only")
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--tasks", type=int, default=10, help="train tasks per fitness eval")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="use MockLLMWorker (no API calls)")
    parser.add_argument("--out", type=Path, help="write coordinator checkpoint JSON")
    parser.add_argument("--checkpoint", type=Path, help="alias for --out")
    parser.add_argument("--replicates", type=int, default=1, help="fitness replicates per candidate (Trinity uses ~16)")
    args = parser.parse_args()
    out_path = args.checkpoint or args.out
    result = train_live(
        args.pool,
        task_limit=args.tasks,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
        mock=args.mock,
        replicates=args.replicates,
    )
    serializable = {k: v for k, v in result.items() if k != "coordinator"}
    print(json.dumps(serializable, indent=2))
    if out_path:
        save_checkpoint(
            result["coordinator"],
            out_path,
            meta={k: serializable[k] for k in ("train_pass_at_1", "seed", "generations", "task_limit", "population")},
        )
        print(f"checkpoint: {out_path}")


if __name__ == "__main__":
    main()
