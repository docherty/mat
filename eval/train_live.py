"""Train coordinator weights on live orchestration rollouts (train split only)."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"cma\.s")

import cma  # noqa: E402
import numpy as np  # noqa: E402

from connectors.loader import load_connectors_dir  # noqa: E402
from connectors.paths import default_pool_dir  # noqa: E402
from coordinator.checkpoint import save_checkpoint  # noqa: E402
from coordinator.policy import TrainedCoordinator  # noqa: E402
from coordinator.train import _heuristic_weights  # noqa: E402
from eval.live_loop import LiveCodingLoop, RoleCoordinator  # noqa: E402
from eval.oracle import load_tasks  # noqa: E402
from workers.mock import MockLLMWorker  # noqa: E402


def train_live(
    pool_dir: str | Path,
    *,
    task_limit: int = 10,
    generations: int = 8,
    population: int = 8,
    seed: int = 42,
    mock: bool = False,
) -> dict:
    pool = load_connectors_dir(pool_dir)
    train_path = Path(__file__).parent / "tasks" / "humaneval_train.json"
    if not train_path.exists():
        raise FileNotFoundError("run: python -m eval.datasets.build_humaneval_split")
    tasks = load_tasks(train_path, split="train")[:task_limit]
    worker = MockLLMWorker() if mock else None

    def fitness(flat: list[float]) -> float:
        coord = TrainedCoordinator(np.array(flat))
        role_coord = RoleCoordinator(coord)
        scored_loop = LiveCodingLoop(pool, role_coord, worker=worker)
        passed = 0
        cost = 0.0
        for task in tasks:
            result = scored_loop.run_orchestrated(task)
            passed += int(result.passed)
            cost += result.estimated_cost_usd
        n = len(tasks) or 1
        return passed / n - 0.001 * cost

    x0 = _heuristic_weights()
    opts = cma.CMAOptions()
    opts.set("popsize", population)
    opts.set("verbose", -9)
    opts.set("seed", seed)
    es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
    es.optimize(fitness, iterations=generations)
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
    parser = argparse.ArgumentParser(description="Live CMA-ES on HumanEval train split only")
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--tasks", type=int, default=5, help="train tasks per fitness eval")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="use MockLLMWorker (no API calls)")
    parser.add_argument("--out", type=Path, help="write coordinator checkpoint JSON")
    parser.add_argument("--checkpoint", type=Path, help="alias for --out")
    args = parser.parse_args()
    out_path = args.checkpoint or args.out
    result = train_live(
        args.pool or default_pool_dir(),
        task_limit=args.tasks,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
        mock=args.mock,
    )
    serializable = {k: v for k, v in result.items() if k != "coordinator"}
    print(json.dumps(serializable, indent=2))
    if out_path:
        save_checkpoint(
            result["coordinator"],
            out_path,
            meta={k: serializable[k] for k in ("train_pass_at_1", "seed", "generations")},
        )
        print(f"checkpoint: {out_path}")


if __name__ == "__main__":
    main()
