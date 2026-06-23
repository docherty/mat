"""Train SLM coordinator head with sep-CMA-ES on live orchestration rollouts."""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"cma\.s")

import cma  # noqa: E402
import numpy as np  # noqa: E402

from connectors.dotenv import load_env  # noqa: E402
from connectors.pool_resolver import resolve_pool  # noqa: E402
from coordinator.checkpoint import save_slm_checkpoint  # noqa: E402
from coordinator.slm_coordinator import SLMCoordinator  # noqa: E402
from eval.live_loop import LiveCodingLoop, RoleCoordinator  # noqa: E402
from eval.oracle import load_tasks  # noqa: E402
from workers.mock import MockLLMWorker  # noqa: E402


def train_live_slm(
    pool_dir: str | Path | None,
    *,
    task_limit: int = 20,
    generations: int = 10,
    population: int = 8,
    seed: int = 42,
    mock: bool = False,
    parallel_workers: int = 4,
    task_workers: int = 2,
) -> dict:
    pool = resolve_pool(pool_dir=pool_dir).pool if pool_dir is not None else resolve_pool().pool
    train_path = Path(__file__).parent / "tasks" / "humaneval_train.json"
    tasks = load_tasks(train_path, split="train")[:task_limit]
    worker = MockLLMWorker() if mock else None

    base = SLMCoordinator.from_pretrained() if not mock else _mock_slm()
    backbone, tokenizer, config = base.backbone, base.tokenizer, base.config
    dim = base.dim
    print(
        f"train-live-slm: {len(tasks)} tasks pop={population} gens={generations} "
        f"dim={dim} model={config.model_id} parallel={parallel_workers}",
        flush=True,
    )

    eval_count = 0
    lock = __import__("threading").Lock()

    def _run_tasks(loop: LiveCodingLoop) -> tuple[int, float, int]:
        passed = 0
        cost = 0.0
        tokens = 0

        def one(task):
            try:
                return loop.run_orchestrated(task)
            except Exception as exc:
                print(f"train-live-slm task {task.id} ERROR {type(exc).__name__}: {exc}", flush=True)
                from eval.live_loop import LiveLoopResult

                return LiveLoopResult(
                    task_id=task.id,
                    passed=False,
                    code="",
                    turns=[],
                    steps=0,
                    revisions=0,
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
                    connector_ids=[],
                    stages=["error"],
                    error=str(exc),
                )

        if task_workers > 1 and not mock:
            with ThreadPoolExecutor(max_workers=task_workers) as ex:
                results = list(ex.map(one, tasks))
        else:
            results = [one(t) for t in tasks]
        for r in results:
            passed += int(r.passed)
            cost += r.estimated_cost_usd
            tokens += r.output_tokens
        return passed, cost, tokens

    def eval_candidate(flat: list[float]) -> float:
        nonlocal eval_count
        coord = SLMCoordinator(
            np.array(flat),
            config=config,
            backbone=backbone,
            tokenizer=tokenizer,
        )
        loop = LiveCodingLoop(pool, RoleCoordinator(coord), worker=worker)
        passed, cost, tokens = _run_tasks(loop)
        n = len(tasks) or 1
        score = passed / n - 0.001 * cost - 1e-6 * (tokens / n)
        with lock:
            eval_count += 1
            n_eval = eval_count
        print(f"train-live-slm eval {n_eval} pass@{n}={passed / n:.2f}", flush=True)
        return score

    x0 = np.zeros(dim)
    sigma = 0.15
    opts = cma.CMAOptions()
    opts.set("popsize", population)
    opts.set("verbose", -9)
    opts.set("seed", seed)
    opts.set("CMA_diagonal", True)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    for gen in range(generations):
        candidates = es.ask()
        if parallel_workers > 1:
            fitnesses: list[float] = [0.0] * len(candidates)
            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                futures = {ex.submit(eval_candidate, list(x)): i for i, x in enumerate(candidates)}
                for fut in as_completed(futures):
                    try:
                        fitnesses[futures[fut]] = fut.result()
                    except Exception as exc:
                        print(f"train-live-slm candidate ERROR {exc}", flush=True)
                        fitnesses[futures[fut]] = 0.0
        else:
            fitnesses = [eval_candidate(list(x)) for x in candidates]
        es.tell(candidates, fitnesses)
        print(
            f"train-live-slm gen {gen + 1}/{generations} best={float(es.result.fbest):.4f}",
            flush=True,
        )

    best = SLMCoordinator(
        np.array(es.result.xbest),
        config=config,
        backbone=backbone,
        tokenizer=tokenizer,
    )
    eval_loop = LiveCodingLoop(pool, RoleCoordinator(best), worker=worker)
    passed, _, _ = _run_tasks(eval_loop)
    return {
        "train_pass_at_1": passed / len(tasks),
        "generations": generations,
        "population": population,
        "task_limit": task_limit,
        "hidden_size": config.hidden_size,
        "model_id": config.model_id,
        "weights": es.result.xbest.tolist(),
        "seed": seed,
        "coordinator": best,
    }


def _mock_slm() -> SLMCoordinator:
    from coordinator.slm_coordinator import SLMCoordinatorConfig, slm_feature_dim

    cfg = SLMCoordinatorConfig(hidden_size=32)
    return SLMCoordinator(np.zeros(slm_feature_dim(32)), config=cfg)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Train SLM coordinator head (HumanEval train only)")
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4, help="parallel CMA candidates")
    parser.add_argument("--task-workers", type=int, default=2, help="parallel tasks per candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    out_path = args.checkpoint or args.out
    result = train_live_slm(
        args.pool,
        task_limit=args.tasks,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
        mock=args.mock,
        parallel_workers=args.parallel,
        task_workers=args.task_workers,
    )
    serializable = {k: v for k, v in result.items() if k != "coordinator"}
    print(json.dumps(serializable, indent=2))
    if out_path:
        save_slm_checkpoint(
            result["coordinator"],
            out_path,
            meta={k: serializable[k] for k in ("train_pass_at_1", "seed", "generations", "task_limit", "population", "model_id")},
        )
        print(f"checkpoint: {out_path}")


if __name__ == "__main__":
    main()
