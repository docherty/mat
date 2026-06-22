from __future__ import annotations

import cma
import numpy as np

from connectors.schema import CAPABILITY_TAGS, Connector
from coordinator.policy import TrainedCoordinator
from eval.harness import EvalHarness


def _heuristic_weights() -> np.ndarray:
    """Warm-start: upweight coding + reasoning capability dimensions."""
    cap = np.zeros(len(CAPABILITY_TAGS))
    cap[CAPABILITY_TAGS.index("coding")] = 2.0
    cap[CAPABILITY_TAGS.index("reasoning")] = 1.0
    cap[CAPABILITY_TAGS.index("instruction_following")] = 0.5
    task = np.zeros(3)
    return np.concatenate([task, cap])


def train_coordinator(
    pool: list[Connector],
    *,
    generations: int = 40,
    population: int = 20,
    seed: int = 42,
    cost_per_step: float = 0.02,
) -> TrainedCoordinator:
    harness = EvalHarness(cost_per_step=cost_per_step)

    def fitness(flat: list[float]) -> float:
        coord = TrainedCoordinator(np.array(flat))
        metrics = harness.evaluate_routing(coord.pick, pool, seed=seed)
        # heavily weight pass@1; routing accuracy as tie-breaker
        acc = coord.routing_accuracy(harness.tasks, pool)
        return metrics.pass_at_1 + 0.01 * acc - cost_per_step * metrics.mean_steps

    x0 = _heuristic_weights()
    sigma = 0.3
    opts = cma.CMAOptions()
    opts.set("popsize", population)
    opts.set("verbose", -9)
    opts.set("seed", seed)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    es.optimize(fitness, iterations=generations)
    return TrainedCoordinator(np.array(es.result.xbest))
