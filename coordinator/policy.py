from __future__ import annotations

import numpy as np

from connectors.schema import Connector
from eval.harness import best_connector_for_task, connector_score
from eval.oracle import Task


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class PromptedCoordinator:
    """Rule-based baseline: pick highest weighted capability score per task."""

    def pick(self, task: Task, pool: list[Connector], *, transcript: str = "") -> Connector:
        return best_connector_for_task(task, pool)


class TrainedCoordinator:
    """Linear policy over task features + connector profile (shared weights).

    One weight vector over capability-space — not per-connector — so the policy
    generalises to held-out connectors the trainer never saw.
    """

    def __init__(self, weights: np.ndarray, n_task_features: int = 3, n_cap: int = 6):
        self.weights = weights
        self.n_task_features = n_task_features
        self.n_cap = n_cap

    @property
    def dim(self) -> int:
        return self.n_task_features + self.n_cap

    def _features(self, task: Task, connector: Connector, *, transcript: str = "") -> np.ndarray:
        from coordinator.features import encode_task

        base = np.array(encode_task(task), dtype=float)
        # Trinity conditions on conversation length / failure signals, not only static task stats.
        base[1] = min(1.0, (len(task.prompt) + len(transcript)) / 2000.0)
        if "REVISE" in transcript.upper():
            base[2] = min(3.0, base[2] + 1.0)
        return np.concatenate([base, np.array(connector.capability_vector())])

    def score(self, task: Task, connector: Connector, *, transcript: str = "") -> float:
        # Task-weighted capability fit (same signal as harness routing) plus a small
        # learned residual so CMA can nudge without overriding tag fit entirely.
        base = connector_score(task, connector)
        x = self._features(task, connector, transcript=transcript)
        w = self.weights[: len(x)]
        residual = float(np.dot(w, x[: len(w)]))
        return base + 0.05 * residual

    def route_scores(self, task: Task, pool: list[Connector], *, transcript: str = "") -> np.ndarray:
        return np.array([self.score(task, c, transcript=transcript) for c in pool])

    def pick(self, task: Task, pool: list[Connector], *, transcript: str = "") -> Connector:
        scores = self.route_scores(task, pool, transcript=transcript)
        return pool[int(np.argmax(scores))]

    def routing_accuracy(self, tasks: list[Task], pool: list[Connector]) -> float:
        if not tasks:
            return 0.0
        hits = sum(1 for t in tasks if self.pick(t, pool).id == best_connector_for_task(t, pool).id)
        return hits / len(tasks)

    @classmethod
    def from_flat(cls, flat: np.ndarray) -> TrainedCoordinator:
        return cls(flat)

    def score_connector_alignment(self, task: Task, pool: list[Connector]) -> float:
        probs = softmax(self.route_scores(task, pool))
        best = best_connector_for_task(task, pool)
        best_idx = next(i for i, c in enumerate(pool) if c.id == best.id)
        return float(probs[best_idx])
