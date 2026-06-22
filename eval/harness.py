from __future__ import annotations

from dataclasses import dataclass, field

from connectors.schema import Connector
from eval.oracle import Task, load_tasks, run_oracle
from traces.logger import TraceLogger


@dataclass
class RunMetrics:
    pass_count: int = 0
    total: int = 0
    total_steps: int = 0
    routing_hits: int = 0

    @property
    def pass_at_1(self) -> float:
        return self.pass_count / self.total if self.total else 0.0

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.total if self.total else 0.0


@dataclass
class EvalHarness:
    tasks: list[Task] = field(default_factory=load_tasks)
    cost_per_step: float = 0.05
    logger: TraceLogger | None = None

    def fitness(self, pass_at_1: float, mean_steps: float) -> float:
        return pass_at_1 - self.cost_per_step * mean_steps

    def evaluate_routing(
        self,
        pick_connector: callable,
        pool: list[Connector],
        *,
        seed: int | None = None,
    ) -> RunMetrics:
        metrics = RunMetrics()
        for task in self.tasks:
            metrics.total += 1
            connector = pick_connector(task, pool)
            code = synthesize_code(task, connector)
            result = run_oracle(task.prompt + code, task.tests)
            steps = step_budget(task, connector)
            metrics.total_steps += steps
            if result.passed:
                metrics.pass_count += 1
            best = best_connector_for_task(task, pool)
            if connector.id == best.id:
                metrics.routing_hits += 1
            if self.logger:
                self.logger.log(
                    "eval_task",
                    {
                        "task_id": task.id,
                        "connector_id": connector.id,
                        "passed": result.passed,
                        "steps": steps,
                    },
                    seed=seed,
                )
        return metrics


def tag_weights(task: Task) -> dict[str, float]:
    return task.required_tags


def connector_score(task: Task, connector: Connector) -> float:
    weights = tag_weights(task)
    total_w = sum(weights.values()) or 1.0
    score = 0.0
    for tag, weight in weights.items():
        score += weight * connector.capabilities[tag].score
    return score / total_w


def best_connector_for_task(task: Task, pool: list[Connector]) -> Connector:
    return max(pool, key=lambda c: connector_score(task, c))


def synthesize_code(task: Task, connector: Connector) -> str:
    """Simulate code generation quality from connector coding score vs task difficulty."""
    if not task.solution:
        raise ValueError(f"task {task.id} has no solution for simulated harness")
    coding = connector.capabilities["coding"].score
    margin = coding - task.difficulty
    if margin >= 0.1:
        return task.solution
    if margin >= -0.05:
        # near miss: broken edge case
        return task.solution.replace("return", "pass  # bug\n    return", 1)
    return "    raise NotImplementedError\n"


def step_budget(task: Task, connector: Connector) -> int:
    if task.difficulty < 0.2 and connector.capabilities["coding"].score >= 0.5:
        return 1
    if task.difficulty < 0.45:
        return 2
    return 3
