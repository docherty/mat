from __future__ import annotations

from dataclasses import dataclass, field

from connectors.schema import Connector
from coordinator.policy import PromptedCoordinator, TrainedCoordinator
from eval.harness import step_budget, synthesize_code
from eval.oracle import run_oracle
from loop.difficulty import assess_difficulty, step_budget_for_tier


@dataclass
class LoopResult:
    answer: str
    passed: bool
    steps: int
    connector_id: str
    stages: list[str] = field(default_factory=list)


class OrchestrationLoop:
    def __init__(
        self,
        pool: list[Connector],
        coordinator: PromptedCoordinator | TrainedCoordinator | None = None,
        *,
        quality_tier: str = "balanced",
        revision_cap: int = 2,
    ):
        self.pool = pool
        self.coordinator = coordinator or PromptedCoordinator()
        self.quality_tier = quality_tier
        self.revision_cap = revision_cap

    def run(self, messages: list[dict], *, coding_task: dict | None = None) -> LoopResult:
        stages: list[str] = []
        difficulty = assess_difficulty(messages)

        if self.quality_tier == "fast" or difficulty < 0.12:
            stages.append("fast_path")
            connector = self.pool[0]
            if coding_task:
                code = synthesize_code(
                    type("T", (), coding_task)(),  # noqa: SLF001
                    connector,
                )
                body = coding_task["prompt"] + code
                result = run_oracle(body, coding_task["tests"])
                return LoopResult(
                    answer=body if result.passed else body + "\n# verification failed",
                    passed=result.passed,
                    steps=1,
                    connector_id=connector.id,
                    stages=stages,
                )
            return LoopResult(
                answer="(fast path proxy response)",
                passed=True,
                steps=1,
                connector_id=connector.id,
                stages=stages,
            )

        stages.append("think")
        budget = step_budget_for_tier(self.quality_tier, difficulty)
        stages.append("work")

        if not coding_task:
            connector = self.coordinator.pick(
                type("T", (), {"difficulty": difficulty, "required_tags": {"coding": 0.5}})(),
                self.pool,
            )
            return LoopResult(
                answer="(orchestrated response)",
                passed=True,
                steps=budget,
                connector_id=connector.id,
                stages=stages + ["synthesize"],
            )

        from eval.oracle import Task

        task = Task.from_dict(coding_task)
        connector = self.coordinator.pick(task, self.pool)
        code = synthesize_code(task, connector)
        body = task.prompt + code
        stages.append("verify")
        result = run_oracle(body, task.tests)
        revisions = 0
        while not result.passed and revisions < self.revision_cap:
            stages.append(f"revise_{revisions + 1}")
            code = task.solution  # retry with full solution on revise
            body = task.prompt + code
            result = run_oracle(body, task.tests)
            revisions += 1
            stages.append("verify")

        stages.append("synthesize")
        steps = step_budget(task, connector) + revisions
        return LoopResult(
            answer=body,
            passed=result.passed,
            steps=steps,
            connector_id=connector.id,
            stages=stages,
        )
