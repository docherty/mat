from __future__ import annotations

import os
from dataclasses import dataclass, field

from connectors.schema import Connector
from coordinator.factory import load_role_coordinator
from coordinator.policy import PromptedCoordinator, TrainedCoordinator
from eval.harness import step_budget, synthesize_code
from eval.live_loop import LiveCodingLoop, LiveLoopConfig, RoleCoordinator
from eval.oracle import Task, run_oracle
from loop.coding_detect import extract_coding_prompt, guess_entry_point, is_coding_request
from loop.difficulty import assess_difficulty, step_budget_for_tier
from workers.llm import LLMWorker


@dataclass
class LoopResult:
    answer: str
    passed: bool
    steps: int
    connector_id: str
    model_id: str | None = None
    stages: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


def live_enabled() -> bool:
    return os.environ.get("MAT_LIVE", "1").lower() in ("1", "true", "yes")


class OrchestrationLoop:
    def __init__(
        self,
        pool: list[Connector],
        coordinator: PromptedCoordinator | TrainedCoordinator | RoleCoordinator | None = None,
        *,
        quality_tier: str = "balanced",
        revision_cap: int = 2,
        live: bool | None = None,
        worker: LLMWorker | None = None,
    ):
        self.pool = pool
        if isinstance(coordinator, RoleCoordinator):
            self._role_coordinator = coordinator
            self.coordinator = coordinator.base
        else:
            self.coordinator = coordinator or PromptedCoordinator()
            self._role_coordinator = RoleCoordinator(self.coordinator)
        self.quality_tier = quality_tier
        self.revision_cap = revision_cap
        self.live = live_enabled() if live is None else live
        self.worker = worker or LLMWorker()

    @classmethod
    def from_env(
        cls, pool: list[Connector], *, quality_tier: str = "balanced"
    ) -> OrchestrationLoop:
        return cls(pool, load_role_coordinator(), quality_tier=quality_tier)

    def run(self, messages: list[dict], *, coding_task: dict | None = None) -> LoopResult:
        if self.live and self.pool:
            return self._run_live(messages, coding_task=coding_task)
        return self._run_simulated(messages, coding_task=coding_task)

    def _run_live(self, messages: list[dict], *, coding_task: dict | None) -> LoopResult:
        # Non-coding chat should behave like a normal OpenAI gateway: route once, then passthrough.
        if coding_task is None and not is_coding_request(messages):
            difficulty = assess_difficulty(messages)
            task = Task(
                id="chat",
                prompt=extract_coding_prompt(messages),
                tests="",
                difficulty=difficulty,
                required_tags={"instruction_following": 1.0},
                entry_point=None,
            )
            conn = self._role_coordinator.pick(task, self.pool, role="worker")
            completion = self.worker.complete(conn, messages)
            return LoopResult(
                answer=completion.text,
                passed=True,
                steps=1,
                connector_id=conn.id,
                model_id=completion.model,
                stages=["chat_passthrough"],
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                estimated_cost_usd=0.0,
            )

        config = LiveLoopConfig(
            revision_cap=self.revision_cap,
            use_thinker=self.quality_tier != "fast",
            skip_oracle=coding_task is None,
        )
        live = LiveCodingLoop(
            self.pool,
            self._role_coordinator,
            worker=self.worker,
            config=config,
        )

        if coding_task:
            task = Task.from_dict(coding_task)
        else:
            prompt = extract_coding_prompt(messages)
            difficulty = assess_difficulty(messages)
            tags = (
                {"coding": 1.0}
                if is_coding_request(messages)
                else {"instruction_following": 1.0}
            )
            task = Task(
                id="chat",
                prompt=prompt,
                tests="",
                difficulty=difficulty,
                required_tags=tags,
                entry_point=guess_entry_point(prompt),
            )

        if self.quality_tier == "fast":
            conn = self.pool[0]
            result = live.run_single(conn, task, reflect=False)
        else:
            result = live.run_orchestrated(task)

        primary = result.connector_ids[-1] if result.connector_ids else self.pool[0].id
        return LoopResult(
            answer=result.code,
            passed=result.passed,
            steps=result.steps,
            connector_id=primary,
            model_id=result.turns[-1].completion.model if result.turns else None,
            stages=result.stages,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
        )

    def _run_simulated(self, messages: list[dict], *, coding_task: dict | None) -> LoopResult:
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
                answer="(fast path proxy response — set MAT_LIVE=1 for real LLMs)",
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
                answer="(orchestrated proxy — set MAT_LIVE=1 for real LLMs)",
                passed=True,
                steps=budget,
                connector_id=connector.id,
                stages=stages + ["synthesize"],
            )

        task = Task.from_dict(coding_task)
        connector = self.coordinator.pick(task, self.pool)
        code = synthesize_code(task, connector)
        body = task.prompt + code
        stages.append("verify")
        result = run_oracle(body, task.tests)
        revisions = 0
        while not result.passed and revisions < self.revision_cap:
            stages.append(f"revise_{revisions + 1}")
            code = task.solution or code
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
