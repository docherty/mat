from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from connectors.local_affinity import apply_local_pin, resolve_local_pins, routing_pool
from connectors.schema import Connector
from coordinator.policy import PromptedCoordinator, TrainedCoordinator
from eval.coding import (
    TurnRecord,
    build_oracle_script,
    call_role,
    parse_verdict,
)
from eval.oracle import Task, run_oracle
from workers.llm import LLMWorker, estimate_cost_usd


class Coordinator(Protocol):
    def pick(
        self,
        task: Task,
        pool: list[Connector],
        *,
        role: str,
        transcript: str = "",
    ) -> Connector: ...


@dataclass
class LiveLoopConfig:
    max_turns: int = 5
    revision_cap: int = 2
    use_thinker: bool = True
    skip_oracle: bool = False


def trinity_loop_config(*, use_thinker: bool = True) -> LiveLoopConfig:
    """Match Trinity turn budget (5) for fair compare baselines."""
    return LiveLoopConfig(max_turns=5, revision_cap=5, use_thinker=use_thinker)


@dataclass
class LiveLoopResult:
    task_id: str
    passed: bool
    code: str
    turns: list[TurnRecord]
    steps: int
    revisions: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    connector_ids: list[str]
    stages: list[str]
    error: str | None = None


class RoleCoordinator:
    """Capability routing with role-aware tag emphasis."""

    ROLE_TAG_BOOST = {
        "thinker": {"reasoning": 1.5, "instruction_following": 1.2},
        "worker": {"coding": 2.0},
        "verifier": {"verification": 2.0, "reasoning": 1.2},
    }

    def __init__(self, base: PromptedCoordinator | TrainedCoordinator | None = None):
        self.base = base or PromptedCoordinator()
        self._local_pins: dict = {}

    def bind_pool(self, pool: list[Connector]) -> None:
        self._local_pins = resolve_local_pins(pool)

    def pick(
        self,
        task: Task,
        pool: list[Connector],
        *,
        role: str,
        transcript: str = "",
    ) -> Connector:
        boosted = Task(
            id=task.id,
            prompt=task.prompt,
            tests=task.tests,
            difficulty=task.difficulty,
            required_tags=_boost_tags(task.required_tags, self.ROLE_TAG_BOOST.get(role, {})),
            solution=task.solution,
            entry_point=task.entry_point,
            split=task.split,
        )
        from coordinator.slm_coordinator import SLMCoordinator

        if isinstance(self.base, SLMCoordinator):
            choice = self.base.pick(boosted, pool, role=role, transcript=transcript)
        else:
            choice = self.base.pick(boosted, pool, transcript=transcript)
        return apply_local_pin(choice, self._local_pins)


def _boost_tags(base: dict[str, float], boosts: dict[str, float]) -> dict[str, float]:
    if not boosts:
        return base
    out = dict(base)
    for tag, mult in boosts.items():
        out[tag] = out.get(tag, 0.1) * mult
    return out


class LiveCodingLoop:
    def __init__(
        self,
        pool: list[Connector],
        coordinator: Coordinator | None = None,
        *,
        worker: LLMWorker | None = None,
        config: LiveLoopConfig | None = None,
    ):
        self.pool = routing_pool(pool)
        if isinstance(coordinator, RoleCoordinator):
            self.coordinator = coordinator
            self.coordinator.bind_pool(self.pool)
        else:
            role_coord = RoleCoordinator(coordinator)
            role_coord.bind_pool(self.pool)
            self.coordinator = role_coord
        self.worker = worker or LLMWorker()
        self.config = config or LiveLoopConfig()

    def run_single(
        self,
        connector: Connector,
        task: Task,
        *,
        reflect: bool = False,
    ) -> LiveLoopResult:
        """One model — worker + oracle; optional full think/work/verify when reflect=True."""
        return self._run_roles(
            task,
            lambda _role: connector,
            stages=["single_reflect"] if reflect else ["single"],
            force_sequence=reflect,
        )

    def run_orchestrated(self, task: Task) -> LiveLoopResult:
        transcript_parts: list[str] = []

        def pick_role(role: str) -> Connector:
            return self.coordinator.pick(
                task,
                self.pool,
                role=role,
                transcript="\n\n".join(transcript_parts),
            )

        return self._run_roles(
            task,
            pick_role,
            stages=["orchestrated"],
            force_sequence=self.config.use_thinker,
            transcript_parts=transcript_parts,
        )

    def _run_roles(
        self,
        task: Task,
        pick: callable,
        *,
        stages: list[str],
        force_sequence: bool = True,
        transcript_parts: list[str] | None = None,
    ) -> LiveLoopResult:
        turns: list[TurnRecord] = []
        if transcript_parts is None:
            transcript_parts = []
        draft_code = ""
        revisions = 0
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        connector_ids: list[str] = []
        stages_log = list(stages)

        def record(turn: TurnRecord, connector: Connector) -> None:
            nonlocal input_tokens, output_tokens, cost
            turns.append(turn)
            connector_ids.append(connector.id)
            input_tokens += turn.completion.input_tokens
            output_tokens += turn.completion.output_tokens
            cost += estimate_cost_usd(
                connector,
                turn.completion.input_tokens,
                turn.completion.output_tokens,
            )

        def at_turn_budget() -> bool:
            return len(turns) >= self.config.max_turns

        if force_sequence and self.config.use_thinker:
            if at_turn_budget():
                return LiveLoopResult(
                    task_id=task.id,
                    passed=False,
                    code=draft_code,
                    turns=turns,
                    steps=len(turns),
                    revisions=revisions,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                    connector_ids=connector_ids,
                    stages=stages_log + ["turn_budget"],
                    error="max turns exceeded before work",
                )
            thinker = pick("thinker")
            turn, plan = call_role(self.worker, thinker, task, "thinker")
            record(turn, thinker)
            transcript_parts.append(f"[thinker/{thinker.id}]\n{plan}")
            stages_log.append("think")

        while revisions <= self.config.revision_cap:
            if at_turn_budget():
                break
            worker_conn = pick("worker")
            turn, draft_code = call_role(
                self.worker,
                worker_conn,
                task,
                "worker",
                transcript="\n\n".join(transcript_parts),
            )
            record(turn, worker_conn)
            stages_log.append("work")

            if self.config.skip_oracle or not (task.tests or "").strip():
                return LiveLoopResult(
                    task_id=task.id,
                    passed=True,
                    code=draft_code,
                    turns=turns,
                    steps=len(turns),
                    revisions=revisions,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                    connector_ids=connector_ids,
                    stages=stages_log + ["synthesize"],
                )

            script = build_oracle_script(task, draft_code)
            oracle = run_oracle(script, "")
            if oracle.passed:
                return LiveLoopResult(
                    task_id=task.id,
                    passed=True,
                    code=script,
                    turns=turns,
                    steps=len(turns),
                    revisions=revisions,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                    connector_ids=connector_ids,
                    stages=stages_log + ["verify_oracle"],
                )

            verifier = pick("verifier")
            if at_turn_budget():
                break
            turn, verdict_text = call_role(
                self.worker,
                verifier,
                task,
                "verifier",
                transcript="\n\n".join(transcript_parts),
                draft_code=draft_code,
            )
            record(turn, verifier)
            stages_log.append("verify")
            verdict, diagnosis = parse_verdict(verdict_text)
            transcript_parts.append(f"[verifier/{verifier.id}]\n{verdict}: {diagnosis}")

            if verdict == "ACCEPT":
                return LiveLoopResult(
                    task_id=task.id,
                    passed=False,
                    code=script,
                    turns=turns,
                    steps=len(turns),
                    revisions=revisions,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                    connector_ids=connector_ids,
                    stages=stages_log + ["verify_accept_but_failed"],
                    error=oracle.error or "verifier accepted failing code",
                )

            revisions += 1
            if revisions > self.config.revision_cap:
                break
            transcript_parts.append(f"[worker/{worker_conn.id}]\n{draft_code}")
            stages_log.append(f"revise_{revisions}")

        return LiveLoopResult(
            task_id=task.id,
            passed=False,
            code=build_oracle_script(task, draft_code),
            turns=turns,
            steps=len(turns),
            revisions=revisions,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            connector_ids=connector_ids,
            stages=stages_log + ["failed"],
            error="max revisions exceeded",
        )
