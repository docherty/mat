from __future__ import annotations

import re

from connectors.schema import Connector
from eval.oracle import Task
from workers.llm import CompletionResult


class MockLLMWorker:
    """Deterministic worker for tests — returns oracle solutions when possible."""

    def __init__(self, *, fail_ids: set[str] | None = None):
        self.fail_ids = fail_ids or set()
        self.calls: list[tuple[str, str, Connector]] = []

    def complete(
        self,
        connector: Connector,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        task: Task | None = None,
    ) -> CompletionResult:
        role = "worker"
        for msg in messages:
            if msg.get("role") == "system":
                if "THINKER" in msg["content"]:
                    role = "thinker"
                elif "VERIFIER" in msg["content"]:
                    role = "verifier"
                elif "WORKER" in msg["content"]:
                    role = "worker"
        self.calls.append((role, messages[-1]["content"], connector))

        text = self._reply(role, messages[-1]["content"], connector, task)
        return CompletionResult(
            text=text,
            input_tokens=100,
            output_tokens=len(text.split()),
            model=connector.endpoint.model_name,
        )

    def _reply(
        self,
        role: str,
        user: str,
        connector: Connector,
        task: Task | None,
    ) -> str:
        if role == "thinker":
            return "- Parse inputs\n- Handle edge cases\n- Return correct type"
        if role == "verifier":
            coding = connector.capabilities["coding"].score
            if task and task.id in self.fail_ids:
                return "REVISE\nlogic error"
            if coding >= 0.7:
                return "ACCEPT\ncode looks correct"
            return "REVISE\nneeds fix"
        # worker
        if task and task.solution and task.id not in self.fail_ids:
            if task.entry_point:
                body = task.solution
                if f"def {task.entry_point}" not in body:
                    return f"```python\n{task.prompt}{body}```"
                return f"```python\n{body}```"
            return f"```python\n{task.prompt}{task.solution}```"
        return "```python\n    raise NotImplementedError\n```"


def task_from_user_prompt(user: str, tasks: list[Task]) -> Task | None:
    for task in tasks:
        if task.prompt.strip() in user:
            return task
        if task.id in user:
            return task
    m = re.search(r"HumanEval/\d+", user)
    if m:
        tid = m.group(0)
        for task in tasks:
            if task.id == tid:
                return task
    return None
