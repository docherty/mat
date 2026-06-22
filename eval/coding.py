from __future__ import annotations

from dataclasses import dataclass

from connectors.schema import Connector
from eval.oracle import Task
from workers.llm import CompletionResult, LLMWorker, extract_code

ROLES = ("thinker", "worker", "verifier")

ROLE_SYSTEM = {
    "thinker": (
        "You are the THINKER in a coding team. Analyze the task and return a short plan "
        "(bullets). Do not write final code. Be concrete about edge cases and approach."
    ),
    "worker": (
        "You are the WORKER. Implement the function in Python. Return either a fenced "
        "```python``` block with the full function, or only the function body if the "
        "signature is already provided."
    ),
    "verifier": (
        "You are the VERIFIER. Review the proposed solution against the task. "
        "Reply with a first line of exactly ACCEPT or REVISE, then a brief diagnosis."
    ),
}


@dataclass
class TurnRecord:
    role: str
    connector_id: str
    completion: CompletionResult
    extracted: str


@dataclass
class CodingAttempt:
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
    error: str | None = None


def build_oracle_script(task: Task, code: str) -> str:
    entry_point = task.entry_point or _guess_entry_point(task.prompt)
    body = extract_code(code, entry_point=entry_point)
    if entry_point and f"def {entry_point}" not in body:
        body = task.prompt + body
    elif not entry_point:
        body = task.prompt + body
    return f"{body}\n\n{task.tests}\n"


def _guess_entry_point(prompt: str) -> str | None:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped.split("(")[0].removeprefix("def ").strip()
    return None


def role_messages(
    task: Task,
    role: str,
    *,
    transcript: str = "",
    draft_code: str = "",
) -> list[dict[str, str]]:
    user_parts = [f"# Task\n{task.prompt}"]
    if transcript:
        user_parts.append(f"# Prior turns\n{transcript}")
    if role == "verifier" and draft_code:
        user_parts.append(f"# Candidate solution\n{draft_code}")
    if role == "worker" and transcript:
        user_parts.append("# Follow the plan above when implementing.")

    return [
        {"role": "system", "content": ROLE_SYSTEM[role]},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def call_role(
    worker: LLMWorker,
    connector: Connector,
    task: Task,
    role: str,
    *,
    transcript: str = "",
    draft_code: str = "",
) -> tuple[TurnRecord, str]:
    messages = role_messages(task, role, transcript=transcript, draft_code=draft_code)
    completion = worker.complete(connector, messages)
    if role == "worker":
        extracted = extract_code(completion.text, entry_point=task.entry_point)
    else:
        extracted = completion.text.strip()
    turn = TurnRecord(
        role=role,
        connector_id=connector.id,
        completion=completion,
        extracted=extracted,
    )
    return turn, extracted


def parse_verdict(text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "REVISE", "empty verifier response"
    head = lines[0].upper()
    if head.startswith("ACCEPT"):
        return "ACCEPT", "\n".join(lines[1:])
    if head.startswith("REVISE"):
        return "REVISE", "\n".join(lines[1:])
    if "ACCEPT" in head:
        return "ACCEPT", "\n".join(lines[1:])
    return "REVISE", text
