from __future__ import annotations

CODING_HINTS = (
    "def ",
    "function ",
    "implement",
    "write code",
    "python",
    "class ",
    "refactor",
    "debug",
    "```",
)


def is_coding_request(messages: list[dict]) -> bool:
    text = " ".join(m.get("content", "") or "" for m in messages).lower()
    return any(h in text for h in CODING_HINTS)


def extract_coding_prompt(messages: list[dict]) -> str:
    """Use last user message as the coding task body."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"].strip()
    return ""


def guess_entry_point(prompt: str) -> str | None:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped.split("(")[0].removeprefix("def ").strip()
    return None
