from __future__ import annotations


def assess_difficulty(messages: list[dict]) -> float:
    """Heuristic difficulty 0–1 from message content."""
    text = " ".join(m.get("content", "") or "" for m in messages)
    score = min(1.0, len(text) / 4000.0)
    if any(kw in text.lower() for kw in ("prove", "refactor", "debug", "optimise", "optimize")):
        score = min(1.0, score + 0.2)
    if len(text) < 80:
        score = min(score, 0.15)
    return score


def step_budget_for_tier(tier: str, difficulty: float) -> int:
    if tier == "fast" or difficulty < 0.15:
        return 1
    if tier == "balanced":
        return 2 if difficulty < 0.5 else 3
    return 4 if difficulty > 0.6 else 3
