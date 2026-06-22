from __future__ import annotations

from dataclasses import dataclass

PROBE_SUITE_VERSION = "2026.1"

TIER_THRESHOLDS = {
    "weak": (0.0, 0.45),
    "mid": (0.45, 0.65),
    "strong": (0.65, 0.85),
    "frontier": (0.85, 1.01),
}


def score_to_tier(score: float) -> str:
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        if lo <= score < hi:
            return tier
    return "frontier"


@dataclass
class ProbeItem:
    id: str
    tag: str
    prompt: str
    oracle: str  # test code or expected answer


@dataclass
class ProbeSuite:
    version: str = PROBE_SUITE_VERSION
    items: list[ProbeItem] | None = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = _default_items()

    def run_stub(self) -> dict[str, float]:
        """Placeholder scoring until live model probes are wired."""
        scores: dict[str, list[float]] = {}
        for item in self.items or []:
            scores.setdefault(item.tag, []).append(0.5)
        return {tag: sum(vals) / len(vals) for tag, vals in scores.items()}


def _default_items() -> list[ProbeItem]:
    tags = (
        "reasoning",
        "coding",
        "verification",
        "instruction_following",
        "long_context",
        "tool_use",
    )
    items: list[ProbeItem] = []
    for tag in tags:
        for i in range(3):
            items.append(
                ProbeItem(
                    id=f"{tag}-{i}",
                    tag=tag,
                    prompt=f"probe {tag} #{i}",
                    oracle="assert True",
                )
            )
    return items
