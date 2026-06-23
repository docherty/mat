"""Token efficiency helpers for connector speed profiles."""

from __future__ import annotations

# HumanEval worker-shot reference. We want models that answer concisely; set the
# reference to 1000 output tokens for a typical solution with minimal explanation.
DEFAULT_REFERENCE_OUTPUT_TOKENS = 1000.0


def token_efficiency_score(
    median_output_tokens: float,
    *,
    reference: float = DEFAULT_REFERENCE_OUTPUT_TOKENS,
) -> float:
    """Higher is better: 1.0 = at or below reference verbosity; 0.5 = 2× reference tokens."""
    if median_output_tokens <= 0:
        return 1.0
    return min(1.0, reference / median_output_tokens)


def speed_tier_from_efficiency(efficiency: float) -> str:
    if efficiency >= 0.85:
        return "fast"
    if efficiency >= 0.55:
        return "medium"
    return "slow"
