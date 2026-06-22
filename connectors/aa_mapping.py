"""Map Artificial Analysis evaluation fields to mat capability scores."""

from __future__ import annotations

from datetime import date

from connectors.schema import BenchmarkAttestation, CapabilityDim

# (evaluation key or benchmark metric name, unit)
# unit: ratio (0-1), percent (0-100), index_60 (AA intelligence scale), index_100
TAG_METRICS: dict[str, list[tuple[str, str]]] = {
    "reasoning": [
        ("gpqa_diamond", "percent"),
        ("gpqa", "ratio"),
        ("artificial_analysis_intelligence_index", "index_60"),
        ("hle", "ratio"),
        ("critpt", "percent"),
    ],
    "coding": [
        ("livecodebench", "ratio"),
        ("swe_bench_verified", "percent"),
        ("scicode", "ratio"),
        ("terminal_bench_hard", "percent"),
        ("terminal_bench_v2", "percent"),
        ("artificial_analysis_coding_index", "index_100"),
    ],
    "long_context": [
        ("aa_lcr", "percent"),
        ("long_context_recall", "percent"),
    ],
    "instruction_following": [
        ("ifbench", "percent"),
        ("ifbench_strict", "percent"),
    ],
    "verification": [
        ("gdpval_aa_v2", "index_1500"),
        ("tau2_bench_telecom", "percent"),
        ("tau2_bench", "percent"),
    ],
    "tool_use": [
        ("tau2_bench_telecom", "percent"),
        ("tau2_bench", "percent"),
        ("tau3_banking", "percent"),
    ],
}

# Published attestations to copy when present
ATTESTATION_METRICS: list[tuple[str, str, str]] = [
    ("artificial_analysis_intelligence_index", "intelligence_index_v4.1", "index"),
    ("artificial_analysis_coding_index", "coding_index", "index"),
    ("livecodebench", "livecodebench", "ratio"),
    ("swe_bench_verified", "swe_bench_verified", "percent"),
    ("gpqa_diamond", "gpqa_diamond", "percent"),
    ("gpqa", "gpqa_diamond", "ratio"),
    ("tau2_bench_telecom", "tau2_bench_telecom", "percent"),
    ("terminal_bench_hard", "terminal_bench_hard", "percent"),
    ("aa_lcr", "aa_lcr", "percent"),
    ("gdpval_aa_v2", "gdpval_aa_v2", "index"),
    ("ifbench", "ifbench", "percent"),
]


def normalize_value(value: float, unit: str) -> float:
    if unit == "ratio":
        return min(1.0, max(0.0, value if value <= 1.0 else value / 100.0))
    if unit == "percent":
        return min(1.0, max(0.0, value / 100.0))
    if unit == "index_60":
        return min(1.0, max(0.0, value / 60.0))
    if unit == "index_100":
        return min(1.0, max(0.0, value / 100.0))
    if unit == "index_1500":
        return min(1.0, max(0.0, value / 1500.0))
    return min(1.0, max(0.0, float(value)))


def _eval_get(evaluations: dict, key: str) -> float | None:
    if key not in evaluations:
        return None
    val = evaluations[key]
    if val is None:
        return None
    return float(val)


def capabilities_from_evaluations(evaluations: dict) -> dict[str, CapabilityDim]:
    caps: dict[str, CapabilityDim] = {}
    for tag, metrics in TAG_METRICS.items():
        scores: list[float] = []
        for key, unit in metrics:
            val = _eval_get(evaluations, key)
            if val is not None:
                scores.append(normalize_value(val, unit))
        if scores:
            caps[tag] = CapabilityDim.from_score(max(scores))
        else:
            caps[tag] = CapabilityDim.from_score(0.5)
    return caps


def benchmarks_from_evaluations(
    slug: str,
    evaluations: dict,
    *,
    as_of: date | None = None,
) -> list[BenchmarkAttestation]:
    as_of = as_of or date.today()
    url = f"https://artificialanalysis.ai/models/{slug}"
    out: list[BenchmarkAttestation] = []
    seen: set[str] = set()
    for eval_key, metric_name, unit in ATTESTATION_METRICS:
        val = _eval_get(evaluations, eval_key)
        if val is None or metric_name in seen:
            continue
        seen.add(metric_name)
        att_unit: str = unit
        if unit == "ratio" and val <= 1.0:
            att_unit = "ratio"
        elif unit == "ratio":
            att_unit = "percent"
            val = val * 100 if val <= 1 else val
        out.append(
            BenchmarkAttestation(
                source="artificial_analysis",
                metric=metric_name,
                value=float(val),
                unit=att_unit,  # type: ignore[arg-type]
                as_of=as_of,
                url=url,
            )
        )
    return out


def speed_tier_from_tokens_per_sec(tps: float | None) -> str:
    if tps is None:
        return "medium"
    if tps >= 120:
        return "fast"
    if tps >= 50:
        return "medium"
    return "slow"
