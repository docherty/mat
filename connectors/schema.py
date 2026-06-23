from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CAPABILITY_TAGS = (
    "reasoning",
    "coding",
    "long_context",
    "instruction_following",
    "verification",
    "tool_use",
)
TIERS = ("weak", "mid", "strong", "frontier")
TIER_SCORES = {"weak": 0.25, "mid": 0.5, "strong": 0.75, "frontier": 1.0}
TIER_THRESHOLDS = (
    ("weak", 0.0, 0.45),
    ("mid", 0.45, 0.65),
    ("strong", 0.65, 0.85),
    ("frontier", 0.85, 1.01),
)
SUPPORTED_VERSIONS = ("1.0", "1.1")
CURRENT_CONNECTOR_VERSION = "1.1"
CURRENT_PROBE_SUITE = "2026.1"
STALE_DAYS = 90


def tier_to_score(tier: str) -> float:
    return TIER_SCORES[tier]


def score_to_tier(score: float) -> str:
    for tier, lo, hi in TIER_THRESHOLDS:
        if lo <= score < hi:
            return tier
    return "frontier"


class Endpoint(BaseModel):
    type: Literal["openai", "ollama", "anthropic", "custom"]
    base_url: str
    model_name: str
    auth_env: str


class Pricing(BaseModel):
    input_per_1k: float
    output_per_1k: float
    currency: str = "USD"


class Supports(BaseModel):
    tools: bool = True
    json_mode: bool = True
    streaming: bool = True
    system_prompt: bool = True
    reasoning: bool = False


class Speed(BaseModel):
    tier: Literal["fast", "medium", "slow"] = "medium"
    tokens_per_sec: float | None = None
    median_output_tokens: float | None = None
    token_efficiency: float | None = Field(default=None, ge=0.0, le=1.0)


class CapabilityDim(BaseModel):
    """Continuous score for routing/tie-break; tier for coordinator generalisation."""

    score: float = Field(ge=0.0, le=1.0)
    tier: Literal["weak", "mid", "strong", "frontier"]

    @classmethod
    def from_score(cls, score: float) -> CapabilityDim:
        return cls(score=score, tier=score_to_tier(score))  # type: ignore[arg-type]

    @classmethod
    def from_tier(cls, tier: str) -> CapabilityDim:
        return cls(score=tier_to_score(tier), tier=tier)  # type: ignore[arg-type]


class BenchmarkAttestation(BaseModel):
    """Cited external measurement. Tie-break and human review only — not routing alone."""

    source: str
    metric: str
    value: float
    unit: Literal["index", "ratio", "percent", "tokens"] = "ratio"
    as_of: date
    url: str | None = None
    notes: str | None = None


class Profile(BaseModel):
    profile_method: Literal["mat_probe", "benchmark_import", "hand"] = "hand"
    catalog: str | None = None
    catalog_id: str | None = None
    probe_suite_version: str | None = None
    profiled_at: datetime
    contributor: str = "unknown"
    integrity_sha256: str = ""
    notes: str | None = None


class Connector(BaseModel):
    connector_version: str
    id: str
    display_name: str
    endpoint: Endpoint
    context_window: int = 32768
    max_output_tokens: int = 4096
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    pricing: Pricing | None = None
    locality: Literal["api", "local"] = "api"
    supports: Supports = Field(default_factory=Supports)
    tool_format: Literal["openai", "anthropic", "none"] = "openai"
    capabilities: dict[str, CapabilityDim]
    speed: Speed = Field(default_factory=Speed)
    benchmarks: list[BenchmarkAttestation] = Field(default_factory=list)
    profile: Profile

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_capabilities(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        version = str(data.get("connector_version", "1.0"))
        caps = data.get("capabilities")
        if not isinstance(caps, dict):
            return data
        normalized: dict[str, Any] = {}
        for tag, value in caps.items():
            if isinstance(value, str):
                normalized[tag] = {
                    "score": tier_to_score(value),
                    "tier": value,
                }
            else:
                normalized[tag] = value
        data["capabilities"] = normalized
        if version == "1.0":
            data["connector_version"] = "1.1"
            profile = data.get("profile") or {}
            if isinstance(profile, dict):
                profile.setdefault("profile_method", "mat_probe")
                if profile.get("probe_suite_version"):
                    profile.setdefault("profile_method", "mat_probe")
                data["profile"] = profile
        return data

    @field_validator("capabilities")
    @classmethod
    def check_capability_keys(cls, v: dict[str, CapabilityDim]) -> dict[str, CapabilityDim]:
        for tag in CAPABILITY_TAGS:
            if tag not in v:
                raise ValueError(f"missing capability tag: {tag}")
        for key in v:
            if key not in CAPABILITY_TAGS:
                raise ValueError(f"unknown capability tag: {key}")
        for tag, dim in v.items():
            expected = score_to_tier(dim.score)
            if dim.tier != expected:
                raise ValueError(
                    f"capabilities.{tag}.tier ({dim.tier}) inconsistent with "
                    f"score {dim.score} (expected {expected})"
                )
        return v

    def capability_vector(self) -> list[float]:
        return [self.capabilities[tag].score for tag in CAPABILITY_TAGS]

    def capability_tiers(self) -> dict[str, str]:
        return {tag: self.capabilities[tag].tier for tag in CAPABILITY_TAGS}

    def is_stale(self, suite_version: str = CURRENT_PROBE_SUITE) -> bool:
        age = datetime.now(UTC) - self.profile.profiled_at.replace(tzinfo=UTC)
        if age.days > STALE_DAYS:
            return True
        if self.profile.profile_method == "mat_probe" and self.profile.probe_suite_version:
            return self.profile.probe_suite_version != suite_version
        return False


def canonical_payload(connector: Connector) -> dict:
    """Fields covered by integrity_sha256."""
    return {
        "connector_version": connector.connector_version,
        "id": connector.id,
        "display_name": connector.display_name,
        "endpoint": connector.endpoint.model_dump(),
        "context_window": connector.context_window,
        "max_output_tokens": connector.max_output_tokens,
        "modalities": connector.modalities,
        "pricing": connector.pricing.model_dump() if connector.pricing else None,
        "locality": connector.locality,
        "supports": connector.supports.model_dump(),
        "tool_format": connector.tool_format,
        "capabilities": {k: v.model_dump() for k, v in connector.capabilities.items()},
        "speed": connector.speed.model_dump(),
        "benchmarks": [b.model_dump(mode="json") for b in connector.benchmarks],
        "profile": {
            "profile_method": connector.profile.profile_method,
            "catalog": connector.profile.catalog,
            "catalog_id": connector.profile.catalog_id,
            "probe_suite_version": connector.profile.probe_suite_version,
            "profiled_at": connector.profile.profiled_at.isoformat(),
            "contributor": connector.profile.contributor,
            "notes": connector.profile.notes,
        },
    }


def compute_integrity_hash(connector: Connector) -> str:
    payload = json.dumps(canonical_payload(connector), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_connector(connector: Connector, *, check_hash: bool = True) -> list[str]:
    errors: list[str] = []
    if connector.connector_version not in SUPPORTED_VERSIONS:
        errors.append(f"unsupported connector_version: {connector.connector_version}")
    if "://" not in connector.endpoint.base_url and connector.endpoint.type != "custom":
        errors.append("endpoint.base_url must be a URL")
    if connector.supports.tools and connector.tool_format == "none":
        errors.append("tool_format none incompatible with supports.tools=true")
    if check_hash and connector.profile.integrity_sha256:
        expected = compute_integrity_hash(connector)
        if connector.profile.integrity_sha256 != expected:
            errors.append("integrity_sha256 mismatch")
    return errors
