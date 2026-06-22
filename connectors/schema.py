from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
SPEED_TIERS = ("fast", "medium", "slow")
ENDPOINT_TYPES = ("openai", "ollama", "anthropic", "custom")
TOOL_FORMATS = ("openai", "anthropic", "none")
LOCALITIES = ("api", "local")
CURRENT_PROBE_SUITE = "2026.1"
STALE_DAYS = 90


def tier_to_score(tier: str) -> float:
    return TIER_SCORES[tier]


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


class Speed(BaseModel):
    tokens_per_sec: float = 30.0
    tier: Literal["fast", "medium", "slow"] = "medium"


class Profile(BaseModel):
    probe_suite_version: str
    profiled_at: datetime
    probe_scores: dict[str, float] = Field(default_factory=dict)
    contributor: str = "unknown"
    integrity_sha256: str = ""


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
    capabilities: dict[str, Literal["weak", "mid", "strong", "frontier"]]
    speed: Speed = Field(default_factory=Speed)
    profile: Profile

    @field_validator("capabilities")
    @classmethod
    def check_capability_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for tag in CAPABILITY_TAGS:
            if tag not in v:
                raise ValueError(f"missing capability tag: {tag}")
        for key in v:
            if key not in CAPABILITY_TAGS:
                raise ValueError(f"unknown capability tag: {key}")
        return v

    def capability_vector(self) -> list[float]:
        return [tier_to_score(self.capabilities[tag]) for tag in CAPABILITY_TAGS]

    def is_stale(self, suite_version: str = CURRENT_PROBE_SUITE) -> bool:
        age = datetime.now(UTC) - self.profile.profiled_at.replace(tzinfo=UTC)
        return age.days > STALE_DAYS or self.profile.probe_suite_version != suite_version


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
        "capabilities": connector.capabilities,
        "speed": connector.speed.model_dump(),
        "profile": {
            "probe_suite_version": connector.profile.probe_suite_version,
            "profiled_at": connector.profile.profiled_at.isoformat(),
            "probe_scores": connector.profile.probe_scores,
            "contributor": connector.profile.contributor,
        },
    }


def compute_integrity_hash(connector: Connector) -> str:
    payload = json.dumps(canonical_payload(connector), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_connector(connector: Connector, *, check_hash: bool = True) -> list[str]:
    errors: list[str] = []
    if connector.connector_version != "1.0":
        errors.append(f"unsupported connector_version: {connector.connector_version}")
    if "://" not in connector.endpoint.base_url and connector.endpoint.type != "custom":
        errors.append("endpoint.base_url must be a URL")
    if check_hash and connector.profile.integrity_sha256:
        expected = compute_integrity_hash(connector)
        if connector.profile.integrity_sha256 != expected:
            errors.append("integrity_sha256 mismatch")
    return errors
