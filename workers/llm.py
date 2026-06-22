from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from connectors.schema import Connector

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMWorker:
    """OpenAI-compatible chat client (OpenRouter, Venice, LM Studio, mlx-lm, etc.)."""

    def __init__(self, *, timeout_sec: float = 120.0):
        self.timeout_sec = timeout_sec

    def complete(
        self,
        connector: Connector,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        api_key = os.environ.get(connector.endpoint.auth_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{connector.endpoint.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": connector.endpoint.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or connector.max_output_tokens,
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return CompletionResult(
            text=(choice["message"].get("content") or "").strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=str(data.get("model", connector.endpoint.model_name)),
            finish_reason=choice.get("finish_reason"),
        )


def extract_code(text: str, *, entry_point: str | None = None) -> str:
    """Pull Python from a fenced block or bare function body."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip() + "\n"

    if entry_point and f"def {entry_point}" in text:
        start = text.index(f"def {entry_point}")
        return text[start:].strip() + "\n"

    # Model returned only an indented body (HumanEval-style completion).
    lines = text.splitlines()
    if lines and lines[0].startswith((" ", "\t")):
        return text.strip() + "\n"

    return text.strip() + "\n"


def estimate_cost_usd(connector: Connector, input_tokens: int, output_tokens: int) -> float:
    if not connector.pricing:
        return 0.0
    return (input_tokens / 1000.0) * connector.pricing.input_per_1k + (
        output_tokens / 1000.0
    ) * connector.pricing.output_per_1k
