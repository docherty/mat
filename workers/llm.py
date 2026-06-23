from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from connectors.lmstudio_api import ModelNotServedError, resolve_connector_model
from connectors.schema import Connector

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

__all__ = [
    "CompletionResult",
    "LLMWorker",
    "ModelNotServedError",
    "StreamChunk",
    "resolve_model_name",
]


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


def resolve_model_name(connector: Connector) -> str:
    """Use the connector's model_name exactly; validate against LM Studio when local."""
    strict = os.environ.get("MAT_STRICT_LMSTUDIO_MODELS", "1").lower() in ("1", "true", "yes")
    return resolve_connector_model(
        connector.id,
        connector.endpoint.base_url,
        connector.endpoint.model_name,
        validate=strict,
    )


@dataclass
class StreamChunk:
    text: str = ""
    done: bool = False
    result: CompletionResult | None = None
    connector_id: str | None = None
    stages: list[str] | None = None
    passed: bool = True
    cost_usd: float = 0.0


class LLMWorker:
    """OpenAI-compatible chat client (OpenRouter, Venice, LM Studio, mlx-lm, etc.)."""

    def __init__(self, *, timeout_sec: float | None = None):
        if timeout_sec is None:
            timeout_sec = float(os.environ.get("MAT_LLM_TIMEOUT", "300"))
        self.timeout_sec = timeout_sec

    def _headers(self, connector: Connector) -> dict[str, str]:
        api_key = os.environ.get(connector.endpoint.auth_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def complete(
        self,
        connector: Connector,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        headers = self._headers(connector)
        url = f"{connector.endpoint.base_url.rstrip('/')}/chat/completions"
        model_name = resolve_model_name(connector)
        payload = {
            "model": model_name,
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
            model=str(data.get("model", model_name)),
            finish_reason=choice.get("finish_reason"),
        )

    def stream_complete(
        self,
        connector: Connector,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream token deltas from an OpenAI-compatible provider."""
        headers = self._headers(connector)
        url = f"{connector.endpoint.base_url.rstrip('/')}/chat/completions"
        model_name = resolve_model_name(connector)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or connector.max_output_tokens,
            "stream": True,
        }
        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        finish_reason: str | None = None
        resp_model = model_name

        with httpx.Client(timeout=self.timeout_sec) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for raw in response.iter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    data_s = raw.removeprefix("data:").strip()
                    if data_s == "[DONE]":
                        break
                    try:
                        data = json.loads(data_s)
                    except json.JSONDecodeError:
                        continue
                    resp_model = str(data.get("model", resp_model))
                    usage = data.get("usage")
                    if usage:
                        input_tokens = int(usage.get("prompt_tokens", input_tokens))
                        output_tokens = int(usage.get("completion_tokens", output_tokens))
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        text_parts.append(piece)
                        yield StreamChunk(text=piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice.get("finish_reason")

        full = "".join(text_parts).strip()
        result = CompletionResult(
            text=full,
            input_tokens=input_tokens,
            output_tokens=output_tokens or max(1, len(full.split())),
            model=resp_model,
            finish_reason=finish_reason or "stop",
        )
        yield StreamChunk(done=True, result=result)


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
