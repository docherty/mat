"""LM Studio OpenAI-compatible API helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlparse

import httpx

_LMSTUDIO_HOSTS = frozenset({"127.0.0.1", "localhost"})


class ModelNotServedError(RuntimeError):
    """Connector model_name is not exposed by the LM Studio server."""

    def __init__(
        self,
        *,
        connector_id: str,
        model_name: str,
        base_url: str,
        served: tuple[str, ...],
    ):
        self.connector_id = connector_id
        self.model_name = model_name
        self.base_url = base_url
        self.served = served
        served_preview = ", ".join(served[:8])
        more = f" (+{len(served) - 8} more)" if len(served) > 8 else ""
        super().__init__(
            f"connector {connector_id!r} requests model {model_name!r} but LM Studio at "
            f"{base_url} does not list it. Served: {served_preview}{more}. "
            f"Load the model in LM Studio or run mat-pool sync-lmstudio."
        )


def is_lmstudio_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in _LMSTUDIO_HOSTS


@lru_cache(maxsize=16)
def fetch_served_model_ids(base_url: str) -> tuple[str, ...]:
    """Return model ids from GET /v1/models (chat models only)."""
    url = f"{base_url.rstrip('/')}/models"
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json().get("data") or []
    ids: list[str] = []
    for entry in data:
        model_id = entry.get("id")
        if not model_id or entry.get("object") != "model":
            continue
        # Skip embedding-only models in routing pool.
        if "embed" in str(model_id).lower():
            continue
        ids.append(str(model_id))
    return tuple(ids)


def clear_served_model_cache() -> None:
    fetch_served_model_ids.cache_clear()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _folder_tokens(folder_name: str) -> list[str]:
    base = re.sub(r"-(8bit|4bit|gguf|mlx.*|qat.*)$", "", folder_name, flags=re.I)
    parts = re.split(r"[-_/]+", base.lower())
    tokens = [p for p in parts if len(p) >= 3 and p not in {"mlx", "community", "lmstudio", "gguf"}]
    if re.search(r"qwen3-5", folder_name, re.I):
        tokens.extend(["qwen3.5", "qwen35"])
    return tokens


def match_served_model_id(
    folder_name: str,
    catalog_path: str,
    served_ids: list[str] | tuple[str, ...],
    *,
    guess: str | None = None,
) -> str | None:
    """Map a cache folder to an exact id from LM Studio /v1/models."""
    if not served_ids:
        return None
    served = list(served_ids)
    if guess and guess in served:
        return guess

    folder_key = _normalize_key(folder_name)
    catalog_key = _normalize_key(catalog_path.replace("/", "-"))
    tokens = _folder_tokens(folder_name)

    best_id: str | None = None
    best_score = 0
    for model_id in served:
        mid_key = _normalize_key(model_id)
        score = 0
        if mid_key == folder_key or mid_key == catalog_key:
            return model_id
        if mid_key in folder_key or folder_key in mid_key:
            score = max(score, min(len(mid_key), len(folder_key)))
        if mid_key in catalog_key or catalog_key in mid_key:
            score = max(score, min(len(mid_key), len(catalog_key)) // 2 + 4)
        for token in tokens:
            if token in model_id.lower():
                score += len(token)
            if token.replace(".", "") in _normalize_key(model_id):
                score += len(token)
        if score > best_score:
            best_score = score
            best_id = model_id

    return best_id if best_score >= 10 else None


def resolve_connector_model(
    connector_id: str,
    base_url: str,
    model_name: str,
    *,
    validate: bool = True,
) -> str:
    """Return connector model_name; optionally require it on the LM Studio server."""
    if not validate or not is_lmstudio_url(base_url):
        return model_name
    served = fetch_served_model_ids(base_url)
    if not served:
        return model_name
    if model_name not in served:
        raise ModelNotServedError(
            connector_id=connector_id,
            model_name=model_name,
            base_url=base_url,
            served=served,
        )
    return model_name
