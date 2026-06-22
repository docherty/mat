"""Fetch and cache Artificial Analysis model benchmarks."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from connectors.aa_mapping import (
    benchmarks_from_evaluations,
    capabilities_from_evaluations,
    speed_tier_from_tokens_per_sec,
)
from connectors.paths import AA_CACHE_PATH
from connectors.schema import Connector, Endpoint, Pricing, Profile, Speed, Supports

AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models"
AA_MODEL_URL = "https://artificialanalysis.ai/api/v2/language/models/{slug}"
AA_PUBLIC_URL = "https://artificialanalysis.ai/models/{slug}"

_SCRAPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "artificial_analysis_intelligence_index": re.compile(
        r'intelligence_index_v4_1\\":\s*([0-9.]+)'
    ),
    "artificial_analysis_coding_index": re.compile(r'coding_index\\":\s*([0-9.]+)'),
    "livecodebench": re.compile(r'livecodebench\\":\s*([0-9.]+)'),
    "gpqa": re.compile(r'gpqa(?:_diamond)?\\":\s*([0-9.]+)'),
    "tau2_bench_telecom": re.compile(r'tau2_bench_telecom\\":\s*([0-9.]+)'),
    "aa_lcr": re.compile(r'aa_lcr\\":\s*([0-9.]+)'),
    "terminal_bench_hard": re.compile(r'terminal_bench_hard\\":\s*([0-9.]+)'),
}


def _api_headers() -> dict[str, str]:
    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "")
    if not key:
        raise ValueError(
            "ARTIFICIAL_ANALYSIS_API_KEY is required. "
            "Free tier: https://artificialanalysis.ai/data-api (1,000 req/day)."
        )
    return {"x-api-key": key, "Accept": "application/json"}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def fetch_all_models(*, cache_path: Path | None = None) -> list[dict]:
    cache = cache_path or AA_CACHE_PATH
    req = Request(AA_MODELS_URL, headers=_api_headers())
    with urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    models = payload.get("data") or payload.get("models") or []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                "intelligence_index_version": payload.get("intelligence_index_version"),
                "data": models,
            },
            indent=2,
        )
        + "\n"
    )
    return models


def fetch_aa_public(slug: str) -> dict:
    """Scrape headline benchmarks from the public AA model page (no API key)."""
    url = AA_PUBLIC_URL.format(slug=slug)
    with urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    evaluations: dict[str, float] = {}
    for key, pattern in _SCRAPE_PATTERNS.items():
        m = pattern.search(html)
        if m:
            evaluations[key] = float(m.group(1))

    if not evaluations:
        raise ValueError(f"could not scrape benchmarks from {url}")

    name_m = re.search(r"<title>([^<]+)</title>", html, re.I)
    name = name_m.group(1).split(" - ")[0].strip() if name_m else slug
    return {
        "slug": slug,
        "name": name,
        "evaluations": evaluations,
        "pricing": {},
        "performance": {},
    }


def fetch_model_by_slug(slug: str) -> dict:
    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "")
    if key:
        try:
            req = Request(AA_MODEL_URL.format(slug=slug), headers=_api_headers())
            with urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            return payload.get("data") or payload
        except Exception:
            pass
    return fetch_aa_public(slug)


def load_aa_cache(*, cache_path: Path | None = None, allow_stale: bool = True) -> list[dict]:
    cache = cache_path or AA_CACHE_PATH
    if cache.exists() and allow_stale:
        raw = json.loads(cache.read_text())
        data = raw.get("data") or []
        if data:
            return data
    if os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY"):
        return fetch_all_models(cache_path=cache)
    raise ValueError(
        "AA cache missing and ARTIFICIAL_ANALYSIS_API_KEY not set. "
        "Run mat-sync-aa with a key, or mat-import-aa <slug> for single-model public scrape."
    )


def find_aa_model(
    query: str,
    models: list[dict] | None = None,
) -> dict | None:
    models_list = models if models is not None else load_aa_cache()
    q = slugify(query)
    for m in models_list:
        if slugify(m.get("slug", "")) == q:
            return m
    for m in models_list:
        slug = slugify(m.get("slug", ""))
        name = slugify(m.get("name", ""))
        if q in slug or slug in q or q in name or name in q:
            return m
    return None


def find_aa_model_or_fetch(query: str, models: list[dict] | None = None) -> dict | None:
    """Cache lookup, then public page scrape for a single slug."""
    candidates = models
    if candidates is None:
        try:
            candidates = load_aa_cache()
        except ValueError:
            candidates = []
    hit = find_aa_model(query, candidates) if candidates else None
    if hit:
        return hit
    slug = slugify(query)
    try:
        return fetch_aa_public(slug)
    except (OSError, ValueError):
        return None


def connector_from_aa(
    aa_model: dict,
    *,
    endpoint: Endpoint,
    connector_id: str | None = None,
    locality: str = "local",
    contributor: str = "mat:import-aa",
    extra_notes: str = "",
) -> Connector:
    slug = aa_model.get("slug") or slugify(aa_model.get("name", "model"))
    evaluations = aa_model.get("evaluations") or {}
    pricing_raw = aa_model.get("pricing") or {}
    perf = aa_model.get("performance") or {}

    pricing = None
    in_price = pricing_raw.get("price_1m_input_tokens")
    out_price = pricing_raw.get("price_1m_output_tokens")
    if in_price is not None and out_price is not None:
        pricing = Pricing(
            input_per_1k=float(in_price) / 1000.0,
            output_per_1k=float(out_price) / 1000.0,
        )

    tps = perf.get("median_output_tokens_per_second")
    if tps is None:
        tps = aa_model.get("median_output_tokens_per_second")

    supports = Supports(
        tools=bool(evaluations.get("tau2_bench") or evaluations.get("tau2_bench_telecom")),
        reasoning=bool(
            aa_model.get("reasoning") or "reasoning" in slugify(aa_model.get("name", ""))
        ),
    )

    notes = (
        f"Capability scores from Artificial Analysis benchmark import. "
        f"AA slug: {slug}. {extra_notes}".strip()
    )

    return Connector(
        connector_version="1.1",
        id=connector_id or f"{slug}@installed",
        display_name=aa_model.get("name", slug),
        endpoint=endpoint,
        context_window=int(aa_model.get("context_window_tokens") or 131072),
        max_output_tokens=4096,
        locality=locality,  # type: ignore[arg-type]
        pricing=pricing,
        supports=supports,
        capabilities=capabilities_from_evaluations(evaluations),
        speed=Speed(
            tier=speed_tier_from_tokens_per_sec(float(tps) if tps else None),
            tokens_per_sec=float(tps) if tps else None,
        ),
        benchmarks=benchmarks_from_evaluations(slug, evaluations),
        profile=Profile(
            profile_method="benchmark_import",
            catalog="artificial_analysis",
            catalog_id=slug,
            profiled_at=datetime.now(UTC),
            contributor=contributor,
            notes=notes,
        ),
    )
