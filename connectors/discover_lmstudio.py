"""Discover models in the LM Studio cache and install mat connectors."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from connectors.dotenv import load_env
from connectors.import_aa import connector_from_aa, find_aa_model_or_fetch, load_aa_cache, slugify
from connectors.lmstudio_api import (
    clear_served_model_cache,
    fetch_served_model_ids,
    is_lmstudio_url,
    match_served_model_id,
)
from connectors.loader import dump_connector, load_connector
from connectors.paths import ensure_user_pool_dir
from connectors.pool_curated import load_curated_ids
from connectors.schema import Endpoint

DEFAULT_LMSTUDIO_CACHE = Path.home() / ".cache" / "lm-studio" / "models"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234/v1"

# Folder-name fragments → AA search hints (first match wins)
AA_HINTS: list[tuple[str, str]] = [
    (r"Qwen3\.6-35B-A3B", "qwen3-6-35b-a3b"),
    (r"Qwen3\.6-27B", "qwen3-6-27b"),
    (r"Holo3-35B", "holo3-35b-a3b"),
    (r"gemma-4-31B", "gemma-4-31b"),
    (r"gemma-4-26B", "gemma-4-26b-a4b"),
    (r"Qwen2\.5-Coder-14B", "qwen2-5-coder-14b-instruct"),
    (r"Qwen3\.5-9B", "qwen3-5-9b"),
    (r"LFM2-24B", "lfm2-24b-a2b"),
    (r"LFM2\.5-1\.2B", "lfm2-5-1.2b"),
]


def scan_lmstudio_cache(cache_dir: Path | None = None) -> list[dict]:
    cache_dir = cache_dir or DEFAULT_LMSTUDIO_CACHE
    if not cache_dir.exists():
        raise FileNotFoundError(f"LM Studio cache not found: {cache_dir}")

    found: list[dict] = []
    for vendor_dir in sorted(cache_dir.iterdir()):
        if not vendor_dir.is_dir() or vendor_dir.name.startswith("."):
            continue
        for model_dir in sorted(vendor_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            has_weights = (
                any(model_dir.glob("*.safetensors"))
                or any(model_dir.glob("*.gguf"))
                or any(model_dir.glob("*.mlx"))
                or (model_dir / "config.json").exists()
            )
            if not has_weights:
                continue
            catalog_path = f"{vendor_dir.name}/{model_dir.name}"
            found.append(
                {
                    "cache_path": str(model_dir),
                    "catalog_path": catalog_path,
                    "folder_name": model_dir.name,
                    "vendor": vendor_dir.name,
                }
            )
    return found


def lmstudio_model_name_guess(folder_name: str) -> str:
    """Heuristic id when LM Studio API is unavailable (offline discover only)."""
    base = re.sub(r"-(8bit|4bit|GGUF|MLX.*)$", "", folder_name, flags=re.I)
    if re.search(r"Qwen3\.6-35B-A3B", base, re.I):
        return "qwen3.6-35b-a3b"
    if re.search(r"Qwen3\.6-27B", base, re.I):
        return "qwen3.6-27b"
    if re.search(r"gemma-4-26B", base, re.I):
        return "gemma-4-26b-a4b-it-mlx"
    if re.search(r"gemma-4-31B", base, re.I):
        return "gemma-4-31b-it-qat-optiq"
    if re.search(r"Qwen3\.5-9B", base, re.I):
        return "qwen/qwen3.5-9b"
    if re.search(r"LFM2-24B", base, re.I):
        return "liquid/lfm2-24b-a2b"
    return slugify(base)


def aa_hint_for_folder(folder_name: str) -> str | None:
    for pattern, hint in AA_HINTS:
        if re.search(pattern, folder_name, re.I):
            return hint
    return slugify(folder_name.replace("_", "-"))


def resolve_model_name_for_entry(
    entry: dict,
    served_ids: tuple[str, ...],
    *,
    offline: bool,
) -> str | None:
    guess = lmstudio_model_name_guess(entry["folder_name"])
    if served_ids:
        matched = match_served_model_id(
            entry["folder_name"],
            entry["catalog_path"],
            served_ids,
            guess=guess,
        )
        return matched
    if offline:
        return guess
    return None


def sync_pool_lmstudio_names(
    pool_dir: Path,
    *,
    base_url: str = DEFAULT_LMSTUDIO_URL,
) -> list[tuple[str, str, str]]:
    """Update connector YAMLs with exact ids from GET /v1/models. Returns (id, old, new)."""
    if not is_lmstudio_url(base_url):
        raise ValueError(f"not an LM Studio base URL: {base_url}")
    clear_served_model_cache()
    served = fetch_served_model_ids(base_url)
    if not served:
        raise RuntimeError(f"LM Studio at {base_url} returned no chat models")

    changes: list[tuple[str, str, str]] = []
    for path in sorted(pool_dir.glob("*.yaml")):
        connector = load_connector(path)
        if not is_lmstudio_url(connector.endpoint.base_url):
            continue
        catalog = connector.profile.catalog_id or ""
        folder = catalog.split("/")[-1] if catalog else path.stem
        matched = match_served_model_id(
            folder,
            catalog,
            served,
            guess=connector.endpoint.model_name,
        )
        if not matched:
            continue
        if matched != connector.endpoint.model_name:
            old = connector.endpoint.model_name
            connector.endpoint.model_name = matched
            dump_connector(connector, path)
            changes.append((connector.id, old, matched))
    return changes


def install_from_cache(
    *,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    base_url: str = DEFAULT_LMSTUDIO_URL,
    skip_missing_aa: bool = True,
    offline: bool = False,
    only_ids: set[str] | None = None,
) -> list[Path]:
    """Write one connector per cached model with AA benchmarks and exact LM Studio model ids."""
    out_dir = out_dir or ensure_user_pool_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = None
    try:
        models = load_aa_cache(allow_stale=True)
    except ValueError:
        models = []

    served: tuple[str, ...] = ()
    if is_lmstudio_url(base_url) and not offline:
        clear_served_model_cache()
        try:
            served = fetch_served_model_ids(base_url)
        except OSError as exc:
            raise RuntimeError(
                f"cannot reach LM Studio at {base_url}: {exc}. "
                "Start the server and load models, or pass --offline."
            ) from exc
        if not served:
            raise RuntimeError(
                f"LM Studio at {base_url} returned no chat models. "
                "Load models in LM Studio, then re-run."
            )

    written: list[Path] = []
    skipped: list[str] = []
    for entry in scan_lmstudio_cache(cache_dir):
        hint = aa_hint_for_folder(entry["folder_name"])
        aa = find_aa_model_or_fetch(hint or entry["folder_name"], models)
        if aa is None and skip_missing_aa:
            continue
        if aa is None:
            raise ValueError(
                f"No Artificial Analysis match for {entry['catalog_path']!r} "
                f"(hint={hint!r}). Run mat-sync-aa or mat-import-aa <slug>."
            )

        safe_id = slugify(entry["catalog_path"].replace("/", "-"))
        connector_id = f"{safe_id}@lmstudio"
        if only_ids and connector_id not in only_ids:
            continue

        model_name = resolve_model_name_for_entry(entry, served, offline=offline)
        if not model_name:
            skipped.append(entry["catalog_path"])
            continue

        endpoint = Endpoint(
            type="openai",
            base_url=base_url,
            model_name=model_name,
            auth_env="LMSTUDIO_API_KEY",
        )

        connector = connector_from_aa(
            aa,
            endpoint=endpoint,
            connector_id=connector_id,
            locality="local",
            contributor="mat:discover-lmstudio",
            extra_notes=(
                f"LM Studio cache: {entry['cache_path']}. "
                f"API model id: {model_name!r} (from GET /v1/models). "
                f"Load this model in LM Studio for routing to succeed."
            ),
        )
        connector.display_name = f"{connector.display_name} (local)"
        connector.profile.catalog = "lmstudio"
        connector.profile.catalog_id = entry["catalog_path"]
        connector.profile.profiled_at = datetime.now(UTC)

        out_path = out_dir / f"{safe_id}.yaml"
        dump_connector(connector, out_path)
        written.append(out_path)

    if skipped:
        print(
            "skipped (not listed by LM Studio /v1/models — load them first):",
            *skipped,
            sep="\n  ",
        )
    return written


def main() -> None:
    import argparse

    load_env()
    parser = argparse.ArgumentParser(
        description="Install connectors from LM Studio cache + AA benchmarks"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_LMSTUDIO_CACHE)
    parser.add_argument("--out", type=Path, help="default: ~/.config/mat/connectors")
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument("--strict-aa", action="store_true", help="fail if any model lacks AA data")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not call LM Studio; use heuristic model names (run mat-pool sync-lmstudio later)",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        help="only install connector ids listed in this yaml (see connectors/curated/)",
    )
    args = parser.parse_args()
    only_ids = set(load_curated_ids(args.curated)) if args.curated else None
    paths = install_from_cache(
        cache_dir=args.cache,
        out_dir=args.out,
        base_url=args.base_url,
        skip_missing_aa=not args.strict_aa,
        offline=args.offline,
        only_ids=only_ids,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
