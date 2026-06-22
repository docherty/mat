"""Discover models in the LM Studio cache and install mat connectors."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from connectors.import_aa import connector_from_aa, find_aa_model, load_aa_cache, slugify
from connectors.paths import ensure_user_pool_dir
from connectors.schema import Endpoint
from connectors.loader import dump_connector

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


def lmstudio_model_name(folder_name: str) -> str:
    """Best-guess id for LM Studio OpenAI API when this folder is loaded."""
    base = re.sub(r"-(8bit|4bit|GGUF|MLX.*)$", "", folder_name, flags=re.I)
    if re.search(r"Qwen3\.6-35B-A3B", base, re.I):
        return "qwen3.6-35b-a3b"
    return slugify(base)


def aa_hint_for_folder(folder_name: str) -> str | None:
    for pattern, hint in AA_HINTS:
        if re.search(pattern, folder_name, re.I):
            return hint
    return slugify(folder_name.replace("_", "-"))


def install_from_cache(
    *,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    base_url: str = DEFAULT_LMSTUDIO_URL,
    port_offset: int = 0,
    skip_missing_aa: bool = False,
) -> list[Path]:
    """Write one connector per cached model with AA benchmark_import scores."""
    out_dir = out_dir or ensure_user_pool_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_aa_cache()
    written: list[Path] = []

    for i, entry in enumerate(scan_lmstudio_cache(cache_dir)):
        hint = aa_hint_for_folder(entry["folder_name"])
        aa = find_aa_model(hint or entry["folder_name"], models) if hint else None
        if aa is None:
            aa = find_aa_model(entry["folder_name"], models)
        if aa is None and skip_missing_aa:
            continue
        if aa is None:
            raise ValueError(
                f"No Artificial Analysis match for {entry['catalog_path']!r} "
                f"(hint={hint!r}). Run mat-sync-aa or pass a manual mat-import-aa."
            )

        slug = aa.get("slug") or slugify(aa.get("name", "model"))
        safe_id = slugify(entry["catalog_path"].replace("/", "-"))
        model_name = lmstudio_model_name(entry["folder_name"])

        endpoint = Endpoint(
            type="openai",
            base_url=base_url,
            model_name=model_name,
            auth_env="LMSTUDIO_API_KEY",
        )

        connector = connector_from_aa(
            aa,
            endpoint=endpoint,
            connector_id=f"{safe_id}@lmstudio",
            locality="local",
            contributor="mat:discover-lmstudio",
            extra_notes=(
                f"LM Studio cache: {entry['cache_path']}. "
                f"Load this folder in LM Studio; set endpoint.model_name to match "
                f"GET /v1/models if needed."
            ),
        )
        connector.display_name = f"{connector.display_name} (local)"
        connector.profile.catalog = "lmstudio"
        connector.profile.catalog_id = entry["catalog_path"]
        connector.profile.profiled_at = datetime.now(UTC)

        out_path = out_dir / f"{safe_id}.yaml"
        dump_connector(connector, out_path)
        written.append(out_path)

    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Install connectors from LM Studio cache + AA benchmarks")
    parser.add_argument("--cache", type=Path, default=DEFAULT_LMSTUDIO_CACHE)
    parser.add_argument("--out", type=Path, help="default: ~/.config/mat/connectors")
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument("--skip-missing-aa", action="store_true")
    args = parser.parse_args()
    paths = install_from_cache(
        cache_dir=args.cache,
        out_dir=args.out,
        base_url=args.base_url,
        skip_missing_aa=args.skip_missing_aa,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
