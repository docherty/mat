"""CLI: import one connector from AA + optional endpoint override."""

from __future__ import annotations

import argparse
from pathlib import Path

from connectors.import_aa import (
    connector_from_aa,
    fetch_model_by_slug,
    find_aa_model,
    load_aa_cache,
)
from connectors.loader import dump_connector
from connectors.paths import ensure_user_pool_dir
from connectors.schema import Endpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import connector from Artificial Analysis benchmarks"
    )
    parser.add_argument("slug", help="AA model slug, e.g. qwen3-6-35b-a3b")
    parser.add_argument("--out", type=Path, help="default: ~/.config/mat/connectors/<slug>.yaml")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-name", help="API model id (OpenRouter slug, LM Studio name, etc.)")
    parser.add_argument("--local", action="store_true", help="local LM Studio endpoint")
    parser.add_argument("--connector-id")
    args = parser.parse_args()

    try:
        aa = fetch_model_by_slug(args.slug)
    except Exception:
        aa = find_aa_model(args.slug, load_aa_cache())
    if not aa:
        raise SystemExit(f"model not found in AA: {args.slug}")

    base_url = "http://127.0.0.1:1234/v1" if args.local else args.base_url
    model_name = args.model_name or aa.get("slug") or args.slug
    auth = "LMSTUDIO_API_KEY" if args.local else "OPENROUTER_API_KEY"

    endpoint = Endpoint(type="openai", base_url=base_url, model_name=model_name, auth_env=auth)
    connector = connector_from_aa(
        aa,
        endpoint=endpoint,
        connector_id=args.connector_id,
        locality="local" if args.local else "api",
    )
    if args.local:
        connector.profile.catalog = "lmstudio"

    out_dir = ensure_user_pool_dir()
    out = args.out or out_dir / f"{args.slug}.yaml"
    dump_connector(connector, out)
    print(out)


if __name__ == "__main__":
    main()
