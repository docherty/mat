"""Profile a model and emit a connector YAML file."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from connectors.loader import dump_connector
from connectors.schema import (
    Connector,
    Endpoint,
    Pricing,
    Profile,
    Speed,
    Supports,
)
from probe_set.suite import PROBE_SUITE_VERSION, ProbeSuite, score_to_tier


def build_connector(
    *,
    connector_id: str,
    display_name: str,
    endpoint_type: str,
    base_url: str,
    model_name: str,
    auth_env: str,
    scores: dict[str, float],
    contributor: str,
) -> Connector:
    capabilities = {tag: score_to_tier(score) for tag, score in scores.items()}
    required_tags = (
        "reasoning",
        "coding",
        "long_context",
        "instruction_following",
        "verification",
        "tool_use",
    )
    for tag in required_tags:
        capabilities.setdefault(tag, "mid")

    return Connector(
        connector_version="1.0",
        id=connector_id,
        display_name=display_name,
        endpoint=Endpoint(
            type=endpoint_type,  # type: ignore[arg-type]
            base_url=base_url,
            model_name=model_name,
            auth_env=auth_env,
        ),
        pricing=Pricing(input_per_1k=0.1, output_per_1k=0.2),
        capabilities=capabilities,  # type: ignore[arg-type]
        speed=Speed(tokens_per_sec=40, tier="medium"),
        supports=Supports(),
        profile=Profile(
            probe_suite_version=PROBE_SUITE_VERSION,
            profiled_at=datetime.now(UTC),
            probe_scores=scores,
            contributor=contributor,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a model and write a connector file")
    parser.add_argument("--endpoint", default="openai")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--auth-env", required=True)
    parser.add_argument("--id", dest="connector_id")
    parser.add_argument("--display-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--contributor", default="local")
    args = parser.parse_args()

    suite = ProbeSuite()
    scores = suite.run_stub()  # TODO: live model inference

    cid = args.connector_id or args.model.replace("/", "-")
    connector = build_connector(
        connector_id=cid,
        display_name=args.display_name or args.model,
        endpoint_type=args.endpoint,
        base_url=args.base_url,
        model_name=args.model,
        auth_env=args.auth_env,
        scores=scores,
        contributor=args.contributor,
    )
    dump_connector(connector, Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
