"""Build connectors from mat probes or external catalogs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from connectors.dotenv import load_env
from connectors.loader import dump_connector
from connectors.schema import (
    CURRENT_PROBE_SUITE,
    CapabilityDim,
    Connector,
    Endpoint,
    Pricing,
    Profile,
    Supports,
)
from probe_set.suite import PROBE_SUITE_VERSION, ProbeSuite

CAPABILITY_TAGS = (
    "reasoning",
    "coding",
    "long_context",
    "instruction_following",
    "verification",
    "tool_use",
)


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
    profile_method: str = "mat_probe",
) -> Connector:
    capabilities = {tag: CapabilityDim.from_score(scores[tag]) for tag in CAPABILITY_TAGS}

    return Connector(
        connector_version="1.1",
        id=connector_id,
        display_name=display_name,
        endpoint=Endpoint(
            type=endpoint_type,  # type: ignore[arg-type]
            base_url=base_url,
            model_name=model_name,
            auth_env=auth_env,
        ),
        pricing=Pricing(input_per_1k=0.1, output_per_1k=0.2),
        capabilities=capabilities,
        supports=Supports(),
        profile=Profile(
            profile_method=profile_method,  # type: ignore[arg-type]
            catalog="openrouter" if "openrouter" in base_url else None,
            catalog_id=model_name if "openrouter" in base_url else None,
            probe_suite_version=PROBE_SUITE_VERSION if profile_method == "mat_probe" else None,
            profiled_at=datetime.now(UTC),
            contributor=contributor,
        ),
    )


def main() -> None:
    load_env()
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
    raw_scores = suite.run_stub()  # TODO: live model inference
    scores = {tag: raw_scores.get(tag, 0.5) for tag in CAPABILITY_TAGS}

    cid = args.connector_id or args.model.replace("/", "-") + "@openrouter"
    connector = build_connector(
        connector_id=cid,
        display_name=args.display_name or args.model,
        endpoint_type=args.endpoint,
        base_url=args.base_url,
        model_name=args.model,
        auth_env=args.auth_env,
        scores=scores,
        contributor=args.contributor,
        profile_method="mat_probe",
    )
    dump_connector(connector, Path(args.output))
    print(f"wrote {args.output} (suite {CURRENT_PROBE_SUITE})")


if __name__ == "__main__":
    main()
