from datetime import UTC, datetime

from connectors.schema import CapabilityDim, Connector, Endpoint, Profile, Speed, Supports


def _conn(cid: str, model: str, *, locality: str = "api") -> Connector:
    return Connector(
        connector_version="1.1",
        id=cid,
        display_name=cid,
        endpoint=Endpoint(
            type="openai",
            base_url="https://api.venice.ai/api/v1",
            model_name=model,
            auth_env="VENICE_API_KEY",
        ),
        context_window=8192,
        max_output_tokens=1024,
        modalities=["text"],
        locality=locality,  # type: ignore[arg-type]
        pricing=None,
        supports=Supports(),
        tool_format="openai",
        capabilities={
            k: CapabilityDim.from_score(0.5)
            for k in (
                "reasoning",
                "coding",
                "long_context",
                "instruction_following",
                "verification",
                "tool_use",
            )
        },
        speed=Speed(),
        benchmarks=[],
        profile=Profile(
            profile_method="benchmark_import",
            catalog="test",
            catalog_id=cid,
            profiled_at=datetime.now(UTC),
            contributor="t",
        ),
    )


def test_build_pool_health_flags_missing_pricing():
    from api.pool_health import build_pool_health

    health = build_pool_health([_conn("a@venice", "m")])
    assert health["status"] == "degraded"
    assert any("pricing" in i for i in health["issues"])
