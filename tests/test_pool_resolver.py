from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from connectors.loader import dump_connector
from connectors.pool_resolver import index_library, load_active_ids, load_active_pool
from connectors.schema import CapabilityDim, Connector, Endpoint, Profile, Speed, Supports


def _connector(cid: str, *, base_url: str = "http://127.0.0.1:1234/v1") -> Connector:
    return Connector(
        connector_version="1.1",
        id=cid,
        display_name=cid,
        endpoint=Endpoint(type="openai", base_url=base_url, model_name="m", auth_env="X"),
        context_window=8192,
        max_output_tokens=1024,
        modalities=["text"],
        locality="local",
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
            profile_method="hand",
            catalog="test",
            catalog_id=cid,
            profiled_at=datetime.now(UTC),
            contributor="t",
        ),
    )


def test_load_active_ids_accepts_dict_or_list(tmp_path: Path):
    p1 = tmp_path / "active.yaml"
    p1.write_text("connectors:\n  - a\n  - b\n")
    assert load_active_ids(p1) == ["a", "b"]

    p2 = tmp_path / "active2.yaml"
    p2.write_text("- a\n- b\n")
    assert load_active_ids(p2) == ["a", "b"]


def test_index_library_rejects_duplicate_ids(tmp_path: Path):
    lib = tmp_path / "library"
    lib.mkdir()
    dump_connector(_connector("dup@x"), lib / "a.yaml")
    dump_connector(_connector("dup@x"), lib / "b.yaml")
    with pytest.raises(ValueError, match="duplicate connector id"):
        index_library(lib)


def test_load_active_pool_errors_on_missing(tmp_path: Path):
    lib = tmp_path / "library"
    lib.mkdir()
    dump_connector(_connector("present@x"), lib / "present.yaml")
    active = tmp_path / "active.yaml"
    active.write_text("connectors:\n  - missing@x\n")
    with pytest.raises(ValueError, match="missing connector"):
        load_active_pool(library_dir=lib, manifest=active)

