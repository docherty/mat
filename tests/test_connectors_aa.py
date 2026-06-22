from __future__ import annotations

import pytest

from connectors.aa_mapping import capabilities_from_evaluations, normalize_value
from connectors.discover_lmstudio import aa_hint_for_folder, scan_lmstudio_cache
from connectors.import_aa import slugify
from connectors.paths import EXAMPLES_DIR, default_pool_dir, is_example_connector


def test_normalize_percent():
    assert normalize_value(80.6, "percent") == pytest.approx(0.806)


def test_capabilities_from_aa_evaluations():
    caps = capabilities_from_evaluations(
        {
            "artificial_analysis_intelligence_index": 32,
            "livecodebench": 0.65,
            "tau2_bench_telecom": 90.0,
            "aa_lcr": 70.0,
        }
    )
    assert caps["coding"].score > 0.6
    assert caps["tool_use"].score == 0.9
    assert caps["reasoning"].score > 0.5


def test_slugify():
    assert "qwen3" in slugify("Qwen3.6-35B-A3B")


def test_aa_hint():
    assert aa_hint_for_folder("Qwen3.6-35B-A3B-8bit") == "qwen3-6-35b-a3b"


def test_scan_lmstudio_cache_finds_models():
    from pathlib import Path

    cache = Path.home() / ".cache" / "lm-studio" / "models"
    if not cache.exists():
        return
    found = scan_lmstudio_cache(cache)
    assert len(found) >= 1
    assert any("Qwen3.6-35B" in e["folder_name"] for e in found)


def test_example_vs_installed_paths():
    assert is_example_connector(EXAMPLES_DIR / "alpha-coder.yaml")
    pool = default_pool_dir()
    assert "connectors" in str(pool) or ".config" in str(pool)
