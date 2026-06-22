from __future__ import annotations

from pathlib import Path

import pytest

from connectors.loader import dump_connector, load_connector
from connectors.schema import Connector, compute_integrity_hash
from coordinator.policy import PromptedCoordinator
from eval.harness import best_connector_for_task
from eval.oracle import load_tasks, run_oracle
from tool_backend.sandbox import Sandbox
from traces.redact import redact_secrets

EXAMPLES = Path(__file__).resolve().parents[1] / "connectors" / "examples"


@pytest.fixture(scope="session", autouse=True)
def ensure_connector_hashes():
    import yaml

    for path in EXAMPLES.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        connector = Connector.model_validate(data)
        if not connector.profile.integrity_sha256:
            dump_connector(connector, path)


def test_oracle_passes_good_code():
    tasks = load_tasks()
    task = tasks[0]
    result = run_oracle(task.prompt + task.solution, task.tests)
    assert result.passed


def test_oracle_fails_bad_code():
    tasks = load_tasks()
    task = tasks[0]
    result = run_oracle(task.prompt + "    return a - b\n", task.tests)
    assert not result.passed


def test_connectors_load():
    for path in EXAMPLES.glob("*.yaml"):
        connector = load_connector(path)
        assert connector.connector_version in ("1.0", "1.1")
        assert len(connector.capability_vector()) == 6
        for tag in connector.capabilities:
            assert 0.0 <= connector.capabilities[tag].score <= 1.0


def test_openrouter_examples_differ_on_coding():
    flash = load_connector(EXAMPLES / "openrouter-deepseek-v4-flash.yaml")
    hy3 = load_connector(EXAMPLES / "openrouter-tencent-hy3-preview.yaml")
    assert flash.capabilities["coding"].score > hy3.capabilities["coding"].score
    assert flash.benchmarks[0].source == "artificial_analysis"


def test_integrity_hash_stable():
    connector = load_connector(EXAMPLES / "alpha-coder.yaml")
    assert connector.profile.integrity_sha256 == compute_integrity_hash(connector)


def test_prompted_routing_picks_best_coding():
    tasks = load_tasks()
    hard = max(tasks, key=lambda t: t.difficulty)
    pool = [load_connector(p) for p in sorted(EXAMPLES.glob("*.yaml")) if "heldout" not in p.name]
    pick = PromptedCoordinator().pick(hard, pool)
    best = best_connector_for_task(hard, pool)
    assert pick.id == best.id


def test_redact_secrets():
    text = "Authorization: Bearer sk-abcdefghijklmnop"
    out = redact_secrets(text)
    assert "sk-" not in out
    assert "Bearer [REDACTED]" in out


def test_sandbox_run_tests():
    sb = Sandbox(fs_scope="/tmp/mat-test-workspace")
    out = sb.run_tests("def add(a,b):\n    return a+b\n", "assert add(1,2)==3")
    assert "ERROR" not in out


def test_phase_a_smoke():
    from eval.phase_a import run_phase_a

    results = run_phase_a(generations=8, seed=0)
    assert "trained" in results
    assert results["trained"]["pass_at_1"] >= 0.0
