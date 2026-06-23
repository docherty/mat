from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from connectors.loader import load_connector
from connectors.schema import Connector
from coordinator.checkpoint import save_checkpoint
from coordinator.factory import load_role_coordinator
from coordinator.policy import TrainedCoordinator
from loop.coding_detect import extract_coding_prompt, guess_entry_point, is_coding_request
from loop.runner import OrchestrationLoop
from workers.mock import MockLLMWorker

EXAMPLES = Path(__file__).resolve().parents[1] / "connectors" / "examples"


@pytest.fixture
def tiny_pool() -> list[Connector]:
    return [
        load_connector(EXAMPLES / "alpha-coder.yaml"),
        load_connector(EXAMPLES / "beta-general.yaml"),
    ]


def test_coding_detect():
    msgs = [{"role": "user", "content": "Write a function def add(a, b):"}]
    assert is_coding_request(msgs)
    assert extract_coding_prompt(msgs).startswith("Write a function")
    assert guess_entry_point("def add(a, b):\n    pass") == "add"


def test_load_role_coordinator_prompted(monkeypatch):
    monkeypatch.delenv("MAT_CHECKPOINT", raising=False)
    monkeypatch.setenv("MAT_COORDINATOR", "prompted")
    coord = load_role_coordinator()
    assert coord.base.__class__.__name__ == "PromptedCoordinator"


def test_load_role_coordinator_from_checkpoint(tmp_path: Path, monkeypatch):
    import numpy as np

    ckpt = tmp_path / "c.json"
    save_checkpoint(TrainedCoordinator(np.zeros(9)), ckpt)
    monkeypatch.setenv("MAT_CHECKPOINT", str(ckpt))
    coord = load_role_coordinator()
    assert isinstance(coord.base, TrainedCoordinator)


def test_orchestration_loop_live_mock(tiny_pool):
    worker = MockLLMWorker()
    loop = OrchestrationLoop(tiny_pool, live=True, worker=worker)
    messages = [{"role": "user", "content": "Explain orchestration in one sentence."}]
    result = loop.run(messages)
    assert result.answer
    assert result.input_tokens > 0
    assert result.steps >= 1


def test_orchestration_loop_simulated_offline(tiny_pool, monkeypatch):
    monkeypatch.setenv("MAT_LIVE", "0")
    loop = OrchestrationLoop.from_env(tiny_pool)
    result = loop.run([{"role": "user", "content": "hello"}])
    assert "proxy" in result.answer.lower()


def test_mat_serve_health_and_chat(tiny_pool, tmp_path: Path, monkeypatch):
    from connectors.loader import dump_connector

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    for c in tiny_pool:
        dump_connector(c, pool_dir / f"{c.id}.yaml")

    monkeypatch.setenv("MAT_GATEWAY_KEY", "test-key")
    monkeypatch.setenv("MAT_LIVE", "0")
    app = create_app(str(pool_dir))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pool_size"] == 2

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={"model": "balanced", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["x_mat"]["stages"]
