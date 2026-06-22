from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from connectors.loader import load_connector
from connectors.schema import Connector
from coordinator.checkpoint import load_checkpoint, save_checkpoint
from coordinator.policy import TrainedCoordinator
from eval.live_loop import LiveCodingLoop
from eval.oracle import load_tasks
from workers.mock import MockLLMWorker

EXAMPLES = Path(__file__).resolve().parents[1] / "connectors" / "examples"


@pytest.fixture
def tiny_pool() -> list[Connector]:
    return [
        load_connector(EXAMPLES / "alpha-coder.yaml"),
        load_connector(EXAMPLES / "beta-general.yaml"),
    ]


def test_checkpoint_roundtrip(tmp_path: Path):
    coord = TrainedCoordinator(np.array([1.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5]))
    path = tmp_path / "ckpt.json"
    save_checkpoint(coord, path, meta={"seed": 1})
    loaded = load_checkpoint(path)
    assert np.allclose(coord.weights, loaded.weights)


def test_mock_orchestrated_loop_passes(tiny_pool):
    tasks = load_tasks(Path(__file__).parents[1] / "eval" / "tasks" / "phase_a_tasks.json")[:3]
    # inject solutions for mock worker
    for t in tasks:
        assert t.solution

    worker = MockLLMWorker()
    loop = LiveCodingLoop(tiny_pool, worker=worker)
    for task in tasks:
        result = loop.run_orchestrated(task)
        assert result.passed, result.error


def test_benchmark_with_checkpoint(tmp_path: Path, tiny_pool, monkeypatch):
    from eval import benchmark as bench_mod

    tasks = load_tasks(Path(__file__).parents[1] / "eval" / "tasks" / "phase_a_tasks.json")[:2]
    ckpt = tmp_path / "c.json"
    save_checkpoint(TrainedCoordinator(np.zeros(9)), ckpt)

    monkeypatch.setattr(bench_mod, "load_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(
        bench_mod,
        "load_connectors_dir",
        lambda _d: tiny_pool,
    )

    def fake_loop(pool, coordinator=None, worker=None, config=None):
        return LiveCodingLoop(pool, coordinator=coordinator, worker=worker or MockLLMWorker())

    monkeypatch.setattr(bench_mod, "LiveCodingLoop", fake_loop)

    summary = bench_mod.run_benchmark(
        pool_dir=tmp_path,
        split="train",
        mode="orchestrated",
        checkpoint=ckpt,
    )
    assert summary.tasks == 2
