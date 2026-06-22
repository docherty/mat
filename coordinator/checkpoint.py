from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from coordinator.policy import TrainedCoordinator

DEFAULT_CHECKPOINT = Path.home() / ".config" / "mat" / "coordinator" / "latest.json"
CHECKPOINT_VERSION = 1


def save_checkpoint(
    coordinator: TrainedCoordinator,
    path: Path | None = None,
    *,
    meta: dict | None = None,
) -> Path:
    path = path or DEFAULT_CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "trained_linear",
        "version": CHECKPOINT_VERSION,
        "weights": coordinator.weights.tolist(),
        "saved_at": datetime.now(UTC).isoformat(),
        **(meta or {}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_checkpoint(path: Path | None = None) -> TrainedCoordinator:
    path = path or DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"coordinator checkpoint not found: {path}")
    raw = json.loads(path.read_text())
    if raw.get("type") != "trained_linear":
        raise ValueError(f"unsupported checkpoint type: {raw.get('type')}")
    return TrainedCoordinator(np.array(raw["weights"], dtype=float))
