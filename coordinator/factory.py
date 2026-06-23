from __future__ import annotations

import os
from pathlib import Path

from coordinator.checkpoint import load_checkpoint
from coordinator.policy import PromptedCoordinator, TrainedCoordinator
from eval.live_loop import RoleCoordinator


def load_role_coordinator(
    *,
    checkpoint: Path | None = None,
    style: str | None = None,
) -> RoleCoordinator:
    """Build RoleCoordinator from env or explicit checkpoint.

    MAT_COORDINATOR: prompted | trained (default: trained if checkpoint exists)
    MAT_CHECKPOINT: path to coordinator JSON
    """
    ckpt_path = checkpoint or _checkpoint_path()
    style = (style or os.environ.get("MAT_COORDINATOR", "")).lower()

    if style == "prompted" or not ckpt_path:
        return RoleCoordinator(PromptedCoordinator())

    if ckpt_path.exists():
        try:
            return RoleCoordinator(load_checkpoint(ckpt_path))
        except (OSError, ValueError, KeyError):
            pass

    if style == "trained":
        return RoleCoordinator(TrainedCoordinator.from_flat(_zeros()))

    return RoleCoordinator(PromptedCoordinator())


def _checkpoint_path() -> Path | None:
    raw = os.environ.get("MAT_CHECKPOINT", "")
    if raw:
        return Path(raw).expanduser()
    default = Path.home() / ".config" / "mat" / "coordinator" / "latest.json"
    return default if default.exists() else None


def _zeros():
    import numpy as np

    from coordinator.train import _heuristic_weights

    return np.array(_heuristic_weights())
