from __future__ import annotations

import json
import os
from pathlib import Path

from coordinator.checkpoint import checkpoint_type, load_checkpoint
from coordinator.policy import PromptedCoordinator, TrainedCoordinator
from coordinator.slm_coordinator import SLMCoordinator
from eval.live_loop import RoleCoordinator


def load_role_coordinator(
    *,
    checkpoint: Path | None = None,
    style: str | None = None,
) -> RoleCoordinator:
    """Build RoleCoordinator from env or explicit checkpoint.

    MAT_COORDINATOR: prompted | trained | slm
    MAT_CHECKPOINT: path to coordinator JSON
    """
    ckpt_path = checkpoint or _checkpoint_path()
    style = (style or os.environ.get("MAT_COORDINATOR", "")).lower()

    if ckpt_path and ckpt_path.exists():
        ckpt_kind = checkpoint_type(ckpt_path) or ""
        if ckpt_kind == "slm_linear" or style == "slm":
            return RoleCoordinator(_load_slm_coordinator(ckpt_path))
        if ckpt_kind == "trained_linear":
            return RoleCoordinator(load_checkpoint(ckpt_path))

    if style == "slm":
        return RoleCoordinator(_load_slm_coordinator(None))

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


def _load_slm_coordinator(ckpt_path: Path | None) -> SLMCoordinator:
    if ckpt_path and ckpt_path.exists():
        data = json.loads(ckpt_path.read_text())
        if data.get("type") == "slm_linear":
            base = SLMCoordinator.from_pretrained()
            weights = __import__("numpy").array(data["weights"], dtype=float)
            return SLMCoordinator(
                weights,
                config=base.config,
                backbone=base.backbone,
                tokenizer=base.tokenizer,
            )
    return SLMCoordinator.from_pretrained()


def _checkpoint_path() -> Path | None:
    raw = os.environ.get("MAT_CHECKPOINT", "")
    if raw:
        return Path(raw).expanduser()
    for candidate in (
        Path.home() / ".config" / "mat" / "coordinator" / "latest_slm.json",
        Path.home() / ".config" / "mat" / "coordinator" / "latest.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _zeros():
    import numpy as np

    from coordinator.train import _heuristic_weights

    return np.array(_heuristic_weights())
