"""Trinity-style SLM coordinator (optional — requires pip install mat[slm])."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from connectors.schema import CAPABILITY_TAGS, Connector
from eval.oracle import Task

ROLES = ("thinker", "worker", "verifier")


@dataclass
class SLMCoordinatorConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    device: str = "mps"  # Apple Silicon default; use cuda on 4090


class SLMCoordinator:
    """Hidden-state + linear head over (task embedding, connector capabilities, role).

    Falls back to import error if torch/transformers not installed.
  Training uses the same CMA-ES outer loop as TrainedCoordinator.
    """

    def __init__(
        self,
        head_weights: np.ndarray,
        *,
        config: SLMCoordinatorConfig | None = None,
        backbone=None,
        tokenizer=None,
    ):
        self.config = config or SLMCoordinatorConfig()
        self.head_weights = head_weights
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._n_roles = len(ROLES)

    @property
    def dim(self) -> int:
        return len(self.head_weights)

    def _encode_text(self, text: str) -> np.ndarray:
        if self.backbone is None or self.tokenizer is None:
            # bag-of-words hash embedding fallback for tests without torch
            vec = np.zeros(64)
            for tok in text.lower().split():
                vec[hash(tok) % 64] += 1.0
            n = np.linalg.norm(vec) or 1.0
            return vec / n

        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.backbone(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1].float().cpu().numpy()
        return hidden

    def _features(self, task: Task, connector: Connector, role: str) -> np.ndarray:
        role_oh = np.zeros(self._n_roles)
        if role in ROLES:
            role_oh[ROLES.index(role)] = 1.0
        text = f"task difficulty {task.difficulty:.2f} role {role} tags {task.required_tags}"
        emb = self._encode_text(text)
        cap = np.array(connector.capability_vector())
        # truncate/pad embedding to fixed width for head
        emb = emb[:32] if len(emb) > 32 else np.pad(emb, (0, max(0, 32 - len(emb))))
        return np.concatenate([emb, cap, role_oh, [task.difficulty]])

    def score(self, task: Task, connector: Connector, *, role: str) -> float:
        x = self._features(task, connector, role)
        w = self.head_weights[: len(x)]
        return float(np.dot(w, x[: len(w)]))

    def pick(self, task: Task, pool: list[Connector], *, role: str) -> Connector:
        scores = [self.score(task, c, role=role) for c in pool]
        return pool[int(np.argmax(scores))]

    @classmethod
    def from_pretrained(cls, head_weights: np.ndarray | None = None) -> SLMCoordinator:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("pip install mat[slm] for SLMCoordinator") from exc

        cfg = SLMCoordinatorConfig()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        backbone = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16,
        ).to(cfg.device)
        backbone.eval()
        dim = 32 + len(CAPABILITY_TAGS) + len(ROLES) + 1
        weights = head_weights if head_weights is not None else np.zeros(dim)
        return cls(weights, config=cfg, backbone=backbone, tokenizer=tokenizer)
