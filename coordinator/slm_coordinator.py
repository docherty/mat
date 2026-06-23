"""Trinity-style SLM coordinator (optional — requires pip install mat[slm])."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from connectors.schema import CAPABILITY_TAGS, Connector
from eval.harness import connector_score
from eval.oracle import Task

ROLES = ("thinker", "worker", "verifier")
EMBED_WIDTH = 32  # hash fallback width when torch absent


@dataclass
class SLMCoordinatorConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    device: str = "mps"
    hidden_size: int = EMBED_WIDTH

    @classmethod
    def from_env(cls) -> SLMCoordinatorConfig:
        return cls(
            model_id=os.environ.get("MAT_SLM_MODEL", cls.model_id),
            device=os.environ.get("MAT_SLM_DEVICE", cls.device),
        )


def slm_feature_dim(hidden_size: int) -> int:
    return hidden_size + len(CAPABILITY_TAGS) + len(ROLES) + 1


class SLMCoordinator:
    """SLM hidden state + linear head; transcript-conditioned routing."""

    def __init__(
        self,
        head_weights: np.ndarray,
        *,
        config: SLMCoordinatorConfig | None = None,
        backbone=None,
        tokenizer=None,
    ):
        self.config = config or SLMCoordinatorConfig.from_env()
        self.head_weights = np.asarray(head_weights, dtype=float)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._n_roles = len(ROLES)
        if self.backbone is not None:
            hs = getattr(getattr(self.backbone, "config", None), "hidden_size", None)
            if hs:
                self.config.hidden_size = int(hs)

    @property
    def dim(self) -> int:
        return slm_feature_dim(self.config.hidden_size)

    def _encode_text(self, text: str) -> np.ndarray:
        if self.backbone is None or self.tokenizer is None:
            vec = np.zeros(EMBED_WIDTH)
            for tok in text.lower().split():
                vec[hash(tok) % EMBED_WIDTH] += 1.0
            n = np.linalg.norm(vec) or 1.0
            return vec / n

        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.backbone(**inputs, output_hidden_states=True)
            # Penultimate token — Trinity uses pre-EOS hidden state.
            hidden = out.hidden_states[-1][0, -2].float().cpu().numpy()
        return hidden

    def _features(self, task: Task, connector: Connector, role: str, *, transcript: str = "") -> np.ndarray:
        role_oh = np.zeros(self._n_roles)
        if role in ROLES:
            role_oh[ROLES.index(role)] = 1.0
        prompt = task.prompt[:2000]
        tail = transcript[-3000:] if transcript else ""
        text = f"role={role}\ndifficulty={task.difficulty:.2f}\n{prompt}\n---\n{tail}"
        emb = self._encode_text(text)
        hs = self.config.hidden_size
        if len(emb) < hs:
            emb = np.pad(emb, (0, hs - len(emb)))
        elif len(emb) > hs:
            emb = emb[:hs]
        cap = np.array(connector.capability_vector())
        return np.concatenate([emb, cap, role_oh, [task.difficulty]])

    def score(self, task: Task, connector: Connector, *, role: str, transcript: str = "") -> float:
        base = connector_score(task, connector)
        x = self._features(task, connector, role, transcript=transcript)
        w = self.head_weights[: len(x)]
        return base + float(np.dot(w, x[: len(w)]))

    def pick(self, task: Task, pool: list[Connector], *, role: str, transcript: str = "") -> Connector:
        scores = [self.score(task, c, role=role, transcript=transcript) for c in pool]
        return pool[int(np.argmax(scores))]

    @classmethod
    def from_pretrained(cls, head_weights: np.ndarray | None = None) -> SLMCoordinator:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("pip install mat[slm] for SLMCoordinator") from exc

        cfg = SLMCoordinatorConfig.from_env()
        if cfg.device == "mps" and not torch.backends.mps.is_available():
            cfg.device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        backbone = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16 if cfg.device != "cpu" else torch.float32,
        ).to(cfg.device)
        backbone.eval()
        hs = int(backbone.config.hidden_size)
        cfg.hidden_size = hs
        dim = slm_feature_dim(hs)
        weights = head_weights if head_weights is not None else np.zeros(dim)
        return cls(weights, config=cfg, backbone=backbone, tokenizer=tokenizer)
