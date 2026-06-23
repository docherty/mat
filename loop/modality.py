from __future__ import annotations

from connectors.schema import Connector


def required_modalities(messages: list[dict]) -> set[str]:
    """Infer required input modalities from OpenAI-style chat messages.

    We only gate on obvious multimodal content blocks; default is text.
    """
    req: set[str] = {"text"}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type") or "")
                if ptype == "image_url":
                    req.add("image")
                elif ptype == "input_audio":
                    req.add("audio")
                elif ptype == "video_url":
                    req.add("video")
    return req


def compatible_pool(pool: list[Connector], *, required: set[str]) -> list[Connector]:
    """Return connectors that support all required modalities."""
    out: list[Connector] = []
    for c in pool:
        offered = set(c.modalities or ["text"])
        if required.issubset(offered):
            out.append(c)
    return out

