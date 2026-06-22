from __future__ import annotations

import re

_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
_SK = re.compile(r"sk-[A-Za-z0-9]{8,}")
_ENV_VALUE = re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+")


def redact_secrets(text: str, extra_patterns: list[str] | None = None) -> str:
    if not text:
        return text
    out = _BEARER.sub("Bearer [REDACTED]", text)
    out = _SK.sub("sk-[REDACTED]", out)
    out = _ENV_VALUE.sub(r"\1=[REDACTED]", out)
    for pattern in extra_patterns or []:
        if pattern and len(pattern) > 4:
            out = out.replace(pattern, "[REDACTED]")
    return out


def redact_dict(data: dict, secret_values: list[str] | None = None) -> dict:
    import json

    raw = json.dumps(data, default=str)
    return json.loads(redact_secrets(raw, secret_values))
