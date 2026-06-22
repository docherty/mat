from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traces.redact import redact_dict


class TraceLogger:
    def __init__(self, directory: str | Path = "traces", secret_values: list[str] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.secret_values = secret_values or []
        self._path = self.directory / f"run-{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"

    def log(self, event: str, payload: dict[str, Any], *, seed: int | None = None) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "seed": seed,
            "payload": redact_dict(payload, self.secret_values),
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
