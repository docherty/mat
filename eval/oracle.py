from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TASKS_PATH = Path(__file__).parent / "tasks" / "phase_a_tasks.json"


@dataclass
class Task:
    id: str
    prompt: str
    solution: str
    tests: str
    difficulty: float
    required_tags: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            solution=data["solution"],
            tests=data["tests"],
            difficulty=float(data["difficulty"]),
            required_tags=dict(data["required_tags"]),
        )


def load_tasks(path: Path | None = None) -> list[Task]:
    raw = json.loads((path or TASKS_PATH).read_text())
    return [Task.from_dict(item) for item in raw]


@dataclass
class OracleResult:
    passed: bool
    stdout: str
    stderr: str
    error: str | None = None


def run_oracle(code: str, tests: str, *, timeout_sec: float = 5.0) -> OracleResult:
    """Execute code + tests in an isolated subprocess."""
    script = f"{code}\n\n{tests}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return OracleResult(
            passed=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=None if proc.returncode == 0 else proc.stderr or "test failure",
        )
    except subprocess.TimeoutExpired as exc:
        return OracleResult(passed=False, stdout="", stderr="", error=f"timeout: {exc}")
    except OSError as exc:
        return OracleResult(passed=False, stdout="", stderr="", error=str(exc))
    finally:
        Path(path).unlink(missing_ok=True)
