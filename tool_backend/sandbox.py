from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


class Sandbox:
    def __init__(
        self,
        *,
        fs_scope: str = "./workspace",
        timeout_sec: float = 30.0,
        network: bool = False,
    ):
        self.fs_scope = Path(fs_scope)
        self.fs_scope.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self.network = network

    def run_tests(self, code: str, tests: str) -> str:
        script = f"{code}\n\n{tests}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=self.fs_scope, delete=False) as f:
            f.write(script)
            path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=self.fs_scope,
            )
            if proc.returncode == 0:
                return proc.stdout or "ok"
            return f"ERROR: {proc.stderr or 'test failure'}"
        except subprocess.TimeoutExpired:
            return "ERROR: timeout"
        finally:
            Path(path).unlink(missing_ok=True)

    def bash(self, command: str) -> str:
        if not command.strip():
            return "ERROR: empty command"
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=self.fs_scope,
            )
            return proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return "ERROR: timeout"
