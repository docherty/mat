from __future__ import annotations

from eval.coding import build_oracle_script, parse_verdict
from eval.oracle import Task
from workers.llm import extract_code


def test_extract_code_fence():
    text = "Here:\n```python\ndef foo():\n    return 1\n```"
    assert "def foo" in extract_code(text)


def test_extract_code_body_only():
    body = "    return a + b\n"
    assert extract_code(body).strip() == "return a + b"


def test_parse_verdict():
    assert parse_verdict("ACCEPT\nlooks good")[0] == "ACCEPT"
    assert parse_verdict("REVISE\nmissing edge case")[0] == "REVISE"


def test_build_oracle_script_phase_a():
    task = Task(
        id="t",
        prompt="def add(a, b):\n",
        tests="assert add(1, 2) == 3",
        difficulty=0.1,
        required_tags={"coding": 1.0},
        solution="    return a + b\n",
    )
    script = build_oracle_script(task, "    return a + b\n")
    assert "assert add(1, 2) == 3" in script
