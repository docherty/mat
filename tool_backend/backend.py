from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tool_backend.sandbox import Sandbox


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


class ToolBackend:
    def __init__(self, *, max_iterations: int = 8, sandbox: Sandbox | None = None):
        self.max_iterations = max_iterations
        self.sandbox = sandbox or Sandbox()

    def parse_openai(self, message: dict) -> list[ToolCall]:
        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        return calls

    def parse_anthropic(self, content: list[dict]) -> list[ToolCall]:
        calls = []
        for block in content:
            if block.get("type") == "tool_use":
                calls.append(ToolCall(name=block.get("name", ""), arguments=block.get("input", {})))
        return calls

    def parse_prompt_injected(self, text: str) -> list[ToolCall]:
        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        calls = []
        for match in pattern.finditer(text):
            data = json.loads(match.group(1))
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        return calls

    def execute_internal(self, call: ToolCall) -> ToolResult:
        if call.name == "run_tests":
            code = call.arguments.get("code", "")
            tests = call.arguments.get("tests", "")
            out = self.sandbox.run_tests(code, tests)
            err = out.startswith("ERROR")
            return ToolResult(call_id=call.name, name=call.name, content=out, is_error=err)
        if call.name == "bash":
            out = self.sandbox.bash(call.arguments.get("command", ""))
            return ToolResult(call_id=call.name, name=call.name, content=out)
        return ToolResult(
            call_id=call.name,
            name=call.name,
            content="unknown internal tool",
            is_error=True,
        )
