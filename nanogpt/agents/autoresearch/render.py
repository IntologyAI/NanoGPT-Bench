#!/usr/bin/env python3
"""Render Claude stream-json events as readable text."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

Json = dict[str, Any]


@dataclass
class Tool:
    """Track one in-flight tool use block."""

    name: str
    payload: str = ""


class Render:
    """Convert Claude stream-json output into readable text."""

    def __init__(self) -> None:
        self.blocks: dict[int, str] = {}
        self.tools: dict[int, Tool] = {}

    def run(self) -> None:
        """Read newline-delimited JSON events from stdin."""

        for raw in sys.stdin:
            if raw.strip():
                self.handle(json.loads(raw))

    def handle(self, record: Json) -> None:
        """Dispatch one parsed record."""

        record_type = str(record.get("type", ""))
        if record_type == "stream_event":
            self.handle_stream(record["event"])
            return
        if record_type == "system":
            self.handle_system(record)
            return
        if record_type == "user":
            self.handle_user(record)

    def handle_stream(self, event: Json) -> None:
        """Render one stream event."""

        event_type = str(event.get("type", ""))
        if event_type == "content_block_start":
            self.start_block(event)
            return
        if event_type == "content_block_delta":
            self.delta_block(event)
            return
        if event_type == "content_block_stop":
            self.stop_block(event)

    def start_block(self, event: Json) -> None:
        """Record the start of a streamed content block."""

        index = int(event["index"])
        block = event["content_block"]
        block_type = str(block["type"])
        self.blocks[index] = block_type
        if block_type == "tool_use":
            self.tools[index] = Tool(name=str(block["name"]))

    def delta_block(self, event: Json) -> None:
        """Render one streamed content delta."""

        index = int(event["index"])
        block_type = self.blocks.get(index, "")
        delta = event["delta"]
        delta_type = str(delta.get("type", ""))
        if block_type == "text" and delta_type == "text_delta":
            self.emit(str(delta["text"]))
            return
        if block_type == "tool_use" and delta_type == "input_json_delta":
            self.tools[index].payload += str(delta["partial_json"])

    def stop_block(self, event: Json) -> None:
        """Render the end of a content block."""

        index = int(event["index"])
        block_type = self.blocks.pop(index, "")
        if block_type == "text":
            self.emit("\n\n")
            return
        if block_type == "tool_use":
            tool = self.tools.pop(index)
            self.emit(f"{self.format_tool(tool)}\n\n")

    def handle_system(self, record: Json) -> None:
        """Render one system event."""

        subtype = str(record.get("subtype", ""))
        if subtype == "task_started":
            description = self.clean(record.get("description", ""))
            self.emit(f"[task] {description}\n")
            return
        if subtype == "task_progress":
            description = self.clean(record.get("description", ""))
            self.emit(f"[progress] {description}\n")
            return
        if subtype == "task_completed":
            description = self.clean(record.get("description", ""))
            self.emit(f"[done] {description}\n")
            return
        if subtype == "api_retry":
            attempt = record.get("attempt", "")
            max_retries = record.get("max_retries", "")
            delay = record.get("retry_delay_ms", "")
            self.emit(f"[retry] attempt {attempt}/{max_retries} in {delay}ms\n")

    def handle_user(self, record: Json) -> None:
        """Render tool results from the assistant runtime."""

        message = record.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")) != "tool_result":
                continue
            text = self.result_text(item.get("content"))
            if text:
                self.emit(f"[result]\n{text}\n\n")

    def format_tool(self, tool: Tool) -> str:
        """Summarize one tool call."""

        payload = json.loads(tool.payload) if tool.payload else {}
        name = tool.name
        if name == "Bash":
            description = self.clean(payload.get("description", ""))
            command = self.clean(payload.get("command", ""), limit=240)
            if description and command:
                return f"[tool {name}] {description}: {command}"
            return f"[tool {name}] {description or command}"
        if name in {"Read", "Write", "Edit"}:
            path = self.clean(payload.get("file_path", ""))
            return f"[tool {name}] {path}"
        if name == "NotebookEdit":
            path = self.clean(payload.get("notebook_path", ""))
            return f"[tool {name}] {path}"
        if name == "Agent":
            description = self.clean(payload.get("description", ""))
            prompt = self.clean(payload.get("prompt", ""), limit=240)
            return f"[tool {name}] {description or prompt}"
        description = self.clean(payload.get("description", ""))
        return f"[tool {name}] {description}".rstrip()

    def result_text(self, content: Any) -> str:
        """Extract a readable excerpt from a tool result payload."""

        if isinstance(content, str):
            return self.clean_lines(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and str(item.get("type", "")) == "text":
                    parts.append(str(item.get("text", "")))
            return self.clean_lines("\n".join(parts))
        return ""

    def clean(self, value: Any, limit: int = 160) -> str:
        """Collapse whitespace and truncate long strings."""

        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def clean_lines(self, text: str, limit: int = 1200) -> str:
        """Normalize a multi-line payload excerpt."""

        stripped = self.strip_reminders(text).strip()
        if len(stripped) <= limit:
            return stripped
        return f"{stripped[: limit - 3]}..."

    def strip_reminders(self, text: str) -> str:
        """Drop system reminder blocks from rendered tool output."""

        lines: list[str] = []
        skipping = False
        for line in text.splitlines():
            if line.strip() == "<system-reminder>":
                skipping = True
                continue
            if line.strip() == "</system-reminder>":
                skipping = False
                continue
            if not skipping:
                lines.append(line)
        return "\n".join(lines)

    def emit(self, text: str) -> None:
        """Write text to stdout immediately."""

        sys.stdout.write(text)
        sys.stdout.flush()


def main() -> None:
    """Run the renderer."""

    Render().run()


if __name__ == "__main__":
    main()
