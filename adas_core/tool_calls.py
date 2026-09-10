"""
Tool call execution and validation utilities for ADAS.

Provides:
- validate_tool_history: Strict tool-call protocol validator rejecting unanswered,
  orphaned, malformed, or duplicate ToolMessages.
- execute_tool_calls: Runtime helper executing tool calls from an AIMessage against
  an available tools dictionary.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from adas_core.exceptions import ToolProtocolError


def validate_tool_history(messages: list[BaseMessage]) -> None:
    """
    Validate that each AI tool call has a following matching ToolMessage.
    Rejects malformed, orphaned, and duplicate tool messages.
    """
    active_pending_calls: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    answered_call_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            if active_pending_calls:
                missing = sorted(active_pending_calls.keys())
                raise ToolProtocolError(
                    f"AIMessage at index {idx} encountered while prior tool calls {missing} remain unanswered."
                )

            tool_calls = getattr(msg, "tool_calls", []) or []
            invalid_calls = getattr(msg, "invalid_tool_calls", []) or []
            for call in tool_calls + invalid_calls:
                cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                cname = call.get("name") if isinstance(call, dict) else getattr(call, "name", "unknown")
                if cid:
                    seen_call_ids.add(cid)
                    active_pending_calls[cid] = str(cname)

        elif isinstance(msg, ToolMessage):
            cid = getattr(msg, "tool_call_id", None)
            if not cid:
                raise ToolProtocolError(f"Malformed ToolMessage at index {idx}: missing or empty 'tool_call_id'.")
            if cid not in seen_call_ids:
                raise ToolProtocolError(
                    f"Orphaned ToolMessage at index {idx} with tool_call_id='{cid}' has no matching prior AIMessage tool call."
                )
            if cid in answered_call_ids:
                raise ToolProtocolError(f"Duplicate ToolMessage at index {idx} for tool_call_id='{cid}'.")

            answered_call_ids.add(cid)
            active_pending_calls.pop(cid, None)

        else:
            if active_pending_calls:
                missing = sorted(active_pending_calls.keys())
                raise ToolProtocolError(
                    f"Message {type(msg).__name__} at index {idx} encountered while tool calls {missing} remain unanswered."
                )

    if active_pending_calls:
        missing = sorted(active_pending_calls.keys())
        raise ToolProtocolError(
            f"Cannot invoke model: AIMessage tool calls {missing} have no corresponding ToolMessages."
        )


def execute_tool_calls(
    response: AIMessage, available_tools: dict[str, BaseTool]
) -> tuple[list[ToolMessage], dict[str, Any]]:
    """Execute available tool calls from an AIMessage response."""
    tool_messages: list[ToolMessage] = []
    tool_results: dict[str, Any] = {}

    for invalid_call in getattr(response, "invalid_tool_calls", []) or []:
        if isinstance(invalid_call, dict):
            err = invalid_call.get("error")
            cid = invalid_call.get("id")
            name = invalid_call.get("name")
        else:
            err = getattr(invalid_call, "error", None)
            cid = getattr(invalid_call, "id", None)
            name = getattr(invalid_call, "name", None)

        content = f"Failed to parse tool call. Error: {err}"
        tool_messages.append(ToolMessage(content=content, tool_call_id=cid, name=name))

    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls:
        return tool_messages, tool_results

    for tool_call in tool_calls:
        if not tool_call:
            continue

        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {}) or {}
            tool_id = tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {}) or {}
            tool_id = getattr(tool_call, "id", None)

        if not tool_name:
            tool_messages.append(
                ToolMessage(
                    content="Malformed tool call: missing name",
                    tool_call_id=tool_id,
                    name=None,
                )
            )
            continue

        if tool_name in available_tools:
            try:
                result = available_tools[tool_name].invoke(tool_args)
                content = str(result) if result is not None else f"Tool {tool_name} executed successfully."
                tool_messages.append(ToolMessage(content=content, tool_call_id=tool_id, name=tool_name))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {e!r}"
                tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_id, name=tool_name))
                tool_results[tool_name] = error_message
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found.",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

    return tool_messages, tool_results


__all__ = [
    "execute_tool_calls",
    "validate_tool_history",
]
