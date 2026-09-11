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
from adas_core.logging_config import get_logger

logger = get_logger("adas_core.tool_calls")


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
                if cid and str(cid).strip():
                    cid_str = str(cid).strip()
                    seen_call_ids.add(cid_str)
                    active_pending_calls[cid_str] = str(cname)

        elif isinstance(msg, ToolMessage):
            raw_cid = getattr(msg, "tool_call_id", None)
            cid = str(raw_cid).strip() if raw_cid is not None and str(raw_cid).strip() else None
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
    """Execute available tool calls from an AIMessage response.

    - Only emits ToolMessage objects for tool calls having valid, non-empty IDs.
    - Malformed calls or calls lacking valid IDs do not emit ToolMessages into strict history.
    """
    tool_messages: list[ToolMessage] = []
    tool_results: dict[str, Any] = {}

    for idx, invalid_call in enumerate(getattr(response, "invalid_tool_calls", []) or []):
        if isinstance(invalid_call, dict):
            err = invalid_call.get("error")
            cid = invalid_call.get("id")
            name = invalid_call.get("name")
        else:
            err = getattr(invalid_call, "error", None)
            cid = getattr(invalid_call, "id", None)
            name = getattr(invalid_call, "name", None)

        content = f"Failed to parse tool call. Error: {err}"
        if cid and str(cid).strip():
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=str(cid).strip(), name=str(name) if name else None)
            )
        else:
            err_key = str(name) if name else f"invalid_tool_call_{idx}"
            tool_results[err_key] = content
            tool_results.setdefault("errors", []).append(content)
            logger.warning(
                "Skipping ToolMessage for invalid tool call without valid ID: %s (error: %s)",
                name,
                err,
            )

    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls:
        return tool_messages, tool_results

    for idx, tool_call in enumerate(tool_calls):
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

        has_valid_id = bool(tool_id and str(tool_id).strip())
        valid_id = str(tool_id).strip() if has_valid_id else None

        if not tool_name:
            content = "Malformed tool call: missing name"
            if valid_id:
                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=valid_id,
                        name="unknown",
                    )
                )
            else:
                err_key = f"malformed_tool_call_{idx}"
                tool_results[err_key] = content
                tool_results.setdefault("errors", []).append(content)
                logger.warning("Skipping ToolMessage for tool call without valid name and ID (index %d)", idx)
            continue

        if not valid_id:
            content = f"Malformed tool call for '{tool_name}': missing or invalid tool call ID"
            tool_results[tool_name] = content
            tool_results.setdefault("errors", []).append(content)
            logger.warning("Skipping ToolMessage for tool call '%s' lacking valid tool_call_id", tool_name)
            continue

        if tool_name in available_tools:
            try:
                result = available_tools[tool_name].invoke(tool_args)
                content = str(result) if result is not None else f"Tool {tool_name} executed successfully."
                tool_messages.append(ToolMessage(content=content, tool_call_id=valid_id, name=tool_name))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {e!r}"
                tool_messages.append(ToolMessage(content=error_message, tool_call_id=valid_id, name=tool_name))
                tool_results[tool_name] = error_message
        else:
            content = f"Tool {tool_name} not found."
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=valid_id,
                    name=tool_name,
                )
            )
            tool_results[tool_name] = content

    return tool_messages, tool_results


__all__ = [
    "execute_tool_calls",
    "validate_tool_history",
]
