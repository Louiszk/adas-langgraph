import ast
from collections.abc import Sequence
from typing import Any

from adas_core.logging_config import get_logger

logger = get_logger("meta_system.helpers")


def normalize_response_content(content: Any) -> str:
    """Normalize string or list of content dicts from LLM response into a single string."""
    if isinstance(content, list):
        content_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                content_parts.append(str(item.get("text", "")))
            else:
                content_parts.append(str(item))
        return " ".join(content_parts)
    return str(content or "")


def ignored_nodes_message(ignored_nodes: Sequence[ast.AST]) -> str:
    """Creates a formatted 'Note:' string for any AST nodes that were ignored by a tool."""
    if not ignored_nodes:
        return ""

    messages = []
    for node in ignored_nodes:
        readable_format = f"A code structure of type '{type(node).__name__}'"

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            readable_format = f"{type(node).__name__} '{node.name}'"
        elif isinstance(node, ast.Assign):
            if node.targets and isinstance(node.targets[0], ast.Name):
                readable_format = f"Variable assignment for '{node.targets[0].id}'"
            else:
                readable_format = "A variable assignment"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                readable_format = f"Typed variable assignment for '{node.target.id}'"
            else:
                readable_format = "A typed variable assignment"
        messages.append(readable_format)

    note = "\nNote: The following structure(s) were ignored as they are not allowed in this block: "
    limit = 4
    if len(messages) > limit:
        ignored_list_str = ", ".join(messages[:limit])
        remaining_count = len(messages) - limit
        note += f"[{ignored_list_str}, ... (and {remaining_count} more)]."
    else:
        ignored_list_str = ", ".join(messages)
        note += f"[{ignored_list_str}]"
    return note
