import ast
import io
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from adas_core.environment import DEFAULT_EXCLUDED_PACKAGES


def escape_system_name(system_name: str) -> str:
    """Sanitize a system name by removing path and filesystem separator characters."""
    return system_name.replace("/", "").replace("\\", "").replace(":", "")


def sanitize_identifier(name: str, prefix_if_digit: str = "") -> str:
    """Convert an arbitrary string into a safe Python identifier component."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip())
    if cleaned and cleaned[0].isdigit() and prefix_if_digit:
        cleaned = f"{prefix_if_digit}{cleaned}"
    return cleaned or "default"


def sanitize_test_id(test_id: str) -> str:
    """Convert a test-case identifier into a Python identifier component."""
    return sanitize_identifier(test_id, prefix_if_digit="case_")


def validate_safe_relative_path(raw_path: str, field_name: str = "path") -> str:
    """Validate that raw_path is a safe relative path, rejecting empty, traversal, and absolute paths."""
    if not raw_path or not str(raw_path).strip():
        raise ValueError(f"Invalid {field_name}: path cannot be empty.")
    raw = str(raw_path).strip()
    if raw.startswith(("/", "\\")) or re.match(r"^[a-zA-Z]:", raw) or Path(raw).is_absolute():
        raise ValueError(f"Invalid {field_name} '{raw_path}': absolute paths are not allowed.")
    clean = raw.replace("\\", "/")
    parts = [p for p in clean.split("/") if p and p != "."]
    if not parts or ".." in parts:
        if ".." in parts:
            raise ValueError(f"Invalid {field_name} '{raw_path}': path traversal ('..') is not allowed.")
        raise ValueError(f"Invalid {field_name} '{raw_path}': empty or root normalized path is not allowed.")
    return raw_path


def get_filtered_packages(exclude_packages: list[str] | None = None) -> list[str]:
    if exclude_packages is None:
        exclude_packages = DEFAULT_EXCLUDED_PACKAGES

    result = subprocess.run(["pip", "list", "--not-required"], capture_output=True, text=True)

    packages = []
    for line in result.stdout.strip().split("\n")[2:]:  # Skip header lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                package_name = parts[0]
                version = parts[1]

                if package_name not in exclude_packages:
                    packages.append(f"{package_name} {version}")
    return packages


def validate_node_conditional_edge_signature(function_code: str) -> tuple[bool, str | None]:
    """
    Validates the signature of a node or conditional-edge function.
    It should accept exactly one argument named 'state'.
    """

    try:
        tree = ast.parse(function_code.strip())
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

    # Find the function definition node
    func_def_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def_node = node
            break

    if not func_def_node:
        return False, "No function definition found in the provided code."

    # Check the arguments
    args = func_def_node.args

    num_pos_args = len(args.args)
    has_vararg = args.vararg is not None
    has_kwarg = args.kwarg is not None
    num_kwonly_args = len(args.kwonlyargs)

    if num_pos_args == 1 and args.args[0].arg == "state" and not has_vararg and not has_kwarg and num_kwonly_args == 0:
        return True, None
    else:
        error_parts = []
        if num_pos_args != 1:
            error_parts.append(f"Expected 1 positional argument, but found {num_pos_args}.")
        elif args.args[0].arg != "state":
            error_parts.append(f"Expected the positional argument to be named 'state', but found '{args.args[0].arg}'.")

        if args.vararg is not None:
            error_parts.append(f"Unexpected *args (variable positional arguments) found: '{args.vararg.arg}'.")
        if args.kwarg is not None:
            error_parts.append(f"Unexpected **kwargs (variable keyword arguments) found: '{args.kwarg.arg}'.")
        if num_kwonly_args > 0:
            kwonly_names = [kw.arg for kw in args.kwonlyargs]
            error_parts.append(f"Unexpected keyword-only arguments found: {', '.join(kwonly_names)}.")

        return (
            False,
            f"Invalid signature for '{func_def_node.name}'. Nodes and conditional-edge functions must accept exactly one argument named 'state'. Issues: {' '.join(error_parts)}",
        )


class TruncatingStringIO(io.StringIO):
    """A custom StringIO that truncates each individual write operation by removing the middle."""

    def __init__(self, limit: int = 1200, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.limit = limit

    def write(self, s: str) -> int:
        """Overrides the default write method to truncate before writing it to the buffer."""
        truncated_message = "\n...[OUTPUT TRUNCATED]...\n"
        if len(s) > (self.limit + len(truncated_message)):
            start_chunk = s[: (self.limit // 2)]
            end_chunk = s[-(self.limit // 2) :]
            s = start_chunk + truncated_message + end_chunk
        return super().write(s)


def clean_messages(messages: list[Any]) -> list[Any]:
    allowed_attributes = {"type", "content", "tool_calls", "invalid_tool_calls"}
    allowed_tool_call_keys = {"name", "args"}

    for message in messages:
        for attr_name in list(vars(message).keys()):
            if attr_name not in allowed_attributes:
                try:
                    delattr(message, attr_name)
                except AttributeError:
                    pass

        if isinstance(message, AIMessage):
            for tool_call_list_name in ["tool_calls", "invalid_tool_calls"]:
                if hasattr(message, tool_call_list_name):
                    cleaned_calls = []
                    original_calls = getattr(message, tool_call_list_name)
                    if not original_calls:
                        continue

                    for call in original_calls:
                        if isinstance(call, dict):
                            cleaned_call = {k: v for k, v in call.items() if k in allowed_tool_call_keys}
                            cleaned_calls.append(cleaned_call)

                    setattr(message, tool_call_list_name, cleaned_calls)
    return messages


def truncate_state(state: dict[str, Any], max_chars: int = 1200) -> dict[str, Any] | None:
    if not state:
        return None

    truncated_state = {}
    truncated_message_template = "...[VALUE FOR '{}' (Type: {}) HAS BEEN TRUNCATED]..."
    msg_content_truncated_template = "...[MESSAGE CONTENT TRUNCATED]..."

    for key, value in state.items():
        if key == "messages" and isinstance(value, list):
            cleaned_msgs = clean_messages(value)

            # Truncate the content of each message
            for msg in cleaned_msgs:
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    if len(msg.content) > (max_chars + len(msg_content_truncated_template)):
                        start_chunk = msg.content[: (max_chars // 2)]
                        end_chunk = msg.content[-(max_chars // 2) :]
                        msg.content = start_chunk + msg_content_truncated_template + end_chunk

            truncated_state[key] = cleaned_msgs
        else:
            value_str = str(value)
            truncated_message = truncated_message_template.format(key, type(value).__name__)
            if len(value_str) > (max_chars + len(truncated_message)):
                start_chunk = value_str[: (max_chars // 2)]
                end_chunk = value_str[-(max_chars // 2) :]
                truncated_value_str = start_chunk + truncated_message + end_chunk
                truncated_state[key] = truncated_value_str
            else:
                truncated_state[key] = value

    return truncated_state


def remove_old_test_results(start_index, messages):
    test_report_pattern = re.compile(r"Test suite completed\..*?</ValidatorResult>", re.DOTALL)

    def create_summary(match_obj):
        report_text = match_obj.group(0)
        score_match = re.search(r"The system passed (\d+)/(\d+) tests", report_text)
        if score_match:
            passed, total = score_match.groups()
            return f"[Test executed. The system passed {passed}/{total} tests.]"
        else:
            return "[Previous test result condensed.]"

    for i in range(start_index, len(messages)):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            msg.content = test_report_pattern.sub(create_summary, str(msg.content))


def normalize_future_imports(code: str) -> str:
    """Ensure unique 'from __future__ import ...' statements appear at the beginning of the source using AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    future_nodes = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    if not future_nodes:
        return code

    lines = code.splitlines()
    remove_indices: set[int] = set()
    seen: set[str] = set()
    unique_future_lines: list[str] = []

    for node in future_nodes:
        seg = ast.get_source_segment(code, node)
        if seg:
            norm_seg = " ".join(seg.split())
            if norm_seg not in seen:
                seen.add(norm_seg)
                unique_future_lines.append(seg)
        if node.lineno is not None and node.end_lineno is not None:
            for idx in range(node.lineno - 1, node.end_lineno):
                remove_indices.add(idx)

    # Check if tree.body starts with a module docstring
    has_docstring = (
        len(tree.body) > 0
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )

    insert_idx = 0
    if has_docstring and tree.body[0].end_lineno is not None:
        insert_idx = tree.body[0].end_lineno
    else:
        # Keep leading shebang, encoding, and header comments at the very top
        first_non_future_lineno = None
        for node in tree.body:
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                first_non_future_lineno = node.lineno
                break

        limit = (first_non_future_lineno - 1) if first_non_future_lineno is not None else len(lines)
        for i in range(limit):
            stripped = lines[i].strip()
            if stripped.startswith("#") or not stripped:
                insert_idx = i + 1
            else:
                break

    prefix_lines = [lines[i] for i in range(insert_idx) if i not in remove_indices]
    suffix_lines = [lines[i] for i in range(insert_idx, len(lines)) if i not in remove_indices]

    while prefix_lines and not prefix_lines[-1].strip():
        prefix_lines.pop()

    while suffix_lines and not suffix_lines[0].strip():
        suffix_lines.pop(0)

    result_parts: list[str] = []
    if prefix_lines:
        result_parts.append("\n".join(prefix_lines))
    result_parts.append("\n".join(unique_future_lines))
    if suffix_lines:
        result_parts.append("\n".join(suffix_lines))

    result = "\n\n".join(result_parts)
    if code.endswith("\n"):
        result += "\n"
    return result
