"""
Specification and contract tests for core helper utilities.
"""

import pytest
from langchain_core.messages import AIMessage

from adas_core.helpers import (
    clean_messages,
    escape_system_name,
    normalize_fixture_path,
    normalize_future_imports,
    safe_write_text,
    truncate_state,
    validate_identifier,
    validate_node_conditional_edge_signature,
)


class TestNodeConditionalEdgeSignatureValidation:
    def test_valid_signature_with_state_param(self):
        """Contract: Functions accepting exactly 'state' positional argument are valid."""
        valid_code = "def my_node(state: dict) -> dict:\n    return state"
        is_valid, err = validate_node_conditional_edge_signature(valid_code)
        assert is_valid is True
        assert err is None

    def test_invalid_param_name(self):
        """Contract: Functions accepting a parameter not named 'state' must be rejected."""
        invalid_code = "def my_node(ctx: dict) -> dict:\n    return ctx"
        is_valid, err = validate_node_conditional_edge_signature(invalid_code)
        assert is_valid is False
        assert err is not None
        assert "Expected the positional argument to be named 'state'" in err

    def test_invalid_argument_counts_and_varargs(self):
        """Contract: Functions accepting extra positional args, *args, or **kwargs must be rejected."""
        codes = [
            "def my_node(state, extra):\n    pass",
            "def my_node(*args):\n    pass",
            "def my_node(state, **kwargs):\n    pass",
        ]
        for c in codes:
            is_valid, err = validate_node_conditional_edge_signature(c)
            assert is_valid is False
            assert err is not None


class TestMessageSanitization:
    def test_clean_messages_cleans_tool_calls(self):
        """
        Contract: clean_messages must preserve allowed attributes and clean tool_calls
        dict elements to only retain 'name' and 'args'.
        """
        msg = AIMessage(
            content="hello",
            tool_calls=[{"name": "test_tool", "args": {"x": 1}, "id": "call_123", "type": "tool_call"}],
        )
        cleaned = clean_messages([msg])
        assert len(cleaned[0].tool_calls) == 1
        call = cleaned[0].tool_calls[0]
        assert call == {"name": "test_tool", "args": {"x": 1}}
        assert "id" not in call
        assert "type" not in call

    def test_clean_messages_does_not_mutate_original(self):
        """clean_messages should not mutate the original message objects."""
        msg = AIMessage(
            content="hello",
            id="msg_id_123",
            tool_calls=[{"name": "test_tool", "args": {"x": 1}, "id": "call_123", "type": "tool_call"}],
        )
        cleaned = clean_messages([msg])
        assert cleaned[0] is not msg
        assert hasattr(msg, "id")
        assert msg.id == "msg_id_123"
        assert msg.tool_calls[0]["id"] == "call_123"


class TestStateTruncation:
    def test_truncate_state_shortens_large_strings_symmetrically(self):
        """Contract: Values exceeding max_chars must be truncated with middle placeholder."""
        huge_text = "A" * 2000
        state = {"large_field": huge_text, "small_field": "ok"}
        truncated = truncate_state(state, max_chars=100)
        assert truncated is not None
        assert "small_field" in truncated and truncated["small_field"] == "ok"
        assert "HAS BEEN TRUNCATED" in truncated["large_field"]
        assert len(truncated["large_field"]) < len(huge_text)

    def test_truncate_state_does_not_mutate_original_messages(self):
        """truncate_state should not mutate message objects in the input state."""
        msg = AIMessage(content="A" * 2000, id="msg_id_123")
        state = {"messages": [msg]}
        truncated = truncate_state(state, max_chars=100)
        assert truncated is not None
        assert len(msg.content) == 2000
        assert hasattr(msg, "id")
        assert len(truncated["messages"][0].content) < 2000


class TestNormalizeFutureImports:
    def test_deduplicates_repeated_future_imports(self):
        code = (
            "from __future__ import annotations\n"
            "from __future__ import annotations\n"
            "x = 1\n"
            "from __future__ import annotations\n"
        )
        normalized = normalize_future_imports(code)
        assert normalized.count("from __future__ import annotations") == 1
        assert normalized.startswith("from __future__ import annotations\n\nx = 1")

    def test_leaves_code_without_future_imports_unchanged(self):
        code = "x = 1\ny = 2\n"
        assert normalize_future_imports(code) == code

    def test_does_not_corrupt_multiline_string_containing_future_import(self):
        """Regression test: string literals containing 'from __future__' must not be extracted or corrupted."""
        code = 'def generate_template():\n    return """\nfrom __future__ import annotations\nimport sys\n"""\n'
        normalized = normalize_future_imports(code)
        assert normalized == code
        # Verify it doesn't hoist a future import to module level
        assert not normalized.startswith("from __future__")

    def test_hoists_future_import_without_corrupting_internal_strings(self):
        """Top-level future import is hoisted, but future import text inside a multiline string is untouched."""
        code = (
            'SETUP = ["pandas"]\n\n'
            "from __future__ import annotations\n\n"
            'template = """\n'
            "from __future__ import division\n"
            '"""\n'
        )
        normalized = normalize_future_imports(code)
        assert normalized.startswith("from __future__ import annotations")
        assert "from __future__ import division" in normalized
        assert normalized.count("from __future__ import annotations") == 1
        compile(normalized, "<test>", "exec")

    def test_preserves_module_docstring_and_shebang(self):
        """Module docstring and shebang stay before the hoisted future import."""
        code = (
            "#!/usr/bin/env python3\n"
            '"""Module docstring."""\n\n'
            'SETUP = ["pandas"]\n\n'
            "from __future__ import annotations\n"
        )
        normalized = normalize_future_imports(code)
        assert normalized.startswith('#!/usr/bin/env python3\n"""Module docstring."""')
        compile(normalized, "<test>", "exec")


class TestNormalizeFixturePath:
    def test_valid_fixture_paths(self):
        assert normalize_fixture_path("test.csv") == "test.csv"
        assert normalize_fixture_path("reports/") == "reports"
        assert normalize_fixture_path("nested/dir/file.json") == "nested/dir/file.json"
        assert normalize_fixture_path("nested\\dir\\file.json") == "nested/dir/file.json"
        assert normalize_fixture_path("./reports/q1.csv") == "reports/q1.csv"

    def test_strips_environment_prefixes(self):
        assert normalize_fixture_path("sandbox/workspace/data/input/orders.csv") == "orders.csv"
        assert normalize_fixture_path("sandbox/workspace/input/metrics.json") == "metrics.json"
        assert normalize_fixture_path("data/input/sales.csv") == "sales.csv"
        assert normalize_fixture_path("input/sub/report.pdf") == "sub/report.pdf"
        assert normalize_fixture_path("input\\sub\\report.pdf") == "sub/report.pdf"
        assert normalize_fixture_path("./data/input/sales.csv") == "sales.csv"

    def test_rejects_empty_and_whitespace(self):
        with pytest.raises(ValueError, match="path cannot be empty"):
            normalize_fixture_path("")
        with pytest.raises(ValueError, match="path cannot be empty"):
            normalize_fixture_path("   ")

    def test_rejects_absolute_paths(self):
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path("/etc/passwd")
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path("C:/Windows/System32")
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path(r"\\server\share\file.csv")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("../../outside.txt")
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("input/../../outside.txt")
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("sub/../..")

    def test_rejects_root_and_input_directories(self):
        with pytest.raises(ValueError, match="empty or root normalized path"):
            normalize_fixture_path(".")
        with pytest.raises(ValueError, match="empty or root normalized path"):
            normalize_fixture_path("./")
        with pytest.raises(ValueError, match="resolves to root input directory"):
            normalize_fixture_path("input")
        with pytest.raises(ValueError, match="resolves to root input directory"):
            normalize_fixture_path("input/")
        with pytest.raises(ValueError, match="resolves to root input directory"):
            normalize_fixture_path("sandbox/workspace/input/")


class TestValidateIdentifier:
    def test_valid_identifiers(self):
        assert validate_identifier("valid_name") == "valid_name"
        assert validate_identifier("case-1_test") == "case-1_test"
        assert validate_identifier("System123") == "System123"
        assert validate_identifier("a") == "a"

    def test_rejects_empty_or_whitespace(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_identifier("")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_identifier("   ")

    def test_rejects_special_and_traversal_characters(self):
        with pytest.raises(ValueError, match="must match pattern"):
            validate_identifier("../traversal")
        with pytest.raises(ValueError, match="must match pattern"):
            validate_identifier("name with spaces")
        with pytest.raises(ValueError, match="must match pattern"):
            validate_identifier("name$evil")
        with pytest.raises(ValueError, match="must match pattern"):
            validate_identifier("name;rm")
        with pytest.raises(ValueError, match="must match pattern"):
            validate_identifier("name/slash")


class TestSafeWriteText:
    def test_writes_within_root(self, tmp_path):
        root = tmp_path / "allowed_root"
        root.mkdir()
        target = root / "subdir" / "file.txt"
        safe_write_text(target, "hello content", root_dir=root)
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == "hello content"

    def test_rejects_path_traversal(self, tmp_path):
        root = tmp_path / "allowed_root"
        root.mkdir()
        target = root / ".." / "escaped.txt"
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_write_text(target, "evil content", root_dir=root)


class TestEscapeSystemName:
    def test_strips_path_separators_and_dangerous_chars(self):
        assert escape_system_name("Folder/Subfolder\\MySystem:v1") == "FolderSubfolderMySystemv1"
        assert escape_system_name("../../evil;name$") == "evilname"
        assert escape_system_name("simple_system") == "simple_system"

    def test_defaults_when_empty_after_cleaning(self):
        assert escape_system_name("///\\\\:::") == "default_system"
