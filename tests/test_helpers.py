"""
Specification and contract tests for core helper utilities.
"""

from langchain_core.messages import AIMessage

from adas_core.helpers import (
    clean_messages,
    normalize_future_imports,
    truncate_state,
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
