"""
Specification and contract tests for core helper utilities.
"""

from langchain_core.messages import AIMessage
from adas_core.helpers import (
    validate_node_router_signature,
    clean_messages,
    truncate_state,
)


class TestNodeRouterSignatureValidation:
    def test_valid_signature_with_state_param(self):
        """Contract: Functions accepting exactly 'state' positional argument are valid."""
        valid_code = "def my_node(state: dict) -> dict:\n    return state"
        is_valid, err = validate_node_router_signature(valid_code)
        assert is_valid is True
        assert err is None

    def test_invalid_param_name(self):
        """Contract: Functions accepting a parameter not named 'state' must be rejected."""
        invalid_code = "def my_node(ctx: dict) -> dict:\n    return ctx"
        is_valid, err = validate_node_router_signature(invalid_code)
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
            is_valid, err = validate_node_router_signature(c)
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
