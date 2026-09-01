"""
Specification tests for LargeLanguageModel token usage metrics tracking and tool execution wrapper.
Verifies global metrics increment correctly for target and meta systems and tool calls execute predictably.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from adas_core.llm_wrapper import LargeLanguageModel, execute_tool_calls


class TestLargeLanguageModelUsageMetricsSpecification:
    @pytest.fixture(autouse=True)
    def reset_usage_metrics(self):
        """Reset global LargeLanguageModel usage_metrics dictionaries before and after each test."""
        LargeLanguageModel.usage_metrics = {
            "meta_usage": {
                "overall": {
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            },
            "target_usage": {
                "overall": {
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            },
        }
        yield

    @patch("adas_core.llm_wrapper.get_model")
    def test_invoke_increments_target_usage_metrics(self, mock_get_model):
        """
        Contract: When is_meta=False, LargeLanguageModel.invoke must increment target_usage metrics
        both for the specific model name and for 'overall'.
        """
        mock_model_instance = MagicMock()
        mock_get_model.return_value = mock_model_instance

        llm = LargeLanguageModel(model_name="test-target-model", is_meta=False)
        mock_response = AIMessage(
            content="Target reply",
            id="msg_target_1",
            usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        )
        mock_model_instance.invoke.return_value = mock_response

        res = llm.invoke(["Hello target"])
        assert res.content == "Target reply"

        metrics = LargeLanguageModel.usage_metrics["target_usage"]
        assert metrics["overall"]["llm_calls"] == 1
        assert metrics["overall"]["input_tokens"] == 120
        assert metrics["overall"]["output_tokens"] == 40
        assert metrics["overall"]["total_tokens"] == 160

        model_metrics = metrics["test-target-model"]
        assert model_metrics["llm_calls"] == 1
        assert model_metrics["input_tokens"] == 120
        assert model_metrics["output_tokens"] == 40
        assert model_metrics["total_tokens"] == 160

    @patch("adas_core.llm_wrapper.get_model")
    def test_invoke_increments_meta_usage_metrics(self, mock_get_model):
        """
        Contract: When is_meta=True, LargeLanguageModel.invoke must increment meta_usage metrics
        without modifying target_usage metrics.
        """
        mock_model_instance = MagicMock()
        mock_get_model.return_value = mock_model_instance

        llm = LargeLanguageModel(model_name="test-meta-model", is_meta=True)
        mock_response = AIMessage(
            content="Meta reply",
            id="msg_meta_1",
            usage_metadata={"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
        )
        mock_model_instance.invoke.return_value = mock_response

        llm.invoke(["Hello meta"])

        meta_metrics = LargeLanguageModel.usage_metrics["meta_usage"]
        assert meta_metrics["overall"]["llm_calls"] == 1
        assert meta_metrics["overall"]["input_tokens"] == 300
        assert meta_metrics["overall"]["output_tokens"] == 150
        assert meta_metrics["overall"]["total_tokens"] == 450

        target_metrics = LargeLanguageModel.usage_metrics["target_usage"]
        assert target_metrics["overall"]["llm_calls"] == 0
        assert target_metrics["overall"]["total_tokens"] == 0

    @patch("adas_core.llm_wrapper.get_model")
    def test_invoke_with_count_metrics_false_does_not_increment(self, mock_get_model):
        """Contract: When count_metrics=False, invoke must not modify any usage metrics."""
        mock_model_instance = MagicMock()
        mock_get_model.return_value = mock_model_instance

        llm = LargeLanguageModel(model_name="test-model", is_meta=False)
        mock_response = AIMessage(
            content="No count reply",
            id="msg_2",
            usage_metadata={"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
        )
        mock_model_instance.invoke.return_value = mock_response

        llm.invoke(["Hello"], count_metrics=False)
        assert LargeLanguageModel.usage_metrics["target_usage"]["overall"]["llm_calls"] == 0
        assert LargeLanguageModel.usage_metrics["target_usage"]["overall"]["total_tokens"] == 0


class TestExecuteToolCallsSpecification:
    def test_execute_tool_calls_valid_and_missing_tool(self):
        """
        Contract: execute_tool_calls executes matching tools, returns ToolMessage list and result dict,
        and gracefully emits error message for missing tools.
        """

        @tool
        def add_nums(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        available_tools = {"add_nums": add_nums}
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "add_nums", "args": {"a": 5, "b": 7}, "id": "call_1"},
                {"name": "non_existent_tool", "args": {}, "id": "call_2"},
            ],
        )

        tool_msgs, results = execute_tool_calls(ai_message, available_tools)
        assert len(tool_msgs) == 2
        assert "12" in tool_msgs[0].content
        assert results["add_nums"] == 12
        assert "not found" in tool_msgs[1].content
