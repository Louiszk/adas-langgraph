"""
Comprehensive specification tests for ChatModel, ModelRegistry, ModelCapabilities,
and UsageRecorder scoped telemetry.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.tools import tool

from adas_core.chat_model import (
    ChatModel,
    ModelRegistry,
    UsageRecorder,
    convert_to_messages,
    execute_tool_calls,
    usage_scope,
    validate_tool_history,
)


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Ensure clean state for every test."""
    UsageRecorder.reset()
    ModelRegistry.reset()
    ChatModel.allowed_target_models = None
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ADAS_TASK_SPEC_PATH", raising=False)
    yield
    UsageRecorder.reset()
    ModelRegistry.reset()
    ChatModel.allowed_target_models = None


# ============================================================================
# 1. Model Resolution & Allowlist Authorization Tests
# ============================================================================


class TestModelResolutionAndAuthorization:
    @patch("adas_core.chat_model.ChatOpenAI")
    def test_default_primary_model_from_task_spec(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "primary-model"},
            {"provider": "openai", "model_name": "secondary-model"},
        ]
        llm = ChatModel()
        assert llm.model == "primary-model"
        assert llm.provider == "openai"

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_explicit_allowed_model_without_provider(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "primary-model"},
            {"provider": "openai", "model_name": "secondary-model"},
        ]
        llm = ChatModel(model="secondary-model")
        assert llm.model == "secondary-model"
        assert llm.provider == "openai"

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_explicit_allowed_model_with_explicit_provider(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "primary-model"},
        ]
        llm = ChatModel(provider="openai", model="primary-model")
        assert llm.model == "primary-model"
        assert llm.provider == "openai"

    def test_unauthorized_target_model_raises_value_error(self):
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "allowed-model"},
        ]
        with pytest.raises(ValueError, match="Model 'forbidden-model' is not available"):
            ChatModel(model="forbidden-model")

    def test_task_spec_file_allowlist_enforcement(self, tmp_path, monkeypatch):
        task_json = tmp_path / "task.json"
        task_json.write_text(
            json.dumps(
                {
                    "name": "SpecAgent",
                    "system_goal": "Goal",
                    "architecture_contract": {"execution_mode": "single_turn", "state_schema": {"q": "str"}},
                    "available_models": [{"provider": "openai", "model_name": "permitted-model"}],
                    "dev_suite": [{"id": "c1", "description": "d", "turns": [{"q": "1"}]}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ADAS_TASK_SPEC_PATH", str(task_json))
        with pytest.raises(ValueError, match="Model 'unlisted-model' is not available"):
            ChatModel(model="unlisted-model")

    def test_ambiguous_model_across_multiple_providers_raises_value_error(self):
        ChatModel.allowed_target_models = [
            {"provider": "provider_a", "model_name": "shared-model"},
            {"provider": "provider_b", "model_name": "shared-model"},
        ]
        with pytest.raises(ValueError, match="ambiguous across multiple providers"):
            ChatModel(model="shared-model")

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_meta_scope_bypasses_target_allowlist(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "allowed-model"},
        ]
        with usage_scope(system="meta"):
            llm = ChatModel(model="arbitrary-meta-model")
            assert llm.model == "arbitrary-meta-model"

    def test_missing_api_key_raises_value_error(self, monkeypatch):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-4o"}]
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing environment variable: OPENAI_API_KEY"):
            ChatModel(model="gpt-4o")

    def test_unsupported_provider_raises_value_error(self):
        ChatModel.allowed_target_models = [
            {"provider": "unsupported_provider", "model_name": "m1"},
        ]
        with pytest.raises(ValueError, match="Unsupported provider: 'unsupported_provider'"):
            ChatModel(provider="unsupported_provider", model="m1")


# ============================================================================
# 2. Parameter Capabilities & Reasoning Effort Tests
# ============================================================================


class TestParameterCapabilities:
    @patch("adas_core.chat_model.ChatOpenAI")
    def test_standard_model_accepts_temperature(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-4o"}]
        llm = ChatModel(model="gpt-4o", temperature=0.7)
        assert llm.model == "gpt-4o"
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert "reasoning_effort" not in call_kwargs

    def test_standard_model_rejects_reasoning_effort(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-4o"}]
        with pytest.raises(ValueError, match="Model 'gpt-4o' \\(openai\\) does not support 'reasoning_effort'"):
            ChatModel(model="gpt-4o", reasoning_effort="low")

    def test_reasoning_model_rejects_temperature(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        with pytest.raises(ValueError, match="Model 'gpt-5.6-luna' \\(openai\\) does not support 'temperature'"):
            ChatModel(model="gpt-5.6-luna", temperature=0.7)

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_reasoning_model_forwards_reasoning_effort_none(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel(model="gpt-5.6-luna", reasoning_effort="none")
        assert llm.model == "gpt-5.6-luna"
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "none"
        assert "temperature" not in call_kwargs

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_reasoning_model_forwards_reasoning_effort_medium(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-terra"}]
        ChatModel(model="gpt-5.6-terra", reasoning_effort="medium")
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"

    def test_reasoning_model_rejects_invalid_effort_level(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "o1"}]
        with pytest.raises(ValueError, match="Invalid reasoning_effort 'unsupported_effort'"):
            ChatModel(model="o1", reasoning_effort="unsupported_effort")

    def test_older_o_series_rejects_reasoning_effort_none(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "o1"}]
        with pytest.raises(ValueError, match="Invalid reasoning_effort 'none'"):
            ChatModel(model="o1", reasoning_effort="none")

    def test_temperature_out_of_range_raises_value_error(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-4o"}]
        with pytest.raises(ValueError, match="Temperature 2.5 is out of range"):
            ChatModel(model="gpt-4o", temperature=2.5)


# ============================================================================
# 3. Composition, Immutability & Tool Binding Tests
# ============================================================================


class TestCompositionAndToolBinding:
    @patch("adas_core.chat_model.ChatOpenAI")
    def test_underlying_runnable_is_private(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        assert isinstance(llm.model, str)
        assert hasattr(llm, "_runnable")
        assert llm.model == "gpt-5.6-luna"

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_bind_tools_returns_new_instance_without_mutating_original(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_bound_runnable = MagicMock()
        mock_model.bind_tools.return_value = mock_bound_runnable
        mock_chat_openai.return_value = mock_model

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()

        @tool
        def calc_tool(expr: str) -> str:
            """Calculate expression."""
            return "42"

        bound_llm = llm.bind_tools([calc_tool], parallel_tool_calls=True)

        assert bound_llm is not llm
        assert bound_llm._runnable is mock_bound_runnable
        assert llm._runnable is mock_model
        assert bound_llm.model == llm.model
        assert bound_llm.provider == llm.provider
        mock_model.bind_tools.assert_called_once_with([calc_tool], parallel_tool_calls=True)

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_bind_tools_with_invalid_tools_raises_value_error(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        with pytest.raises(ValueError, match="All values in tools must be tool instances"):
            llm.bind_tools(["not_a_tool"])  # type: ignore

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_with_structured_output_returns_new_instance(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_model

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        schema = {"type": "object", "properties": {"ans": {"type": "string"}}}
        structured_llm = llm.with_structured_output(schema)

        assert structured_llm is not llm
        assert structured_llm._runnable is mock_structured
        assert llm._runnable is mock_model
        mock_model.with_structured_output.assert_called_once_with(schema, include_raw=True)

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_structured_output_preserves_raw_usage_while_returning_parsed_result(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": AIMessage(
                content='{"answer": "42"}',
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            ),
            "parsed": {"answer": "42"},
            "parsing_error": None,
        }
        mock_chat_openai.return_value = mock_model
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]

        result = ChatModel().with_structured_output({"type": "object"}).invoke("Answer the question")

        assert result == {"answer": "42"}
        record = UsageRecorder.get_records()[0]
        assert record.success is True
        assert record.usage_incomplete is False
        assert record.total_tokens == 16


# ============================================================================
# 4. Message Conversion & Tool Call Protocol Validation Tests
# ============================================================================


class TestMessageConversionAndProtocolValidation:
    def test_convert_to_messages_with_strings_and_dicts_preserving_tool_calls(self):
        inputs = [
            "User query string",
            {
                "role": "assistant",
                "content": "Tool call required",
                "tool_calls": [{"id": "c1", "name": "search", "args": {}}],
            },
            {"role": "tool", "content": "search result", "tool_call_id": "c1"},
        ]
        msgs = convert_to_messages(inputs)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert len(msgs[1].tool_calls) == 1
        assert msgs[1].tool_calls[0]["id"] == "c1"
        assert isinstance(msgs[2], ToolMessage)
        assert msgs[2].tool_call_id == "c1"

    def test_valid_tool_history_passes_validation(self):
        msgs = [
            HumanMessage(content="Calculate 2 + 2"),
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 2, "b": 2}}]),
            ToolMessage(content="4", tool_call_id="call_1"),
        ]
        validate_tool_history(msgs)  # Must not raise

    def test_unanswered_tool_call_raises_value_error(self):
        msgs = [
            HumanMessage(content="Run query"),
            AIMessage(content="", tool_calls=[{"id": "call_unanswered", "name": "q", "args": {}}]),
        ]
        with pytest.raises(ValueError, match="tool calls \\['call_unanswered'\\] have no corresponding ToolMessages"):
            validate_tool_history(msgs)

    def test_orphaned_tool_message_raises_value_error(self):
        msgs = [
            HumanMessage(content="Hello"),
            ToolMessage(content="4", tool_call_id="call_orphan"),
        ]
        with pytest.raises(ValueError, match="Orphaned ToolMessage .* has no matching prior AIMessage tool call"):
            validate_tool_history(msgs)

    def test_malformed_tool_message_without_id_raises_value_error(self):
        msgs = [
            HumanMessage(content="Hello"),
            ToolMessage(content="4", tool_call_id=""),
        ]
        with pytest.raises(ValueError, match="Malformed ToolMessage .* missing or empty 'tool_call_id'"):
            validate_tool_history(msgs)

    def test_duplicate_tool_message_raises_value_error(self):
        msgs = [
            HumanMessage(content="Run tool"),
            AIMessage(content="", tool_calls=[{"id": "call_dup", "name": "fn", "args": {}}]),
            ToolMessage(content="res1", tool_call_id="call_dup"),
            ToolMessage(content="res2", tool_call_id="call_dup"),
        ]
        with pytest.raises(ValueError, match="Duplicate ToolMessage .* for tool_call_id='call_dup'"):
            validate_tool_history(msgs)

    def test_interrupted_tool_calls_with_subsequent_human_message_raises_value_error(self):
        msgs = [
            AIMessage(content="", tool_calls=[{"id": "call_pending", "name": "fn", "args": {}}]),
            HumanMessage(content="Interrupting text"),
        ]
        with pytest.raises(
            ValueError,
            match="Message HumanMessage .* encountered while tool calls \\['call_pending'\\] remain unanswered",
        ):
            validate_tool_history(msgs)


# ============================================================================
# 5. Token Counting & trim_messages Integration Tests
# ============================================================================


class TestTokenCounting:
    def test_token_counter_and_get_num_tokens(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()

        messages = [
            SystemMessage(content="You are an expert helper."),
            HumanMessage(content="What is 1 + 1?"),
        ]
        assert llm.token_counter is llm._raw_model
        count_from_instance = llm.get_num_tokens_from_messages(messages)
        count_from_class = ChatModel.token_counter.get_num_tokens_from_messages(messages)

        assert count_from_instance > 0
        assert count_from_class > 0

    def test_token_counter_uses_model_specific_tokenizer(self):
        ChatModel.allowed_target_models = [
            {"provider": "openai", "model_name": "gpt-4o"},
            {"provider": "openai", "model_name": "gpt-3.5-turbo"},
        ]
        llm_4o = ChatModel(model="gpt-4o")
        llm_35 = ChatModel(model="gpt-3.5-turbo")
        assert llm_4o.token_counter.model_name == "gpt-4o"
        assert llm_35.token_counter.model_name == "gpt-3.5-turbo"

        # Japanese text encodes to 10 tokens with o200k_base (gpt-4o) vs 12 with cl100k_base (gpt-3.5)
        unicode_msgs = [HumanMessage(content="こんにちは世界！")]
        assert llm_4o.token_counter.get_num_tokens_from_messages(unicode_msgs) == 10
        assert llm_35.token_counter.get_num_tokens_from_messages(unicode_msgs) == 12

    def test_trim_messages_integration(self):
        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()

        messages = [
            SystemMessage(content="System " * 20),
            HumanMessage(content="Message 1 " * 20),
            AIMessage(content="Response 1 " * 20),
            HumanMessage(content="Message 2 " * 20),
            AIMessage(content="Response 2"),
        ]

        trimmed = trim_messages(
            messages,
            max_tokens=40,
            strategy="last",
            token_counter=llm.get_num_tokens_from_messages,
        )
        assert len(trimmed) < len(messages)
        assert trimmed[-1].content == "Response 2"

        trimmed_via_instance = trim_messages(
            messages,
            max_tokens=40,
            strategy="last",
            token_counter=llm.token_counter,
        )
        assert len(trimmed_via_instance) < len(messages)
        assert trimmed_via_instance[-1].content == "Response 2"

        trimmed_via_class = trim_messages(
            messages,
            max_tokens=40,
            strategy="last",
            token_counter=ChatModel.token_counter,
        )
        assert len(trimmed_via_class) < len(messages)
        assert trimmed_via_class[-1].content == "Response 2"


# ============================================================================
# 6. Scoped Telemetry (UsageRecorder, Scopes & Streams) Tests
# ============================================================================


class TestScopedUsageRecorder:
    @patch("adas_core.chat_model.ChatOpenAI")
    def test_target_and_meta_scopes_isolate_records(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        mock_target_resp = AIMessage(
            content="Target reply",
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )
        mock_meta_resp = AIMessage(
            content="Meta reply",
            usage_metadata={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
        )

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]

        # Target call
        mock_model.invoke.return_value = mock_target_resp
        with usage_scope(system="target", run_id="run_101", case_id="case_1"):
            llm_target = ChatModel()
            llm_target.invoke("Hello target")

        # Meta call
        mock_model.invoke.return_value = mock_meta_resp
        with usage_scope(system="meta", run_id="meta_run_1"):
            llm_meta = ChatModel(model="gpt-5.6-luna")
            llm_meta.invoke("Hello meta")

        target_records = UsageRecorder.get_records(system="target")
        meta_records = UsageRecorder.get_records(system="meta")

        assert len(target_records) == 1
        assert target_records[0].run_id == "run_101"
        assert target_records[0].case_id == "case_1"
        assert target_records[0].input_tokens == 50
        assert target_records[0].output_tokens == 20
        assert target_records[0].total_tokens == 70

        assert len(meta_records) == 1
        assert meta_records[0].run_id == "meta_run_1"
        assert meta_records[0].total_tokens == 300

        target_agg = UsageRecorder.get_aggregate(system="target", run_id="run_101")
        assert target_agg["llm_calls"] == 1
        assert target_agg["total_tokens"] == 70

        # Verify legacy metrics mirror
        assert ChatModel.usage_metrics["target_usage"]["overall"]["total_tokens"] == 70
        assert ChatModel.usage_metrics["meta_usage"]["overall"]["total_tokens"] == 300

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_missing_usage_metadata_does_not_crash(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="No tokens")

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        llm.invoke("Hello")

        records = UsageRecorder.get_records()
        assert len(records) == 1
        assert records[0].usage_incomplete is True
        assert records[0].total_tokens == 0

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_stream_usage_recording_completed_and_interrupted(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        # Completed stream
        chunks = [
            AIMessageChunk(content="chunk 1"),
            AIMessageChunk(
                content="chunk 2", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            ),
        ]
        mock_model.stream.return_value = iter(chunks)

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        result_chunks = list(llm.stream("Stream test"))
        assert len(result_chunks) == 2

        records = UsageRecorder.get_records()
        assert len(records) == 1
        assert records[0].success is True
        assert records[0].total_tokens == 15
        assert records[0].usage_incomplete is False

        # Interrupted stream
        def failing_stream(*args, **kwargs):
            yield AIMessageChunk(content="part 1")
            raise RuntimeError("Network cut")

        mock_model.stream.side_effect = failing_stream
        with pytest.raises(RuntimeError, match="Network cut"):
            list(llm.stream("Fail stream"))

        records = UsageRecorder.get_records()
        assert len(records) == 2
        assert records[1].success is False
        assert records[1].usage_incomplete is True
        assert records[1].error == "Network cut"

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_nested_usage_scope_inherits_and_restores(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(
            content="ok", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        )

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()

        with usage_scope(system="target", run_id="run_A", case_id="case_A"):
            llm.invoke("call in outer")
            with usage_scope(node="node_inner"):
                llm.invoke("call in inner")
            llm.invoke("call after inner")

        records = UsageRecorder.get_records()
        assert len(records) == 3
        assert records[0].run_id == "run_A"
        assert records[0].node is None
        assert records[1].run_id == "run_A"
        assert records[1].case_id == "case_A"
        assert records[1].node == "node_inner"
        assert records[2].run_id == "run_A"
        assert records[2].node is None

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_batch_methods_are_explicitly_unsupported(self, mock_chat_openai):
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        ChatModel.allowed_target_models = [{"provider": "openai", "model_name": "gpt-5.6-luna"}]
        llm = ChatModel()
        with pytest.raises(NotImplementedError, match=r"ChatModel.batch\(\) is not supported"):
            llm.batch(["q1", "q2"])
        with pytest.raises(NotImplementedError, match=r"ChatModel.abatch\(\) is not supported"):
            asyncio.run(llm.abatch(["q1", "q2"]))
        mock_model.batch.assert_not_called()
        mock_model.abatch.assert_not_called()


# ============================================================================
# 7. Execute Tool Calls Helper Tests
# ============================================================================


class TestExecuteToolCalls:
    def test_execute_tool_calls_success_and_missing(self):
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two ints."""
            return a * b

        tools_map = {"multiply": multiply}
        response = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "multiply", "args": {"a": 3, "b": 4}},
                {"id": "c2", "name": "unknown_tool", "args": {}},
            ],
        )
        tool_msgs, results = execute_tool_calls(response, tools_map)
        assert len(tool_msgs) == 2
        assert "12" in tool_msgs[0].content
        assert results["multiply"] == 12
        assert "not found" in tool_msgs[1].content

    def test_tool_calls_reexported_from_chat_model(self):
        from adas_core.chat_model import execute_tool_calls as chat_etc
        from adas_core.chat_model import validate_tool_history as chat_vth
        from adas_core.tool_calls import execute_tool_calls, validate_tool_history

        assert chat_etc is execute_tool_calls
        assert chat_vth is validate_tool_history


# ============================================================================
# 8. Vision Modality & Capabilities Tests
# ============================================================================


class TestVisionCapabilities:
    def test_model_registry_vision_capabilities_and_prefix_ordering(self):
        # Explicit models
        assert ModelRegistry.get_capabilities("openai", "gpt-4o").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-4o-mini").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-4-turbo").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-4").supports_vision is False
        assert ModelRegistry.get_capabilities("openai", "gpt-3.5-turbo").supports_vision is False

        # Reasoning models
        assert ModelRegistry.get_capabilities("openai", "o1").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "o1-mini").supports_vision is False
        assert ModelRegistry.get_capabilities("openai", "o1-preview").supports_vision is False
        assert ModelRegistry.get_capabilities("openai", "o3").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "o3-mini").supports_vision is False

        # GPT-5.6 family
        assert ModelRegistry.get_capabilities("openai", "gpt-5.6-sol").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-5.6-terra").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-5.6-luna").supports_vision is True

        # Prefix sorting tests: ensure longer prefixes match before shorter prefixes
        # 1. o1-mini prefix must match o1-mini (False), NOT o1 (True)
        assert ModelRegistry.get_capabilities("openai", "o1-mini-2024-09-12").supports_vision is False
        assert ModelRegistry.get_capabilities("openai", "o1-2024-12-17").supports_vision is True

        # 2. gpt-4o prefix must match gpt-4o (True), NOT gpt-4 (False)
        assert ModelRegistry.get_capabilities("openai", "gpt-4o-2024-08-06").supports_vision is True
        assert ModelRegistry.get_capabilities("openai", "gpt-4-0613").supports_vision is False

    def test_has_image_content_detection(self):
        from adas_core.chat_model import has_image_content

        msg_text = [HumanMessage(content="Simple text prompt")]
        assert has_image_content(msg_text) is False

        msg_blocks_text = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ]
            )
        ]
        assert has_image_content(msg_blocks_text) is False

        msg_with_img = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Inspect image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ]
            )
        ]
        assert has_image_content(msg_with_img) is True

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_chat_model_rejects_images_for_non_vision_model(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        llm = ChatModel(model="o3-mini", provider="openai", is_meta=True)

        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this chart:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}},
            ]
        )

        with pytest.raises(ValueError, match="does not support vision/image inputs"):
            llm.invoke([image_message])

    @patch("adas_core.chat_model.ChatOpenAI")
    def test_chat_model_allows_images_for_vision_model(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(content="I see the chart")
        mock_chat_openai.return_value = mock_instance

        llm = ChatModel(model="gpt-4o", provider="openai", is_meta=True)
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this chart:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
            ]
        )

        response = llm.invoke([image_message])
        assert response.content == "I see the chart"
        mock_instance.invoke.assert_called_once()
