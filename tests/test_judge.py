from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from adas_core.chat_model import ModelCapabilities, ModelRegistry, get_current_scope, usage_scope
from adas_core.judge import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    JudgeEvaluation,
    LLMJudge,
    format_image_payload,
)


class TestJudgeEvaluationSchema:
    def test_valid_evaluation(self):
        eval_obj = JudgeEvaluation(
            is_pass=True,
            reasoning="The agent correctly queried the database and returned the requested rows.",
            score=0.95,
            checklist_results={"accuracy": True, "completeness": True},
        )
        assert eval_obj.is_pass is True
        assert "correctly queried" in eval_obj.reasoning
        assert eval_obj.score == 0.95
        assert eval_obj.checklist_results == {"accuracy": True, "completeness": True}

    def test_minimal_evaluation(self):
        eval_obj = JudgeEvaluation(
            is_pass=False,
            reasoning="Output was missing expected summary.",
        )
        assert eval_obj.is_pass is False
        assert eval_obj.score is None
        assert eval_obj.checklist_results is None

    def test_validation_error_on_missing_fields(self):
        with pytest.raises(ValidationError):
            JudgeEvaluation.model_validate({"is_pass": True})  # missing reasoning


class TestFormatImagePayload:
    def test_format_image_payload_local_file(self, tmp_path):
        dummy_img = tmp_path / "plot.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        payload = format_image_payload(dummy_img)
        assert payload["type"] == "image_url"
        assert payload["image_url"]["url"].startswith("data:image/png;base64,")

    def test_format_image_payload_remote_and_data_url(self):
        url = "https://example.com/chart.jpg"
        payload = format_image_payload(url)
        assert payload == {"type": "image_url", "image_url": {"url": url}}

        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        payload2 = format_image_payload(data_url)
        assert payload2 == {"type": "image_url", "image_url": {"url": data_url}}

    def test_format_image_payload_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError, match="Judge image file not found"):
            format_image_payload(missing)

    def test_format_image_payload_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            format_image_payload("   ")

    def test_format_image_payload_unsupported_format(self, tmp_path):
        svg_file = tmp_path / "diagram.svg"
        svg_file.write_text("<svg></svg>", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported image format '.svg'"):
            format_image_payload(svg_file)

    def test_format_image_payload_exceeds_size_limit(self, tmp_path, monkeypatch):
        dummy_img = tmp_path / "large.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")
        # Monkeypatch st_size to simulate > 20 MB
        stat_result = dummy_img.stat()
        monkeypatch.setattr(
            Path,
            "stat",
            lambda self: (
                type(stat_result)(
                    (
                        stat_result.st_mode,
                        stat_result.st_ino,
                        stat_result.st_dev,
                        stat_result.st_nlink,
                        stat_result.st_uid,
                        stat_result.st_gid,
                        25 * 1024 * 1024,
                        stat_result.st_atime,
                        stat_result.st_mtime,
                        stat_result.st_ctime,
                    )
                )
                if self == dummy_img
                else stat_result
            ),
        )
        with pytest.raises(ValueError, match="exceeds the 20 MB size limit"):
            format_image_payload(dummy_img)


class TestLLMJudge:
    def test_initialization_defaults(self):
        judge = LLMJudge()
        assert judge.system_prompt == DEFAULT_JUDGE_SYSTEM_PROMPT
        assert judge.model is not None
        assert judge.provider is not None
        assert judge.max_retries == 2

        custom_judge = LLMJudge(system_prompt="Custom system instructions", max_retries=3)
        assert custom_judge.system_prompt == "Custom system instructions"
        assert custom_judge.max_retries == 3

    @patch("adas_core.judge.ChatModel")
    def test_evaluate_default_schema_success(self, mock_chat_model_cls):
        expected_eval = JudgeEvaluation(
            is_pass=True,
            reasoning="The agent output adheres to all specified constraints.",
            score=1.0,
            checklist_results={"valid_format": True},
        )

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = expected_eval

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge()
        prompt = "Task: Format telemetry\nCriteria: Valid JSON\nOutput: {'status': 'ok'}"
        result = judge.evaluate(prompt=prompt)

        assert isinstance(result, JudgeEvaluation)
        assert result.is_pass is True
        assert result.score == 1.0
        assert result.checklist_results == {"valid_format": True}

        # Verify structured prompt configuration
        mock_chat_instance.with_structured_output.assert_called_once_with(JudgeEvaluation)
        invoke_call_args = mock_structured_model.invoke.call_args[0][0]
        sys_msg = invoke_call_args[0]
        human_msg = invoke_call_args[1]
        assert sys_msg.content == DEFAULT_JUDGE_SYSTEM_PROMPT
        assert human_msg.content == prompt

    @patch("adas_core.judge.ChatModel")
    def test_evaluate_custom_pydantic_schema(self, mock_chat_model_cls):
        class ExtractedReport(BaseModel):
            has_error: bool
            issue_count: int
            categories: list[str]

        expected_report = ExtractedReport(
            has_error=False,
            issue_count=0,
            categories=["performance", "security"],
        )

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = expected_report

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge()
        custom_prompt = "Extract report metrics from the logs: [LOG] all clear."
        result = judge.evaluate(prompt=custom_prompt, schema=ExtractedReport)

        assert isinstance(result, ExtractedReport)
        assert result.has_error is False
        assert result.issue_count == 0
        assert result.categories == ["performance", "security"]
        mock_chat_instance.with_structured_output.assert_called_once_with(ExtractedReport)

    @patch("adas_core.judge.ChatModel")
    def test_evaluate_custom_system_prompt_override(self, mock_chat_model_cls):
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = JudgeEvaluation(is_pass=True, reasoning="Passed")

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge(system_prompt="Initial system prompt")
        judge.evaluate(prompt="Check", system_prompt="Overridden system prompt")

        invoke_call_args = mock_structured_model.invoke.call_args[0][0]
        sys_msg = invoke_call_args[0]
        assert sys_msg.content == "Overridden system prompt"

    @patch("adas_core.judge.ChatModel")
    def test_retries_on_failure(self, mock_chat_model_cls):
        expected_eval = JudgeEvaluation(
            is_pass=True,
            reasoning="Recovered on attempt 2.",
        )

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = [
            ValueError("Malformed JSON response from provider"),
            expected_eval,
        ]

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge(max_retries=2)
        result = judge.evaluate(prompt="Task description and context")

        assert result.is_pass is True
        assert mock_structured_model.invoke.call_count == 2

    @patch("adas_core.judge.ChatModel")
    def test_evaluate_raises_when_all_retries_exhausted(self, mock_chat_model_cls):
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = RuntimeError("Provider timeout")

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge(max_retries=2)
        with pytest.raises(RuntimeError, match="LLMJudge failed after 2 attempts"):
            judge.evaluate(prompt="Task description")

    @patch("adas_core.judge.ChatModel")
    def test_telemetry_meta_scope_isolation(self, mock_chat_model_cls):
        """Ensure LLMJudge invocations execute strictly under meta scope with node='judge'."""
        captured_scopes: list[tuple[str, str]] = []

        def fake_invoke(messages: Any) -> JudgeEvaluation:
            scope = get_current_scope()
            captured_scopes.append((scope.system, scope.node or ""))
            return JudgeEvaluation(is_pass=True, reasoning="Pass")

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = fake_invoke

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge()

        # Execute judge while ostensibly in target scope
        with usage_scope(system="target", node="agent_node"):
            result = judge.evaluate(prompt="Check output")

        assert result.is_pass is True
        assert len(captured_scopes) == 1
        # The judge call must have overridden the active scope to meta / judge!
        assert captured_scopes[0] == ("meta", "judge")

    @patch("adas_core.judge.ChatModel")
    def test_multimodal_image_evaluation(self, mock_chat_model_cls, tmp_path):
        """Ensure local and remote images are properly embedded into HumanMessage content blocks."""
        test_img = tmp_path / "diagram.png"
        test_img.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes")

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = JudgeEvaluation(
            is_pass=True,
            reasoning="Diagram correctly shows flow.",
        )

        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge()
        prompt = "Task: Generate pipeline diagram\nCriteria: Ensure diagram contains 3 nodes"
        result = judge.evaluate(
            prompt=prompt,
            images=[test_img, "https://example.com/reference.png"],
        )

        assert result.is_pass is True
        assert "Diagram correctly shows flow" in result.reasoning

        # Inspect the messages passed to invoke
        invoke_call_args = mock_structured_model.invoke.call_args[0][0]
        human_msg = invoke_call_args[1]
        assert isinstance(human_msg.content, list)
        assert len(human_msg.content) == 3
        # First block: text prompt
        assert human_msg.content[0]["type"] == "text"
        assert "Generate pipeline diagram" in human_msg.content[0]["text"]
        # Second block: local image encoded as base64 data URL
        assert human_msg.content[1]["type"] == "image_url"
        assert human_msg.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        # Third block: remote image URL
        assert human_msg.content[2]["type"] == "image_url"
        assert human_msg.content[2]["image_url"]["url"] == "https://example.com/reference.png"

    @patch("adas_core.judge.ChatModel")
    def test_missing_image_raises_file_not_found(self, mock_chat_model_cls, tmp_path):
        missing_file = tmp_path / "does_not_exist.png"
        judge = LLMJudge()

        with pytest.raises(FileNotFoundError, match="Judge image file not found"):
            judge.evaluate(prompt="Task", images=[missing_file])

    @patch("adas_core.judge.ChatModel")
    def test_heterogeneous_model_override(self, mock_chat_model_cls):
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = JudgeEvaluation(is_pass=True, reasoning="Pass")
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        # 1. Custom model at init time
        judge_fast = LLMJudge(model="gpt-4o-mini")
        judge_fast.evaluate(prompt="Simple check")
        mock_chat_model_cls.assert_called_with(
            model="gpt-4o-mini",
            provider=judge_fast.provider,
            temperature=None,
            is_meta=True,
            default_tools=None,
        )

        # 2. Custom model override at evaluate() time
        judge_default = LLMJudge()
        judge_default.evaluate(prompt="Deep reasoning check", model="o3")
        mock_chat_model_cls.assert_called_with(
            model="o3",
            provider=judge_default.provider,
            temperature=None,
            is_meta=True,
            default_tools=None,
        )

    def test_evaluate_fails_fast_when_model_does_not_support_vision(self, tmp_path):
        """Passing images to a non-vision model fails fast."""
        dummy_img = tmp_path / "chart.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        ModelRegistry.register_capabilities("openai", "text-only-eval", ModelCapabilities(supports_vision=False))
        judge = LLMJudge()

        with pytest.raises(ValueError, match="does not support vision/image evaluation"):
            judge.evaluate(
                prompt="Verify chart",
                images=[dummy_img],
                model="text-only-eval",
            )

    @patch("adas_core.judge.ChatModel")
    def test_vision_evaluation_with_gpt_5_6_sol(self, mock_chat_model_cls, tmp_path):
        """Flagship gpt-5.6-sol supports vision evaluation."""
        dummy_img = tmp_path / "sol_chart.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = JudgeEvaluation(is_pass=True, reasoning="Sol passed")
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_structured_model
        mock_chat_model_cls.return_value = mock_chat_instance

        judge = LLMJudge()
        result = judge.evaluate(
            prompt="Sol visual inspection",
            images=[dummy_img],
            model="gpt-5.6-sol",
        )
        assert result.is_pass is True
        assert result.reasoning == "Sol passed"
        mock_chat_model_cls.assert_called_with(
            model="gpt-5.6-sol",
            provider=judge.provider,
            temperature=None,
            is_meta=True,
            default_tools=None,
        )
