"""
LLMJudge SDK for open-ended semantic and criteria-based evaluation of target agent systems.

Provides:
- JudgeEvaluation: Default structured Pydantic schema for judge verdicts and reasoning.
- LLMJudge: Impartial evaluator executing under meta usage scope with arbitrary structured schema support.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from adas_core.chat_model import ChatModel, ModelRegistry, usage_scope
from adas_core.logging_config import get_logger
from meta_system.config import validation_model, validation_wrapper

logger = get_logger("judge")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per OpenAI specification

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are an impartial, rigorous, and objective evaluation judge assessing an AI agent system.
Analyze the provided materials carefully against the evaluation instructions and provide a structured assessment."""

JUDGE_SYSTEM_PROMPT = DEFAULT_JUDGE_SYSTEM_PROMPT

T = TypeVar("T", bound=BaseModel)


def format_image_payload(image_input: str | Path) -> dict[str, Any]:
    """Format a local file path, remote URL, or base64 data URL for ChatModel vision input."""
    img_str = str(image_input).strip()
    if not img_str:
        raise ValueError("Image input path or URL cannot be empty.")

    # Remote web URL or data URL
    if img_str.startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": img_str}}

    # Local file path
    file_path = Path(img_str)
    if not file_path.is_file():
        raise FileNotFoundError(f"Judge image file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{suffix}' for file '{file_path}'. "
            f"Supported formats are: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    file_size = file_path.stat().st_size
    if file_size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"Image file '{file_path}' exceeds the 20 MB size limit ({file_size} bytes).")

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_map.get(suffix, "image/png")

    file_bytes = file_path.read_bytes()
    b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_encoded}"
    return {"type": "image_url", "image_url": {"url": data_url}}


class JudgeEvaluation(BaseModel):
    """Structured evaluation output produced by LLMJudge."""

    is_pass: bool = Field(
        ...,
        description="True if the execution meets all required evaluation criteria; False otherwise.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed step-by-step rationale for the verdict, citing specific evidence from the context.",
    )
    score: float | None = Field(
        default=None,
        description="Optional numerical quality score between 0.0 and 1.0.",
    )
    checklist_results: dict[str, bool] | None = Field(
        default=None,
        description="Mapping of each checklist item description to its pass (true) or fail (false) verdict.",
    )


class LLMJudge:
    """
    Impartial semantic evaluator for target agent outputs and state artifacts.

    Evaluations run strictly inside a meta usage scope (`system="meta", node="judge"`) to ensure
    that judge tokens and latency never contaminate target system metrics.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
        self.model = model or validation_model
        self.provider = provider or validation_wrapper
        self.temperature = temperature
        self.max_retries = max_retries

    def evaluate(
        self,
        prompt: str,
        *,
        schema: type[T | JudgeEvaluation] = JudgeEvaluation,
        system_prompt: str | None = None,
        images: list[str | Path] | str | Path | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> Any:
        """
        Evaluate target agent execution against instructions and criteria.

        Parameters:
            prompt: Direct, detailed evaluation prompt containing task context, criteria, and outputs to evaluate.
            schema: Optional Pydantic model for custom structured return data. Defaults to JudgeEvaluation.
            system_prompt: Optional override for the judge's system prompt.
            images: Optional local path(s) or remote URL(s) for multimodal visual evaluation.
            model: Optional model override for this evaluation call.
            provider: Optional provider override for this evaluation call.

        Returns:
            An instance of schema (defaults to JudgeEvaluation).
        """
        target_schema: type[BaseModel] = schema if schema is not None else JudgeEvaluation
        effective_system_prompt = system_prompt or self.system_prompt
        effective_model = model or self.model
        effective_provider = provider or self.provider

        formatted_images: list[dict[str, Any]] = []
        if images:
            img_list = [images] if isinstance(images, (str, Path)) else list(images)
            for img in img_list:
                formatted_images.append(format_image_payload(img))

        if formatted_images:
            capabilities = ModelRegistry.get_capabilities(effective_provider, effective_model)
            if not capabilities.supports_vision:
                raise ValueError(
                    f"Model '{effective_model}' ({effective_provider}) does not support vision/image evaluation. "
                    f"Please configure a vision-capable judge model (e.g. 'gpt-4o', 'gpt-4o-mini', 'gpt-5.6-luna', 'gpt-5.6-sol')."
                )
            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    *formatted_images,
                ]
            )
        else:
            human_message = HumanMessage(content=prompt)

        messages = [
            SystemMessage(content=effective_system_prompt),
            human_message,
        ]

        # Enforce meta scope with node='judge' so judge tokens are never attributed to target system
        with usage_scope(system="meta", node="judge"):
            llm = ChatModel(
                model=effective_model,
                provider=effective_provider,
                temperature=self.temperature,
                is_meta=True,
            ).with_structured_output(target_schema)

            last_error: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    result = llm.invoke(messages)
                    if isinstance(result, target_schema):
                        return result
                    if isinstance(result, dict):
                        return target_schema.model_validate(result)
                    raise ValueError(f"Unexpected response type from structured judge: {type(result).__name__}")
                except Exception as e:
                    last_error = e
                    logger.warning(f"LLMJudge structured invocation attempt {attempt} failed: {e!r}")

            raise RuntimeError(f"LLMJudge failed after {self.max_retries} attempts: {last_error!r}") from last_error


__all__ = [
    "DEFAULT_JUDGE_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "MAX_IMAGE_SIZE_BYTES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "JudgeEvaluation",
    "LLMJudge",
    "format_image_payload",
]
