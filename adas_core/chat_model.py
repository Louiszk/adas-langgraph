"""
ChatModel runtime and telemetry architecture for ADAS.

Provides:
- ModelRegistry and ModelCapabilities for parameter validation and model resolution.
- UsageRecorder with scoped context tracking (system, run_id, case_id, node).
- ChatModel: composition-based LangChain chat delegator with strict tool protocol
  validation, immutable bind_tools, structured output, and token counting.
- execute_tool_calls runtime helper.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from adas_core.environment import SANDBOX_TASK_SPEC_PATH
from adas_core.exceptions import ModelConfigurationError, StructuredOutputError
from adas_core.tool_calls import execute_tool_calls, validate_tool_history
from config.logging import get_logger

logger = get_logger("chat_model")


# ============================================================================
# 1. Model Capabilities & Registry
# ============================================================================


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared parameter capabilities for a model."""

    supports_temperature: bool = True
    temperature_range: tuple[float, float] = (0.0, 2.0)
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: frozenset[str] = frozenset()
    supports_structured_output: bool = True
    supports_vision: bool = True
    supports_web_search: bool = False


_STANDARD_CHAT_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    temperature_range=(0.0, 2.0),
    supports_reasoning_effort=False,
    supported_reasoning_efforts=frozenset(),
    supports_structured_output=True,
    supports_vision=True,
    supports_web_search=True,
)

_OPENAI_REASONING_VISION = ModelCapabilities(
    supports_temperature=False,
    supports_reasoning_effort=True,
    supported_reasoning_efforts=frozenset({"low", "medium", "high"}),
    supports_structured_output=True,
    supports_vision=True,
    supports_web_search=True,
)

_OPENAI_REASONING_FULL = ModelCapabilities(
    supports_temperature=False,
    supports_reasoning_effort=True,
    supported_reasoning_efforts=frozenset({"none", "low", "medium", "high", "xhigh"}),
    supports_structured_output=True,
    supports_vision=True,
    supports_web_search=True,
)


class ModelRegistry:
    """Internal registry of model capabilities and known provider specifications."""

    _lock = threading.Lock()
    _capabilities: dict[tuple[str, str], ModelCapabilities] = {}

    @staticmethod
    def _matches_registered_name(configured_model: str, model_name: str) -> bool:
        """Accept a catalog model or one of its OpenAI date-versioned snapshots."""
        if model_name == configured_model:
            return True
        snapshot_suffix = model_name.removeprefix(f"{configured_model}-")
        return snapshot_suffix != model_name and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_suffix))

    @classmethod
    def _init_defaults(cls) -> None:
        defaults: dict[tuple[str, str], ModelCapabilities] = {
            # OpenAI standard models
            ("openai", "gpt-4o"): _STANDARD_CHAT_CAPABILITIES,
            ("openai", "gpt-4o-mini"): _STANDARD_CHAT_CAPABILITIES,
            # OpenAI GPT-5.4 family
            ("openai", "gpt-5.4"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.4-mini"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.4-nano"): _OPENAI_REASONING_FULL,
            # OpenAI reasoning models
            ("openai", "o3"): _OPENAI_REASONING_VISION,
            # OpenAI GPT-5.6 family
            ("openai", "gpt-5.6-sol"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.6-terra"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.6-luna"): _OPENAI_REASONING_FULL,
        }
        cls._capabilities = dict(defaults)

    @classmethod
    def register_capabilities(cls, provider: str, model_name: str, capabilities: ModelCapabilities) -> None:
        with cls._lock:
            cls._capabilities[(provider.lower(), model_name)] = capabilities

    @classmethod
    def get_capabilities(cls, provider: str, model_name: str) -> ModelCapabilities:
        with cls._lock:
            if not cls._capabilities:
                cls._init_defaults()

            key = (provider.lower(), model_name)
            if key in cls._capabilities:
                return cls._capabilities[key]

            # Match model family prefixes sorted by descending prefix length
            matching_candidates = [
                (m, caps)
                for (p, m), caps in cls._capabilities.items()
                if p == provider.lower() and cls._matches_registered_name(m, model_name)
            ]
            if matching_candidates:
                matching_candidates.sort(key=lambda x: len(x[0]), reverse=True)
                return matching_candidates[0][1]

            # Fallback default for unregistered models
            return _STANDARD_CHAT_CAPABILITIES

    @classmethod
    def is_registered_model(cls, provider: str, model_name: str) -> bool:
        """Return whether a model has an explicit capability declaration."""
        with cls._lock:
            if not cls._capabilities:
                cls._init_defaults()
            provider = provider.lower()
            return any(
                configured_provider == provider and cls._matches_registered_name(configured_model, model_name)
                for configured_provider, configured_model in cls._capabilities
            )

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._init_defaults()


ModelRegistry._init_defaults()


# ============================================================================
# 2. Scoped Usage Recording & Telemetry
# ============================================================================


@dataclass(frozen=True)
class ScopeContext:
    system: str = "target"  # "target" or "meta"
    run_id: str | None = None
    case_id: str | None = None
    node: str | None = None


_default_scope = ScopeContext()
_active_scope: ContextVar[ScopeContext | None] = ContextVar("_active_scope", default=None)


def get_current_scope() -> ScopeContext:
    scope = _active_scope.get()
    return scope if scope is not None else _default_scope


@contextmanager
def usage_scope(
    system: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    node: str | None = None,
) -> Iterator[ScopeContext]:
    """Context manager setting scoped telemetry execution attributes."""
    parent = get_current_scope()
    new_scope = ScopeContext(
        system=system if system is not None else parent.system,
        run_id=run_id if run_id is not None else parent.run_id,
        case_id=case_id if case_id is not None else parent.case_id,
        node=node if node is not None else parent.node,
    )
    token = _active_scope.set(new_scope)
    try:
        yield new_scope
    finally:
        _active_scope.reset(token)


@dataclass
class UsageRecord:
    timestamp: float
    system: str  # "meta" or "target"
    run_id: str | None
    case_id: str | None
    node: str | None
    provider: str
    model: str
    name: str
    duration_seconds: float
    success: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_incomplete: bool = False
    error: str | None = None


class UsageRecorder:
    """Thread-safe scoped usage recorder and aggregator."""

    _lock = threading.Lock()
    _records: list[UsageRecord] = []
    _legacy_metrics: dict[str, Any] = {
        "meta_usage": {
            "overall": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
        "target_usage": {
            "overall": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
    }

    @classmethod
    def record(cls, record: UsageRecord) -> None:
        with cls._lock:
            cls._records.append(record)

            usage_type = "meta_usage" if record.system == "meta" else "target_usage"
            cls._legacy_metrics.setdefault(usage_type, {})
            cls._legacy_metrics[usage_type].setdefault(
                "overall", {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            )
            cls._legacy_metrics[usage_type]["overall"]["llm_calls"] += 1
            cls._legacy_metrics[usage_type]["overall"]["input_tokens"] += record.input_tokens
            cls._legacy_metrics[usage_type]["overall"]["output_tokens"] += record.output_tokens
            cls._legacy_metrics[usage_type]["overall"]["total_tokens"] += record.total_tokens

            model_key = record.name or record.model
            cls._legacy_metrics[usage_type].setdefault(
                model_key, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            )
            cls._legacy_metrics[usage_type][model_key]["llm_calls"] += 1
            cls._legacy_metrics[usage_type][model_key]["input_tokens"] += record.input_tokens
            cls._legacy_metrics[usage_type][model_key]["output_tokens"] += record.output_tokens
            cls._legacy_metrics[usage_type][model_key]["total_tokens"] += record.total_tokens

    @classmethod
    def get_records(
        cls,
        system: str | None = None,
        run_id: str | None = None,
        case_id: str | None = None,
    ) -> list[UsageRecord]:
        with cls._lock:
            records = list(cls._records)
        if system is not None:
            records = [r for r in records if r.system == system]
        if run_id is not None:
            records = [r for r in records if r.run_id == run_id]
        if case_id is not None:
            records = [r for r in records if r.case_id == case_id]
        return records

    @classmethod
    def get_aggregate(
        cls,
        system: str = "target",
        run_id: str | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        records = cls.get_records(system=system, run_id=run_id, case_id=case_id)
        return {
            "llm_calls": len(records),
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "duration_seconds": round(sum(r.duration_seconds for r in records), 4),
            "success_count": sum(1 for r in records if r.success),
            "error_count": sum(1 for r in records if not r.success),
            "incomplete_usage_count": sum(1 for r in records if r.usage_incomplete),
        }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._records.clear()
            cls._legacy_metrics.clear()
            cls._legacy_metrics.update(
                {
                    "meta_usage": {
                        "overall": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    },
                    "target_usage": {
                        "overall": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    },
                }
            )


# ============================================================================
# 3. Message Conversion & Tool Call Protocol Validation
# ============================================================================


def convert_to_messages(input_data: Any) -> list[BaseMessage]:
    """Convert supported input representations into LangChain BaseMessage objects."""
    if isinstance(input_data, str):
        return [HumanMessage(content=input_data)]
    if isinstance(input_data, BaseMessage):
        return [input_data]
    if isinstance(input_data, dict):
        return [_dict_to_message(input_data)]

    if isinstance(input_data, (list, tuple)):
        converted: list[BaseMessage] = []
        for item in input_data:
            if isinstance(item, str):
                converted.append(HumanMessage(content=item))
            elif isinstance(item, BaseMessage):
                converted.append(item)
            elif isinstance(item, dict):
                converted.append(_dict_to_message(item))
            elif isinstance(item, tuple) and len(item) == 2:
                role, content = item
                converted.append(_dict_to_message({"role": str(role), "content": str(content)}))
            else:
                raise ValueError(f"Unsupported message item type in sequence: {type(item).__name__}")
        return converted

    raise ValueError(f"Unsupported input type for message conversion: {type(input_data).__name__}")


def _dict_to_message(data: dict[str, Any]) -> BaseMessage:
    role = data.get("role")
    content = data.get("content", "")
    msg_id = data.get("id")

    if role in ("system",):
        return SystemMessage(content=content, id=msg_id)
    if role in ("user", "human"):
        return HumanMessage(content=content, id=msg_id)
    if role in ("assistant", "ai"):
        tool_calls = data.get("tool_calls") or []
        invalid_tool_calls = data.get("invalid_tool_calls") or []
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            id=msg_id,
        )
    if role in ("tool",):
        cid = data.get("tool_call_id")
        return ToolMessage(content=content, tool_call_id=cid, name=data.get("name"), id=msg_id)

    raise ValueError(f"Unknown message role in dictionary: '{role}'")


def has_image_content(messages: Sequence[BaseMessage]) -> bool:
    """Check if any message in the sequence contains image blocks."""
    for msg in messages:
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") in ("image_url", "image"):
                    return True
    return False


# ============================================================================
# 4. Model Resolution & Provider Construction
# ============================================================================


def get_allowed_target_models() -> list[dict[str, Any]]:
    """Return the list of allowed target models from ChatModel or active TaskSpec."""
    if ChatModel.allowed_target_models is not None:
        return ChatModel.allowed_target_models

    task_spec_path = os.environ.get("ADAS_TASK_SPEC_PATH") or SANDBOX_TASK_SPEC_PATH
    if Path(task_spec_path).exists():
        try:
            from adas_core.task_spec import TaskSpec

            spec = TaskSpec.from_file(task_spec_path)
            if spec.available_models:
                return [m.model_dump() for m in spec.available_models]
        except Exception as e:
            logger.debug(f"Could not load available_models from {task_spec_path}: {e}")

    return [{"provider": "openai", "model_name": "gpt-5.6-luna"}]


def _resolve_target_model(
    model: str | None,
    provider: str | None,
    allowed: list[dict[str, Any]],
) -> tuple[str, str]:
    """Resolve a provider and model name strictly against the allowed models list."""
    if model is None and provider is None:
        if not allowed:
            return "openai", "gpt-5.6-luna"
        first = allowed[0]
        p = first.get("provider", "openai")
        return p, first["model_name"]

    if model is not None and provider is None:
        matches = [m for m in allowed if m.get("model_name") == model]
        if not matches:
            raise ModelConfigurationError(f"Model '{model}' is not available. Allowed Models: {allowed}")
        if len(matches) > 1:
            providers = [m.get("provider", "openai") for m in matches]
            raise ModelConfigurationError(
                f"Model '{model}' is ambiguous across multiple providers {providers}. Please specify provider explicitly."
            )
        p = matches[0].get("provider", "openai")
        return p, model

    if model is not None and provider is not None:
        p_lower = provider.lower()
        matches = [
            m for m in allowed if m.get("provider", "openai").lower() == p_lower and m.get("model_name") == model
        ]
        if not matches:
            raise ModelConfigurationError(
                f"Model '{model}' with provider '{provider}' is not available. Allowed Models: {allowed}"
            )
        matched_provider = matches[0].get("provider", provider)
        return matched_provider, model

    # provider is given but model is None
    if provider is None:
        raise ModelConfigurationError("Provider cannot be None when resolving target model.")
    p_lower = provider.lower()
    provider_models = [m for m in allowed if m.get("provider", "openai").lower() == p_lower]
    if not provider_models:
        raise ModelConfigurationError(f"No allowed models found for provider '{provider}'. Allowed Models: {allowed}")
    first = provider_models[0]
    matched_provider = first.get("provider", provider)
    return matched_provider, first["model_name"]


def _create_provider_runnable(
    provider: str,
    model: str,
    capabilities: ModelCapabilities,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Validate capabilities and construct the underlying LangChain chat runnable."""
    if provider.lower() != "openai":
        raise ModelConfigurationError(f"Unsupported provider: '{provider}'. Supported providers: 'openai'")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ModelConfigurationError(
            f"Missing environment variable: OPENAI_API_KEY required for provider '{provider}'"
        )

    if temperature is not None:
        if not capabilities.supports_temperature:
            raise ModelConfigurationError(f"Model '{model}' ({provider}) does not support 'temperature'.")
        min_temp, max_temp = capabilities.temperature_range
        if not (min_temp <= temperature <= max_temp):
            raise ModelConfigurationError(
                f"Temperature {temperature} is out of range ({min_temp}, {max_temp}) for model '{model}'."
            )

    if reasoning_effort is not None:
        if not capabilities.supports_reasoning_effort:
            raise ModelConfigurationError(f"Model '{model}' ({provider}) does not support 'reasoning_effort'.")
        if reasoning_effort not in capabilities.supported_reasoning_efforts:
            raise ModelConfigurationError(
                f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. Supported: {sorted(capabilities.supported_reasoning_efforts)}"
            )

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": SecretStr(api_key),
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return ChatOpenAI(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatOpenAI for model '{model}': {e!s}") from e


_token_counter_cache: dict[tuple[str, str], Any] = {}


def _get_provider_token_counter(provider: str, model: str) -> Any:
    """Return a provider runnable suitable for offline token counting."""
    key = (provider.lower(), model)
    if key in _token_counter_cache:
        return _token_counter_cache[key]

    if provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or "dummy_key_for_token_counting"
        runnable = ChatOpenAI(model=model, api_key=SecretStr(api_key))
        _token_counter_cache[key] = runnable
        return runnable

    raise ValueError(f"Unsupported provider for token counter: '{provider}'")


class _TokenCounterDescriptor:
    """
    Descriptor providing model-specific and provider-specific token counter.
    - On an instance (llm.token_counter): returns the instance's own provider_runnable.
    - On the class (ChatModel.token_counter): returns the provider_runnable for the active allowed model.
    """

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        if instance is not None and getattr(instance, "_raw_model", None) is not None:
            return instance._raw_model
        allowed = get_allowed_target_models()
        provider, model = _resolve_target_model(None, None, allowed)
        return _get_provider_token_counter(provider, model)


def _normalize_ai_message(message: Any) -> Any:
    """Normalize Responses API content blocks to standard string content when applicable."""
    if isinstance(message, dict) and "raw" in message:
        _normalize_ai_message(message["raw"])
        return message
    if not isinstance(message, AIMessage):
        return message
    if isinstance(message.content, list):
        reasoning_blocks = [b for b in message.content if isinstance(b, dict) and b.get("type") == "reasoning"]
        if reasoning_blocks and "reasoning" not in message.additional_kwargs:
            message.additional_kwargs["reasoning"] = reasoning_blocks

        web_search_blocks = [b for b in message.content if isinstance(b, dict) and b.get("type") == "web_search_call"]
        if web_search_blocks and "web_search_calls" not in message.additional_kwargs:
            message.additional_kwargs["web_search_calls"] = web_search_blocks

        text_blocks = [b for b in message.content if isinstance(b, dict) and b.get("type") == "text"]
        citations: list[dict[str, Any]] = []
        for tb in text_blocks:
            for ann in tb.get("annotations", []) or []:
                if isinstance(ann, dict) and ann.get("type") == "url_citation":
                    citations.append(ann)
        if citations and "citations" not in message.additional_kwargs:
            message.additional_kwargs["citations"] = citations

        non_text_blocks = [
            b
            for b in message.content
            if isinstance(b, dict) and b.get("type") not in ("text", "reasoning", "function_call", "web_search_call")
        ]
        if text_blocks and not non_text_blocks:
            message.content = "".join(b.get("text", "") for b in text_blocks)
        elif not non_text_blocks:
            message.content = ""
    return message


def _normalize_tool_spec(
    tool: Any,
    capabilities: ModelCapabilities,
    provider: str,
    model: str,
) -> Any:
    """Normalize and validate a tool specification for default_tools."""
    if tool == "web_search":
        if not capabilities.supports_web_search:
            raise ModelConfigurationError(f"Model '{model}' ({provider}) does not support web search.")
        return {"type": "web_search"}
    if isinstance(tool, dict):
        if tool.get("type") == "web_search":
            if not capabilities.supports_web_search:
                raise ModelConfigurationError(f"Model '{model}' ({provider}) does not support web search.")
        return tool
    if getattr(tool, "name", None) and callable(getattr(tool, "invoke", None)):
        return tool
    raise ValueError(
        f"All values in default_tools must be tool instances with 'name' and callable 'invoke', "
        f"a tool dictionary (e.g. {{'type': 'web_search'}}), or 'web_search'. Got: {tool!r}"
    )


# ============================================================================
# 6. ChatModel Public API
# ============================================================================


class ChatModel:
    """
    Composition-based chat model delegator for ADAS target systems and meta-agent.
    Enforces allowlist authorization, tool-history protocol, and scoped telemetry.
    """

    allowed_target_models: list[dict[str, Any]] | None = None
    usage_metrics: dict[str, Any] = UsageRecorder._legacy_metrics
    token_counter: Any = _TokenCounterDescriptor()

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        name: str | None = None,
        is_meta: bool | None = None,
        default_tools: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        scope = get_current_scope()
        is_meta_effective = (is_meta is True) or (scope.system == "meta")

        matched_allowed_spec: dict[str, Any] | None = None
        if is_meta_effective:
            effective_provider = provider or "openai"
            effective_model = model or "gpt-5.6-luna"
        else:
            allowed = get_allowed_target_models()
            effective_provider, effective_model = _resolve_target_model(model, provider, allowed)
            matched_allowed_spec = next(
                (
                    m
                    for m in allowed
                    if m.get("provider", "openai").lower() == effective_provider.lower()
                    and m.get("model_name") == effective_model
                ),
                None,
            )

        capabilities = ModelRegistry.get_capabilities(effective_provider, effective_model)
        runnable = _create_provider_runnable(
            provider=effective_provider,
            model=effective_model,
            capabilities=capabilities,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        is_target_web_search_authorized = bool(matched_allowed_spec and matched_allowed_spec.get("enable_web_search"))

        normalized_default_tools: list[Any] = []
        if default_tools:
            for t in default_tools:
                norm_tool = _normalize_tool_spec(t, capabilities, effective_provider, effective_model)
                if not is_meta_effective and isinstance(norm_tool, dict) and norm_tool.get("type") == "web_search":
                    if not is_target_web_search_authorized:
                        raise ModelConfigurationError(
                            f"Model '{effective_model}' ({effective_provider}) is not authorized for web search by active TaskSpec."
                        )
                normalized_default_tools.append(norm_tool)

        self.model: str = effective_model
        self.provider: str = effective_provider
        self.model_name: str = effective_model
        self.name: str = name if name else effective_model
        self.capabilities: ModelCapabilities = capabilities
        self.is_meta: bool = is_meta_effective
        self.default_tools: tuple[Any, ...] = tuple(normalized_default_tools)
        self._bound_client_tools: tuple[Any, ...] = ()
        self._raw_model: Any = runnable
        if self.default_tools:
            self._runnable = runnable.bind_tools(list(self.default_tools), parallel_tool_calls=False)
        else:
            self._runnable = runnable
        self._response_transformer: Callable[[Any], Any] | None = None

    @classmethod
    def _from_runnable(
        cls,
        runnable: Any,
        raw_model: Any,
        provider: str,
        model: str,
        capabilities: ModelCapabilities,
        name: str,
        is_meta: bool,
        default_tools: tuple[Any, ...] = (),
        bound_client_tools: tuple[Any, ...] = (),
        response_transformer: Callable[[Any], Any] | None = None,
    ) -> ChatModel:
        instance = cls.__new__(cls)
        instance.model = model
        instance.model_name = model
        instance.provider = provider
        instance.capabilities = capabilities
        instance.name = name
        instance.is_meta = is_meta
        instance.default_tools = default_tools
        instance._bound_client_tools = bound_client_tools
        instance._raw_model = raw_model
        instance._runnable = runnable
        instance._response_transformer = response_transformer
        return instance

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatModel:
        """Return a NEW ChatModel instance with tools bound, preserving telemetry and immutability."""
        if not tools and not self.default_tools:
            return self

        normalized_tools: list[Any] = []
        if tools:
            for t in tools:
                if t == "web_search" or (isinstance(t, dict) and t.get("type") == "web_search"):
                    raise ValueError(
                        "bind_tools is reserved for client-side tool instances with 'name' and callable 'invoke'. "
                        "For server-side provider tools like web search, pass default_tools=['web_search'] to ChatModel."
                    )
                if not (getattr(t, "name", None) and callable(getattr(t, "invoke", None))):
                    raise ValueError(
                        f"All values in tools must be tool instances with 'name' and callable 'invoke'. "
                        f"For server-side provider tools like web search, pass default_tools=['web_search'] to ChatModel. Got: {t!r}"
                    )
                normalized_tools.append(t)

        combined: list[Any] = list(normalized_tools)
        for t in self.default_tools:
            if t not in combined:
                combined.append(t)

        bind_kwargs = dict(kwargs)
        bind_kwargs["parallel_tool_calls"] = parallel_tool_calls

        bound = self._raw_model.bind_tools(combined, **bind_kwargs)
        return ChatModel._from_runnable(
            runnable=bound,
            raw_model=self._raw_model,
            provider=self.provider,
            model=self.model,
            capabilities=self.capabilities,
            name=self.name,
            is_meta=self.is_meta,
            default_tools=self.default_tools,
            bound_client_tools=tuple(normalized_tools),
            response_transformer=self._response_transformer,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> ChatModel:
        """Return a NEW ChatModel instance bound to produce structured output."""
        if not self.capabilities.supports_structured_output:
            raise ModelConfigurationError(f"Model '{self.model}' ({self.provider}) does not support structured output.")

        if "include_raw" in kwargs:
            raise ModelConfigurationError("ChatModel manages include_raw internally to preserve usage telemetry.")

        if self.default_tools:
            if "tools" in kwargs:
                raise ModelConfigurationError(
                    "ChatModel manages tools for structured output; bind client-side tools with bind_tools() instead."
                )
            if "method" in kwargs and kwargs["method"] != "json_schema":
                raise ModelConfigurationError("Structured output with server-side tools requires method='json_schema'.")
            if "strict" in kwargs and kwargs["strict"] is not True:
                raise ModelConfigurationError("Structured output with server-side tools requires strict=True.")

            # LangChain's public API supports provider tools with structured output only via json_schema + strict + include_raw.
            structured_tools = [*self._bound_client_tools, *self.default_tools]
            structured = self._raw_model.with_structured_output(
                schema,
                method="json_schema",
                strict=True,
                include_raw=True,
                tools=structured_tools,
                **kwargs,
            )
        else:
            structured = self._runnable.with_structured_output(schema, include_raw=True, **kwargs)

        def extract_parsed_output(response: Any) -> Any:
            if not isinstance(response, dict) or "raw" not in response:
                raise StructuredOutputError(
                    "Structured-output runnable did not return the expected raw response envelope."
                )
            parsing_error = response.get("parsing_error")
            if parsing_error:
                if isinstance(parsing_error, BaseException):
                    raise parsing_error
                raise StructuredOutputError(f"Could not parse structured model output: {parsing_error}")
            return response.get("parsed")

        return ChatModel._from_runnable(
            runnable=structured,
            raw_model=self._raw_model,
            provider=self.provider,
            model=self.model,
            capabilities=self.capabilities,
            name=self.name,
            is_meta=self.is_meta,
            default_tools=self.default_tools,
            bound_client_tools=self._bound_client_tools,
            response_transformer=extract_parsed_output,
        )

    def get_num_tokens_from_messages(self, messages: Sequence[BaseMessage] | list[Any]) -> int:
        """Count tokens for the given messages using the model's tokenizer."""
        converted = convert_to_messages(messages)
        if hasattr(self._raw_model, "get_num_tokens_from_messages"):
            try:
                val = self._raw_model.get_num_tokens_from_messages(converted)
                if isinstance(val, int):
                    return val
            except Exception:
                pass
        counter = _get_provider_token_counter(self.provider, self.model)
        return counter.get_num_tokens_from_messages(converted)

    def _record_call(
        self,
        start_time: float,
        response: Any,
        error: Exception | None = None,
        interrupted_stream: bool = False,
    ) -> None:
        duration = time.perf_counter() - start_time
        scope = get_current_scope()
        system = "meta" if (self.is_meta or scope.system == "meta") else "target"

        success = (error is None) and not interrupted_stream
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        usage_incomplete = False

        if response is not None:
            usage_response = response.get("raw") if isinstance(response, dict) and "raw" in response else response
            if isinstance(usage_response, list):
                for m in usage_response:
                    u = getattr(m, "usage_metadata", None)
                    if u:
                        input_tokens += u.get("input_tokens", 0)
                        output_tokens += u.get("output_tokens", 0)
                        total_tokens += u.get("total_tokens", 0)
            else:
                u = getattr(usage_response, "usage_metadata", None)
                if u:
                    input_tokens = u.get("input_tokens", 0)
                    output_tokens = u.get("output_tokens", 0)
                    total_tokens = u.get("total_tokens", 0)
                else:
                    usage_incomplete = True
        else:
            usage_incomplete = True

        record = UsageRecord(
            timestamp=time.time(),
            system=system,
            run_id=scope.run_id,
            case_id=scope.case_id,
            node=scope.node,
            provider=self.provider,
            model=self.model,
            name=self.name,
            duration_seconds=round(duration, 4),
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_incomplete=usage_incomplete,
            error=str(error) if error else None,
        )
        UsageRecorder.record(record)

    def _validate_input_capabilities(self, messages: Sequence[BaseMessage]) -> None:
        """Validate that input modalities match declared model capabilities."""
        if not self.capabilities.supports_vision and has_image_content(messages):
            raise ModelConfigurationError(
                f"Model '{self.model}' ({self.provider}) does not support vision/image inputs."
            )

    def invoke(
        self,
        input: Any,
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = convert_to_messages(input)
        validate_tool_history(messages)
        self._validate_input_capabilities(messages)

        response: Any = None
        start_time = time.perf_counter()
        try:
            response = self._runnable.invoke(messages, config=config, **kwargs)
            response = _normalize_ai_message(response)
            result = self._response_transformer(response) if self._response_transformer else response
            if count_metrics:
                self._record_call(start_time, response)
            return result
        except Exception as e:
            if count_metrics:
                self._record_call(start_time, response, error=e)
            raise

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = convert_to_messages(input)
        validate_tool_history(messages)
        self._validate_input_capabilities(messages)

        response: Any = None
        start_time = time.perf_counter()
        try:
            response = await self._runnable.ainvoke(messages, config=config, **kwargs)
            response = _normalize_ai_message(response)
            result = self._response_transformer(response) if self._response_transformer else response
            if count_metrics:
                self._record_call(start_time, response)
            return result
        except Exception as e:
            if count_metrics:
                self._record_call(start_time, response, error=e)
            raise

    def stream(
        self,
        input: Any,
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        messages = convert_to_messages(input)
        validate_tool_history(messages)
        self._validate_input_capabilities(messages)

        start_time = time.perf_counter()
        accumulated_usage = None
        interrupted = False
        error_encountered: Exception | None = None

        try:
            for chunk in self._runnable.stream(messages, config=config, **kwargs):
                if getattr(chunk, "usage_metadata", None):
                    accumulated_usage = chunk.usage_metadata
                yield _normalize_ai_message(chunk)
        except Exception as e:
            interrupted = True
            error_encountered = e
            raise
        finally:
            if count_metrics:
                dummy_response = None
                if accumulated_usage:
                    dummy_response = AIMessage(content="", usage_metadata=accumulated_usage)
                self._record_call(
                    start_time,
                    dummy_response,
                    error=error_encountered,
                    interrupted_stream=interrupted,
                )

    async def astream(
        self,
        input: Any,
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        messages = convert_to_messages(input)
        validate_tool_history(messages)
        self._validate_input_capabilities(messages)

        start_time = time.perf_counter()
        accumulated_usage = None
        interrupted = False
        error_encountered: Exception | None = None

        try:
            async for chunk in self._runnable.astream(messages, config=config, **kwargs):
                if getattr(chunk, "usage_metadata", None):
                    accumulated_usage = chunk.usage_metadata
                yield _normalize_ai_message(chunk)
        except Exception as e:
            interrupted = True
            error_encountered = e
            raise
        finally:
            if count_metrics:
                dummy_response = None
                if accumulated_usage:
                    dummy_response = AIMessage(content="", usage_metadata=accumulated_usage)
                self._record_call(
                    start_time,
                    dummy_response,
                    error=error_encountered,
                    interrupted_stream=interrupted,
                )

    def batch(
        self,
        inputs: list[Any],
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError(
            "ChatModel.batch() is not supported. Invoke models separately; "
            "use LangGraph parallel branches for concurrent agent work."
        )

    async def abatch(
        self,
        inputs: list[Any],
        config: Any = None,
        count_metrics: bool = True,
        is_meta: bool | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError(
            "ChatModel.abatch() is not supported. Invoke models separately; "
            "use LangGraph parallel branches for concurrent agent work."
        )

    def __call__(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.invoke(input, config=config, **kwargs)


__all__ = [
    "ChatModel",
    "ModelCapabilities",
    "ModelRegistry",
    "UsageRecord",
    "UsageRecorder",
    "convert_to_messages",
    "execute_tool_calls",
    "get_allowed_target_models",
    "get_current_scope",
    "has_image_content",
    "usage_scope",
    "validate_tool_history",
]
