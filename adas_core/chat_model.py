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
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from adas_core.logging_config import get_logger
from adas_core.tool_calls import execute_tool_calls, validate_tool_history

logger = get_logger("chat_model")
load_dotenv()


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


_STANDARD_CHAT_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    temperature_range=(0.0, 2.0),
    supports_reasoning_effort=False,
    supported_reasoning_efforts=frozenset(),
    supports_structured_output=True,
    supports_vision=True,
)

_LEGACY_TEXT_ONLY_CAPABILITIES = ModelCapabilities(
    supports_temperature=True,
    temperature_range=(0.0, 2.0),
    supports_reasoning_effort=False,
    supported_reasoning_efforts=frozenset(),
    supports_structured_output=True,
    supports_vision=False,
)

_OPENAI_REASONING_VISION = ModelCapabilities(
    supports_temperature=False,
    supports_reasoning_effort=True,
    supported_reasoning_efforts=frozenset({"low", "medium", "high"}),
    supports_structured_output=True,
    supports_vision=True,
)

_OPENAI_REASONING_TEXT_ONLY = ModelCapabilities(
    supports_temperature=False,
    supports_reasoning_effort=True,
    supported_reasoning_efforts=frozenset({"low", "medium", "high"}),
    supports_structured_output=True,
    supports_vision=False,
)

_OPENAI_REASONING_FULL = ModelCapabilities(
    supports_temperature=False,
    supports_reasoning_effort=True,
    supported_reasoning_efforts=frozenset({"none", "minimal", "low", "medium", "high", "xhigh"}),
    supports_structured_output=True,
    supports_vision=True,
)


class ModelRegistry:
    """Internal registry of model capabilities and known provider specifications."""

    _lock = threading.Lock()
    _capabilities: dict[tuple[str, str], ModelCapabilities] = {}

    @classmethod
    def _init_defaults(cls) -> None:
        defaults: dict[tuple[str, str], ModelCapabilities] = {
            # OpenAI standard models
            ("openai", "gpt-4o"): _STANDARD_CHAT_CAPABILITIES,
            ("openai", "gpt-4o-mini"): _STANDARD_CHAT_CAPABILITIES,
            ("openai", "gpt-4-turbo"): _STANDARD_CHAT_CAPABILITIES,
            ("openai", "gpt-4"): _LEGACY_TEXT_ONLY_CAPABILITIES,
            ("openai", "gpt-3.5-turbo"): _LEGACY_TEXT_ONLY_CAPABILITIES,
            # OpenAI reasoning models
            ("openai", "o1"): _OPENAI_REASONING_VISION,
            ("openai", "o1-mini"): _OPENAI_REASONING_TEXT_ONLY,
            ("openai", "o1-preview"): ModelCapabilities(
                supports_temperature=False,
                supports_reasoning_effort=False,
                supports_structured_output=False,
                supports_vision=False,
            ),
            ("openai", "o3"): _OPENAI_REASONING_VISION,
            ("openai", "o3-mini"): _OPENAI_REASONING_TEXT_ONLY,
            # OpenAI GPT-5.6 family
            ("openai", "gpt-5.6-sol"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.6-terra"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.6-luna"): _OPENAI_REASONING_FULL,
            ("openai", "gpt-5.6"): _OPENAI_REASONING_FULL,
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
                if p == provider.lower() and model_name.startswith(m)
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
                configured_provider == provider and model_name.startswith(configured_model)
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


_active_scope: ContextVar[ScopeContext] = ContextVar("_active_scope", default=ScopeContext())


def get_current_scope() -> ScopeContext:
    return _active_scope.get()


@contextmanager
def usage_scope(
    system: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    node: str | None = None,
) -> Iterator[ScopeContext]:
    """Context manager setting scoped telemetry execution attributes."""
    parent = _active_scope.get()
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


def get_allowed_target_models() -> list[dict[str, str]]:
    """Return the list of allowed target models from ChatModel or active TaskSpec."""
    if ChatModel.allowed_target_models is not None:
        return ChatModel.allowed_target_models

    task_spec_path = os.environ.get("ADAS_TASK_SPEC_PATH") or "/sandbox/workspace/task_setup/task.json"
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
    allowed: list[dict[str, str]],
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
            raise ValueError(f"Model '{model}' is not available. Allowed Models: {allowed}")
        if len(matches) > 1:
            providers = [m.get("provider", "openai") for m in matches]
            raise ValueError(
                f"Model '{model}' is ambiguous across multiple providers {providers}. Please specify provider explicitly."
            )
        p = matches[0].get("provider", "openai")
        return p, model

    if model is not None and provider is not None:
        matches = [m for m in allowed if m.get("provider", "openai") == provider and m.get("model_name") == model]
        if not matches:
            raise ValueError(f"Model '{model}' with provider '{provider}' is not available. Allowed Models: {allowed}")
        return provider, model

    # provider is given but model is None
    if provider is None:
        raise ValueError("Provider cannot be None when resolving target model.")
    provider_models = [m for m in allowed if m.get("provider", "openai") == provider]
    if not provider_models:
        raise ValueError(f"No allowed models found for provider '{provider}'. Allowed Models: {allowed}")
    first = provider_models[0]
    return provider, first["model_name"]


def _create_provider_runnable(
    provider: str,
    model: str,
    capabilities: ModelCapabilities,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Validate capabilities and construct the underlying LangChain chat runnable."""
    if provider.lower() != "openai":
        raise ValueError(f"Unsupported provider: '{provider}'. Supported providers: 'openai'")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"Missing environment variable: OPENAI_API_KEY required for provider '{provider}'")

    if temperature is not None:
        if not capabilities.supports_temperature:
            raise ValueError(f"Model '{model}' ({provider}) does not support 'temperature'.")
        min_temp, max_temp = capabilities.temperature_range
        if not (min_temp <= temperature <= max_temp):
            raise ValueError(f"Temperature {temperature} is out of range ({min_temp}, {max_temp}) for model '{model}'.")

    if reasoning_effort is not None:
        if not capabilities.supports_reasoning_effort:
            raise ValueError(f"Model '{model}' ({provider}) does not support 'reasoning_effort'.")
        if reasoning_effort not in capabilities.supported_reasoning_efforts:
            raise ValueError(
                f"Invalid reasoning_effort '{reasoning_effort}' for model '{model}'. Supported: {sorted(capabilities.supported_reasoning_efforts)}"
            )

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": SecretStr(api_key),
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


# ============================================================================
# 6. ChatModel Public API
# ============================================================================


class ChatModel:
    """
    Composition-based chat model delegator for ADAS target systems and meta-agent.
    Enforces allowlist authorization, tool-history protocol, and scoped telemetry.
    """

    allowed_target_models: list[dict[str, str]] | None = None
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
        **kwargs: Any,
    ) -> None:
        scope = get_current_scope()
        is_meta_effective = (is_meta is True) or (scope.system == "meta")

        if is_meta_effective:
            effective_provider = provider or "openai"
            effective_model = model or "gpt-5.6-luna"
        else:
            allowed = get_allowed_target_models()
            effective_provider, effective_model = _resolve_target_model(model, provider, allowed)

        capabilities = ModelRegistry.get_capabilities(effective_provider, effective_model)
        runnable = _create_provider_runnable(
            provider=effective_provider,
            model=effective_model,
            capabilities=capabilities,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        self.model: str = effective_model
        self.provider: str = effective_provider
        self.model_name: str = effective_model
        self.name: str = name if name else effective_model
        self.capabilities: ModelCapabilities = capabilities
        self.is_meta: bool = is_meta_effective
        self._raw_model: Any = runnable
        self._runnable: Any = runnable
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
        response_transformer: Callable[[Any], Any] | None = None,
    ) -> ChatModel:
        instance = cls.__new__(cls)
        instance.model = model
        instance.model_name = model
        instance.provider = provider
        instance.capabilities = capabilities
        instance.name = name
        instance.is_meta = is_meta
        instance._raw_model = raw_model
        instance._runnable = runnable
        instance._response_transformer = response_transformer
        return instance

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> ChatModel:
        """Return a NEW ChatModel instance with tools bound, preserving telemetry and immutability."""
        if not tools:
            return self

        invalid = []
        for t in tools:
            if not getattr(t, "name", None) or not callable(getattr(t, "invoke", None)):
                invalid.append(t)
        if invalid:
            raise ValueError("All values in tools must be tool instances with 'name' and callable 'invoke'.")

        bind_kwargs = dict(kwargs)
        if parallel_tool_calls is not None:
            bind_kwargs["parallel_tool_calls"] = parallel_tool_calls

        bound = self._runnable.bind_tools(tools, **bind_kwargs)
        return ChatModel._from_runnable(
            runnable=bound,
            raw_model=self._raw_model,
            provider=self.provider,
            model=self.model,
            capabilities=self.capabilities,
            name=self.name,
            is_meta=self.is_meta,
            response_transformer=self._response_transformer,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> ChatModel:
        """Return a NEW ChatModel instance bound to produce structured output."""
        if not self.capabilities.supports_structured_output:
            raise ValueError(f"Model '{self.model}' ({self.provider}) does not support structured output.")

        if "include_raw" in kwargs:
            raise ValueError("ChatModel manages include_raw internally to preserve usage telemetry.")

        structured = self._runnable.with_structured_output(schema, include_raw=True, **kwargs)

        def extract_parsed_output(response: Any) -> Any:
            if not isinstance(response, dict) or "raw" not in response:
                raise RuntimeError("Structured-output runnable did not return the expected raw response envelope.")
            parsing_error = response.get("parsing_error")
            if parsing_error:
                if isinstance(parsing_error, BaseException):
                    raise parsing_error
                raise RuntimeError(f"Could not parse structured model output: {parsing_error}")
            return response.get("parsed")

        return ChatModel._from_runnable(
            runnable=structured,
            raw_model=self._raw_model,
            provider=self.provider,
            model=self.model,
            capabilities=self.capabilities,
            name=self.name,
            is_meta=self.is_meta,
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
            raise ValueError(f"Model '{self.model}' ({self.provider}) does not support vision/image inputs.")

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
                yield chunk
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
                yield chunk
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
