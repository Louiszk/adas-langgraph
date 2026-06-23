import os
import threading
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from config import settings

load_dotenv()
_metrics_lock = threading.Lock()


def execute_tool_calls(
    response: AIMessage, available_tools: Dict[str, BaseTool]
) -> Tuple[List[ToolMessage], Dict[str, Any]]:
    """Execute available tool calls from the response."""
    tool_messages = []
    tool_results = {}

    # Handle invalid tool calls if present
    for invalid_call in getattr(response, "invalid_tool_calls", []) or []:
        if isinstance(invalid_call, dict):
            err = invalid_call.get("error")
            cid = invalid_call.get("id")
            name = invalid_call.get("name")
        else:
            err = getattr(invalid_call, "error", None)
            cid = getattr(invalid_call, "id", None)
            name = getattr(invalid_call, "name", None)

        content = f"Failed to parse tool call. Error: {err}"
        tool_messages.append(ToolMessage(content=content, tool_call_id=cid, name=name))

    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls:
        return tool_messages, tool_results

    for tool_call in tool_calls:
        if not tool_call:
            continue

        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {}) or {}
            tool_id = tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {}) or {}
            tool_id = getattr(tool_call, "id", None)

        if not tool_name:
            tool_messages.append(
                ToolMessage(
                    content="Malformed tool call: missing name",
                    tool_call_id=tool_id,
                    name=None,
                )
            )
            continue

        if tool_name in available_tools:
            try:
                result = available_tools[tool_name].invoke(tool_args)
                content = str(result) if result is not None else f"Tool {tool_name} executed successfully."
                tool_messages.append(ToolMessage(content=content, tool_call_id=tool_id, name=tool_name))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {repr(e)}"
                tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_id, name=tool_name))
                tool_results[tool_name] = error_message
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found.",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

    return tool_messages, tool_results


def get_model(
    wrapper: str,
    model_name: str,
    temperature: float,
    is_meta: bool,
    reasoning_effort: Optional[str] = None,
) -> Any:
    """Return an initialized LLM wrapper for the requested provider/model."""
    api_keys = {"openai": "OPENAI_API_KEY", "perplexity": "PERPLEXITY_API_KEY"}

    if wrapper not in api_keys:
        raise ValueError(f"Invalid wrapper: '{wrapper}'. Supported: {', '.join(api_keys.keys())}")

    key_name = api_keys[wrapper]
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"Missing environment variable: {key_name} required for {wrapper}")

    # If this is a target model, check allowed list shape defensively
    if not is_meta:
        allowed_models = settings.allowed_target_models or []
        if not any(
            isinstance(m, dict) and m.get("wrapper") == wrapper and m.get("model_name") == model_name
            for m in allowed_models
        ):
            raise ValueError(
                f"ERROR: Model {model_name} is not available. Allowed Models: {settings.allowed_target_models}"
            )

    try:
        if wrapper == "openai":
            if reasoning_effort:
                return ChatOpenAI(
                    model=model_name,
                    reasoning_effort=reasoning_effort,
                    api_key=SecretStr(api_key),
                )
            return ChatOpenAI(model=model_name, temperature=temperature, api_key=SecretStr(api_key))

        if wrapper == "perplexity":
            return ChatOpenAI(
                model="sonar",
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url="https://api.perplexity.ai",
            )

        raise ValueError(f"Unsupported wrapper: {wrapper}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {wrapper} model: {str(e)}") from e


class LargeLanguageModel:
    usage_metrics = {
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

    token_counter = ChatOpenAI(model="gpt-4o", api_key=SecretStr("x"))

    def __init__(
        self,
        temperature: float = 0.2,
        wrapper: str = "openai",
        model_name: str = "gpt-4.1-mini",
        name: Optional[str] = None,
        is_meta: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.name = name if name else model_name
        self.model_name = model_name
        self.model = get_model(wrapper, model_name, temperature, is_meta, reasoning_effort)
        self.wrapper = wrapper

    def _ensure_usage_entry(self, usage_type: str) -> None:
        LargeLanguageModel.usage_metrics.setdefault(usage_type, {})
        LargeLanguageModel.usage_metrics[usage_type].setdefault(
            self.name,
            {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    def bind_tools(
        self,
        tool_objects: List[Any],
        function_call_type: str = "normal",
        parallel_tool_calls: bool = True,
    ) -> "LargeLanguageModel":
        if not tool_objects:
            return self

        if function_call_type == "normal":
            invalid = []
            for t in tool_objects:
                if not getattr(t, "name", None) or not callable(getattr(t, "invoke", None)):
                    invalid.append(t)

            if invalid:
                raise ValueError("All values in tool_objects must be tool instances.")

        self.model = self.model.bind_tools(tool_objects, parallel_tool_calls=parallel_tool_calls)

        return self

    def invoke(
        self,
        messages_input: List[Any],
        count_metrics: bool = True,
        is_meta: bool = False,
    ) -> Any:

        converted_messages = []
        for msg in messages_input:
            if isinstance(msg, str):
                converted_messages.append(HumanMessage(content=msg))
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    converted_messages.append(SystemMessage(content=content))
                elif role == "user":
                    converted_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    converted_messages.append(AIMessage(content=content))
                elif role == "tool":
                    converted_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id")))
            else:
                converted_messages.append(msg)

        messages_input = converted_messages

        def _get_call_id(call: Any) -> Optional[Any]:
            if isinstance(call, dict):
                return call.get("id")
            return getattr(call, "id", None)

        ai_tool_call_ids = set()
        tool_message_ids = set()
        for msg in messages_input:
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", []) or []
                invalid_tool_calls = getattr(msg, "invalid_tool_calls", []) or []
                for call in tool_calls + invalid_tool_calls:
                    call_id = _get_call_id(call)
                    if call_id is not None:
                        ai_tool_call_ids.add(call_id)

            if isinstance(msg, ToolMessage):
                if hasattr(msg, "tool_call_id"):
                    if getattr(msg, "tool_call_id"):
                        tool_message_ids.add(msg.tool_call_id)
                else:
                    raise ValueError(
                        "INVOKE ERROR: A ToolMessage did not contain the 'tool_call_id' attribute. Do NOT construct ToolMessages manually."
                    )

        missing_tool_msgs = ai_tool_call_ids - tool_message_ids
        if missing_tool_msgs:
            raise ValueError(
                f"INVOKE ERROR: AIMessages contain tool_call IDs with no corresponding ToolMessages: {missing_tool_msgs}"
            )

        # Filter out orphaned ToolMessages from trimming
        messages_input = [
            msg
            for msg in messages_input
            if not isinstance(msg, ToolMessage) or getattr(msg, "tool_call_id") in ai_tool_call_ids
        ]

        # Filter out empty messages to avoid client errors
        messages = [
            msg
            for msg in messages_input
            if getattr(msg, "content", None)
            or getattr(msg, "tool_calls", None)
            or getattr(msg, "invalid_tool_calls", None)
            or getattr(msg, "tool_call_id", None)
        ]
        if len(messages) != len(messages_input):
            print(f"Filtered out {len(messages_input) - len(messages)} empty messages before invoking.")

        try:
            response = self.model.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {repr(e)}") from e

        if count_metrics:
            usage_type = "meta_usage" if is_meta else "target_usage"
            self._ensure_usage_entry(usage_type)

            with _metrics_lock:
                LargeLanguageModel.usage_metrics[usage_type][self.name]["llm_calls"] += 1
                LargeLanguageModel.usage_metrics[usage_type]["overall"]["llm_calls"] += 1

            input_tokens = output_tokens = total_tokens = 0
            processed_message_ids = set()
            resp_messages = response if isinstance(response, list) else [response]

            for msg in resp_messages:
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id in processed_message_ids:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)
                    if msg_id:
                        processed_message_ids.add(msg_id)

            # update metrics under a lock
            with _metrics_lock:
                LargeLanguageModel.usage_metrics[usage_type][self.name]["input_tokens"] += input_tokens
                LargeLanguageModel.usage_metrics[usage_type][self.name]["output_tokens"] += output_tokens
                LargeLanguageModel.usage_metrics[usage_type][self.name]["total_tokens"] += total_tokens
                LargeLanguageModel.usage_metrics[usage_type]["overall"]["input_tokens"] += input_tokens
                LargeLanguageModel.usage_metrics[usage_type]["overall"]["output_tokens"] += output_tokens
                LargeLanguageModel.usage_metrics[usage_type]["overall"]["total_tokens"] += total_tokens

        return response
