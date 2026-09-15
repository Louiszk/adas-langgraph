import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from adas_core.chat_model import ChatModel, usage_scope
from adas_core.decorator_logic import execute_decorator_tool_calls
from adas_core.exceptions import MetaStateError
from adas_core.helpers import remove_old_test_results
from adas_core.logging_config import get_logger
from adas_core.materialize import materialize_system
from config.settings import (
    ACTION_CUTOFF,
    meta_agent_enable_web_search,
    meta_agent_model,
    meta_agent_reasoning_effort,
    meta_agent_wrapper,
)
from meta_system.prompts import (
    build_meta_agent_prompt,
    decorator_reminder,
    test_reminder,
    trimming_message,
)
from meta_system.state import MetaState
from meta_system.tools import code_related_tools, function_signatures, tools

logger = get_logger("meta_system.nodes")


def normalize_response_content(content: Any) -> str:
    """Normalize string or list of content dicts from LLM response into a single string."""
    if isinstance(content, list):
        content_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                content_parts.append(str(item.get("text", "")))
            else:
                content_parts.append(str(item))
        return " ".join(content_parts)
    return str(content or "")


def formatting_function(state: MetaState) -> dict[str, Any]:
    initial_task = str(state.get("initial_task", ""))
    max_iterations = state.get("max_iterations", 30)

    designer_task = initial_task.split("--- Specific Validation Instructions ---")[0].strip()
    new_task_statement = (
        designer_task + f"\nThe system design process must be completed in no more than {max_iterations} iterations."
    )

    new_state = {
        "messages": [HumanMessage(new_task_statement)],
        "designer_task": HumanMessage(new_task_statement),
        "system_passed": False,
    }
    return new_state


def initial_test_runner_function(state: MetaState) -> dict[str, Any]:
    if not state.get("optimize"):
        return {}

    test_system_tool = tools.get("TestSystem")
    if not test_system_tool:
        raise MetaStateError("TestSystem tool not found.")

    logger.info("--- Running baseline evaluation for initial target system ---")
    test_result_str = test_system_tool.invoke({"state": state})  # type: ignore

    test_metrics = state.get("test_metrics", {})
    passed = test_metrics.get("passed", 0)
    total = test_metrics.get("total", 0)

    if passed >= total:
        target_goal = (
            f"The initial system already passes all {total}/{total} development tests (100% pass rate).\n"
            "Your goal is to optimize the system for greater robustness, token efficiency, and lower latency "
            "while maintaining a 100% pass rate across all tests."
        )
    else:
        target_goal = (
            f"The initial system passed {passed}/{total} development tests.\n"
            f"Improve upon this baseline by achieving passing tests for all {total} test cases."
        )

    verbose_test_results_content = (
        "--- Initial Test Results ---\n"
        f"{test_result_str}\n"
        "These tests were run right at the start of the design process (Iteration 0), before you made any changes to the system.\n\n"
        f"{target_goal}\n\n"
        "Crucially, the system must be generalized and adaptable to the broader problem domain. "
        "Do not hardcode logic tailored only to these specific test inputs."
    )

    pattern_to_remove = r"<FinalState>.*?</FinalState>|<STDOUT\+STDERR>.*?</STDOUT\+STDERR>"
    cleaned_test_results_content = re.sub(pattern_to_remove, "", verbose_test_results_content, flags=re.DOTALL)

    return {
        "verbose_initial_test_results": HumanMessage(content=verbose_test_results_content),
        "initial_test_results": HumanMessage(content=cleaned_test_results_content),
        "test_metrics": test_metrics,
        "candidates": state.get("candidates", []),
    }


def meta_agent_function(state: MetaState) -> dict[str, Any]:
    with usage_scope(system="meta", node="meta_agent"):
        llm = ChatModel(
            provider=meta_agent_wrapper,
            model=meta_agent_model,
            reasoning_effort=meta_agent_reasoning_effort,
            name="MetaAgent",
            default_tools=["web_search"] if meta_agent_enable_web_search else None,
        )

        context_length = ACTION_CUTOFF * 2
        messages = state.get("messages", [])
        target_agentic_system = state.get("target_agentic_system")
        if target_agentic_system is None:
            raise MetaStateError("target_agentic_system is required in MetaState.")

        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        current_messages = messages[1:]
        initial_messages: list[HumanMessage] = []
        if designer_task := state.get("designer_task"):
            initial_messages.append(designer_task)

        if state.get("initial_test_results"):
            if iteration > 0:
                if init_res := state.get("initial_test_results"):
                    initial_messages.append(init_res)
            else:
                if verbose_res := state.get("verbose_initial_test_results"):
                    initial_messages.append(verbose_res)

        trimmed_messages = current_messages
        try:
            trimmed_messages = trim_messages(
                current_messages,
                max_tokens=context_length,
                strategy="last",
                token_counter=len,
                allow_partial=False,
            )
        except Exception as e:
            logger.warning(f"Error during message trimming: {e}")

        # Reminder that messages have been trimmed
        trimmed_iterations = (len(current_messages) - len(trimmed_messages)) // 2
        if trimmed_iterations > 0:
            initial_messages.append(
                HumanMessage(content=trimming_message.format(trimmed_iterations=trimmed_iterations))
            )

        code = materialize_system(target_agentic_system, output_dir=None)
        code_message = (
            f"\n\n**You are now in Iteration {iteration}**\n--- Current Code of the TargetSystem ---\n```\n{code}\n```"
        )

        task_spec = state.get("task_spec")
        task_dir = state.get("task_dir")
        additional_docs = task_spec.load_additional_documentation(task_dir) if task_spec else ""
        system_prompt = build_meta_agent_prompt(function_signatures, additional_documentation=additional_docs)

        full_messages = (
            [SystemMessage(content=system_prompt)]
            + initial_messages
            + trimmed_messages
            + [HumanMessage(content=code_message)]
        )
        response = llm.invoke(full_messages)

        response_content = normalize_response_content(response.content)

        cleaned_content = re.sub(r"\[Iteration\s*\d+\]\s*\n*", "", response_content)
        iteration_info = f"[Iteration {iteration}]"
        response.content = f"{iteration_info}\n\n{cleaned_content}"

        updated_messages = messages + [response]

        new_state = {"messages": updated_messages}
        return new_state


def tool_execution(state: MetaState) -> dict[str, Any]:
    messages = state.get("messages", [])
    max_iterations = state.get("max_iterations", 30)
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)]) - 1

    response = messages[-1] if messages else AIMessage(content="")
    human_message, tool_results = execute_decorator_tool_calls(str(response.content), tools, code_related_tools, state)

    if not human_message:
        human_message = HumanMessage(content=decorator_reminder)

    is_new_test_result = any(tool_name == "TestSystem" for tool_name, _ in tool_results)
    search_window = ACTION_CUTOFF * 2 + 2
    start_index = max(0, len(messages) - search_window)

    if is_new_test_result:
        remove_old_test_results(start_index, messages)

    if max_iterations - iteration == max(round(max_iterations * 0.2), 4):
        human_message.content += f"\n\nYou have reached {iteration} iterations. Try to finish during the next iterations, run fully successful tests and end the design."

    human_message.content = f"[Iteration {iteration}][System]:\n\n" + str(human_message.content)

    # Remove duplicated test reminders
    parts = human_message.content.split(test_reminder)
    if len(parts) > 1:
        for i in range(start_index, len(messages)):
            msg = messages[i]
            if isinstance(msg, HumanMessage):
                msg.content = str(msg.content).replace(test_reminder, "")
    if len(parts) > 2:
        human_message.content = "".join(parts[:-1]) + test_reminder + parts[-1]

    messages.append(human_message)
    new_state = {
        "messages": messages,
        "system_passed": state.get("system_passed", False),
        "design_completed": state.get("design_completed", False),
        "test_metrics": state.get("test_metrics", {}),
        "candidates": state.get("candidates", []),
    }
    return new_state
