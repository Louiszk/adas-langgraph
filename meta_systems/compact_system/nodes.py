from typing import Dict, Any
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    trim_messages,
)
from adas_core.decorator_logic import execute_decorator_tool_calls
from adas_core.helpers import remove_old_test_results
from adas_core.llm_wrapper import LargeLanguageModel
from adas_core.materialize import materialize_system
import re

from meta_systems.compact_system.configurations import (
    validation_wrapper,
    validation_model,
    meta_agent_wrapper,
    meta_agent_model,
    meta_agent_reasoning_effort,
    ACTION_CUTOFF,
)
from meta_systems.compact_system.utilities import (
    validation_prompt,
    hardening_prompt,
    parse_validation_code,
    trimming_message,
    meta_agent,
    code_related_tools,
    decorator_reminder,
    test_reminder,
)
from meta_systems.compact_system.tools import tools


def formatting_function(state: Dict[str, Any]) -> Dict[str, Any]:
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
        "hardening_steps": 0,
        "validation_code_snippets": [],
    }
    return new_state


def validation_function(state: Dict[str, Any]) -> Dict[str, Any]:
    initial_task = HumanMessage(content=str(state.get("initial_task", "")))
    steps = state.get("hardening_steps", 0)
    snippets = state.get("validation_code_snippets", [])

    reasoning_effort = "medium" if steps <= 1 else "high"
    level = "more" if steps <= 1 else "maximally"

    llm = LargeLanguageModel(
        temperature=0.4,
        reasoning_effort=reasoning_effort,
        wrapper=validation_wrapper,
        model_name=validation_model,
        name="Validation",
        is_meta=True,
    )

    if steps == 0:
        # First-time generation of validation code
        print("--- Generating initial validation suite ---")
        prompt_messages = [SystemMessage(content=validation_prompt), initial_task]
    else:
        # Hardening existing validation code
        print(f"--- System passed. Generating more difficult test cases (Iteration {steps}) ---")

        # Aggregate previous test cases for context
        previous_test_cases_str = ""
        temp_namespace = {
            "LargeLanguageModel": LargeLanguageModel,
            "HumanMessage": HumanMessage,
            "ToolMessage": ToolMessage,
            "SystemMessage": SystemMessage,
            "AIMessage": AIMessage,
        }
        for snippet in snippets:
            try:
                exec(snippet, temp_namespace)
                cases = temp_namespace.get("TARGET_SYSTEM_TEST_CASES", [])
                previous_test_cases_str += "\n".join([f"    {case}," for case in cases])
            except Exception:
                pass

        formatted_hardening_prompt = hardening_prompt.format(
            previous_test_cases_str=previous_test_cases_str, level=level
        )
        prompt_messages = [
            SystemMessage(content=validation_prompt),
            initial_task,
            HumanMessage(content=formatted_hardening_prompt),
        ]

    validation_error = None
    new_snippet = None
    for _ in range(3):
        response = llm.invoke(messages_input=prompt_messages, is_meta=True)
        new_snippet, validation_errors_list = parse_validation_code(response)
        if new_snippet:
            break
        validation_error = (
            "\n".join(validation_errors_list) if validation_errors_list else "No valid markdown block found."
        )
        failed_attempt_message = validation_error + "\nPlease try again."
        prompt_messages.extend([response, HumanMessage(content=failed_attempt_message)])

    if not new_snippet:
        if not snippets:
            raise ValueError("Unable to generate initial Validation Code.")
        else:
            print("Failed to generate a valid hardened test suite.")
            return {"hardening_passed": None}

    updated_snippets = snippets + [new_snippet]

    return {
        "validation_code_snippets": updated_snippets,
        "hardening_steps": steps + 1,
        "hardening_passed": None,
    }


def initial_test_runner_function(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("optimize"):
        return {"hardening_passed": None}

    steps = state.get("hardening_steps", 0)
    initial_test_passes = state.get("initial_test_passes", 0)
    print(f"--- Hardening Loop: (Iteration {steps}) ---")
    test_system_tool = tools.get("TestSystem")
    if not test_system_tool:
        raise ValueError("TestSystem tool not found.")

    # The test_system tool now internally handles the list of snippets
    test_result_str = test_system_tool.invoke({"state": state})  # type: ignore
    pattern = r"The system passed (\d+)/\d+ tests\."
    match = re.search(pattern, test_result_str)
    new_initial_test_passes = int(match.group(1)) if match else 0
    initial_test_passes = max(new_initial_test_passes, initial_test_passes)

    validator_split = test_result_str.split("<ValidatorResult>")
    validator_result = validator_split[-1] if len(validator_split) > 1 else ""

    if "Overall: PASSED" in validator_result:
        return {"hardening_passed": True, "initial_test_passes": initial_test_passes}
    print("--- System failed. Handing off hardened test suite to meta-agent. ---")

    verbose_test_results_content = (
        "--- Initial Test Results ---\n"
        + test_result_str
        + "\nThese tests were run right at the start of the design process (Iteration 0), before you made any changes to the system."
        + f"\nImprove upon this baseline by achieving at least {max(2, initial_test_passes + 1)} passing tests."
        + "\nCrucially, the system must be generalized and adaptable to the broader problem domain. Do not hardcode logic tailored only to these specific test inputs."
    )

    pattern_to_remove = r"<FinalState>.*?</FinalState>|<STDOUT\+STDERR>.*?</STDOUT\+STDERR>"
    cleaned_test_results_content = re.sub(pattern_to_remove, "", verbose_test_results_content, flags=re.DOTALL)

    # Return both versions to update the state
    return {
        "verbose_initial_test_results": HumanMessage(content=verbose_test_results_content),
        "initial_test_results": HumanMessage(content=cleaned_test_results_content),
        "initial_test_passes": initial_test_passes,
        "hardening_passed": False,
    }


def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = LargeLanguageModel(
        wrapper=meta_agent_wrapper,
        model_name=meta_agent_model,
        reasoning_effort=meta_agent_reasoning_effort,
        name="MetaAgent",
        is_meta=True,
    )

    context_length = ACTION_CUTOFF * 2
    messages = state.get("messages", [])
    target_agentic_system = state["target_agentic_system"]

    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    current_messages = messages[1:]
    initial_messages = [state["designer_task"]]
    if state.get("initial_test_results"):
        if iteration > 0:
            initial_messages.insert(1, state["initial_test_results"])
        else:
            initial_messages.insert(1, state["verbose_initial_test_results"])

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
        print(f"Error during message trimming: {e}")

    # Reminder that messages have been trimmed
    trimmed_iterations = (len(current_messages) - len(trimmed_messages)) // 2
    if trimmed_iterations > 0:
        initial_messages.append(HumanMessage(content=trimming_message.format(trimmed_iterations=trimmed_iterations)))

    code = materialize_system(target_agentic_system, output_dir=None)
    code_message = (
        f"\n\n**You are now in Iteration {iteration}**\n--- Current Code of the TargetSystem ---\n```\n{code}\n```"
    )

    full_messages = (
        [SystemMessage(content=meta_agent)] + initial_messages + trimmed_messages + [HumanMessage(content=code_message)]
    )
    response = llm.invoke(messages_input=full_messages, is_meta=True)

    response_content = response.content
    if isinstance(response_content, list):
        # Handles response API format
        content_parts = []
        for item in response_content:
            if isinstance(item, dict) and "text" in item:
                content_parts.append(str(item.get("text", "")))
            else:
                content_parts.append(str(item))
        response_content = " ".join(content_parts)

    cleaned_content = re.sub(r"\[Iteration\s*\d+\]\s*\n*", "", response_content)
    iteration_info = f"[Iteration {iteration}]"
    response.content = f"{iteration_info}\n\n{cleaned_content}"

    updated_messages = messages + [response]

    new_state = {"messages": updated_messages}
    return new_state


# Tool node
def tool_execution(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])
    max_iterations = state.get("max_iterations", 30)
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)]) - 1

    response: AIMessage = messages[-1]
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
    }
    return new_state
