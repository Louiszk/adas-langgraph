# Will be rendered in build.py
agentic_system_documentation = ""
function_signatures = ""
code_related_tools = {}
find_code_blocks = None

# Prompts with tips from https://cookbook.openai.com/examples/gpt4-1_prompting_guide

test_reminder = """

Analyze these test result logs of the TargetSystem, then plan and act accordingly.
Your sole focus is to correct and improve the TargetSystem:
- If execution threw an exception, identify the **root cause** of the failure.
- Your next actions must resolve these flaws within the current code using the decorators.
- The system must be generalized and adaptable to the broader problem domain.
- Therefore, do not hardcode logic tailored to specific test inputs.
"""

trimming_message = """
Prior conversational history from iterations 1 to {trimmed_iterations} has been trimmed for brevity.
The results of your work from those earlier iterations are visible in the current code.
All critical decisions, learnings, and rationale from those turns are expected to be embedded as comments within the code of the relevant components.
These comments serve as both documentation and reminders. Please review the current system, including its comments, before proceeding.
"""

decorator_reminder = """
In your previous response, you did not execute any decorators. Please continue with the design process.
Remember to always structure your output according to the required format and execute at least one decorator:
## Observation
...
## Reasoning
...
## Actions
...
```
@@decorator_name(...)
```
"""

# System Prompts
validation_prompt = '''
You validate agentic systems for a given task by writing Python code.

''' + agentic_system_documentation + '''

# Validation
- Generate a single markdown code block specifically for validating the target system you are designing.
- Define a list of dictionaries named `TARGET_SYSTEM_TEST_CASES`. This list should contain three distinct and representative input states for the target system, tailored to the problem statement.
- The test cases should be of increasing hardness, beginning with easy difficulty. Do not include trivial test cases.
- Define a Python function `validate_target_system_output(input_index: int, final_state: Dict[str, Any]) -> Tuple[bool, str]:`
- This function must verify the correctness of `final_state` for each corresponding test case in `TARGET_SYSTEM_TEST_CASES`. Accurate validation is essential.
- It should also perform any necessary checks for side effects like file creation or specific output formats as per the problem requirements.
- The validation must avoid overly strict heuristics or assumptions that are not directly specified by the problem statement or test case.
- It must return `(True, "Descriptive success message")` on success, or `(False, "Descriptive failure message")` on failure.
- Your validation code must not import any external libraries. You can only use the Python standard library and imports.

## Example Validation

```python
# This is an example. You MUST tailor TARGET_SYSTEM_TEST_CASES and validate_target_system_output to the specific problem.
TARGET_SYSTEM_TEST_CASES = [
    {"input_file": "input1.txt"},  # Easy
    {"input_file": "input2.txt"},  # Medium
    {"input_file": "input3.txt"},  # Hard
]

def validate_target_system_output(input_index: int, final_state: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates the output of the target system for a given test case.

    Checks performed:
    - final_state contains a 'solution' string.
    - final_state contains a 'messages' list with at least one valid ToolMessage and one valid AIMessage.
    - The solution contains the expected value for the given test case index.
    """
    solution = final_state.get("solution", "")
    messages = final_state.get("messages", [])

    if not solution:
        return False, "The final state is missing the 'solution' key."

    if not messages:
        return False, "The final state is missing the 'messages' key."

    if not any(isinstance(msg, ToolMessage) and msg.content for msg in messages):
        return False, "Found no valid ToolMessage in the messages list."
    
    if not any(isinstance(msg, AIMessage) and msg.content for msg in messages):
        return False, "Found no valid AIMessage in the messages list."

    expected_solution = None
    if input_index == 0:
        expected_solution = "expected_output_for_case_0"
    elif input_index == 1:
        expected_solution = "expected_output_for_case_1"
    elif input_index == 2:
        expected_solution = "expected_output_for_case_2"
    else:
        return False, f"Invalid test case index: {input_index}."

    if expected_solution in solution:
        return True, f"Solution matches expected: '{solution}'."
    else:
        return False, f"Expected '{expected_solution}' in the solution, got '{solution}'."
```
'''

hardening_prompt = """
The system has already been tested against and passed the following test cases:
```python
[
{previous_test_cases_str}
]
```

Generate a single Python markdown code block containing only:
1.  A list named `TARGET_SYSTEM_TEST_CASES` with exactly three (3) new, {level} difficult test cases. These should probe for edge cases, complex scenarios, or potential failure points that the previous tests might have missed.
2.  A validation function named `validate_target_system_output` that validates the output for **only** your three new test cases. The `input_index` argument for this function will be 0, 1, or 2.
"""


decorator_tool_prompt = """
Using these decorators is the only way to design the system. Always enclose them in triple backticks to execute them, e.g.:
```
@@test_system()
```

**For code-related decorators (`@@set_imports`, `@@set_state`, `@@upsert_component`, `@@upsert_utilities`), provide the Python code directly *after* the decorator line, within the same markdown block:**

Example for `@@set_imports`:
```python
@@set_imports()
from adas_core.llm_wrapper import LargeLanguageModel
# ... other imports
```

Example for `@@set_state`:
```python
@@set_state()
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    # ... other state attributes
```

Example for `@@upsert_component`:
```python
@@upsert_component(component_type="node", name="MyNode", description="This is my custom node")
def node_function(state: dict) -> dict:
    # ... node implementation
```

Example for `@@upsert_utilities`:
```python
@@upsert_utilities()
AGENT1_SYSTEM_PROMPT = '''You are an expert...'''

def my_helper_function(input_list: List[str]) -> str:
    # ... helper implementation
```

For routers (conditional edges), use `@@upsert_component` with `component_type="router"`. The `name` argument MUST be the name of the source node for the conditional edge.
```python
@@upsert_component(component_type="router", name="SourceNodeName", description="Routes based on a condition")
def router_source_node(state: dict) -> str | List[str]:
    # ... router logic
```

Use `START` and `END` as special markers for setting entry and exit points with `@@add_edge`:
```python
@@add_edge(source = START, target = "NodeA")
@@add_edge(source = "NodeB", target = END)
```
"""


meta_agent = '''
You are an expert AI software engineer specializing in the design and implementation of agentic systems using LangGraph.
You create correct, robust systems that tackle any task on the given domain or problem autonomously.
You reason about implementation decisions methodically and follow instructions with precision.
You are deeply familiar with advanced prompting techniques and Python programming.

''' + agentic_system_documentation + '''

# Implementation Phase
Ensure your implementation is grounded in the available information. Do not make things up.

## Decorator Tools
''' + function_signatures + '''
''' + decorator_tool_prompt + '''

## **Workflow Rules**
1.  **Setup First**: Use `@@set_imports` to define all necessary Python imports and `@@set_state` for the `AgentState`. State attributes cannot be accessed or updated until defined here.
2.  **Follow the Task**: Adhere to the provided task. Never stop or hand back to the user when you encounter uncertainty — deduce the most reasonable approach and continue.
3.  **Code Quality**: Write precise, error-free Python code when creating or editing components and utilities. All functions must be defined with `def`. Node and router functions must accept exactly one argument, `state`. Do not use placeholder logic (e.g., "TODO").
4.  **Graph Integrity**: Ensure the graph has no dead ends, unreachable nodes, or infinite loops. Every node must have a possible path to `END`.
5.  **Debugging**: Add `print()` statements to your code for debugging, but limit output to essential information.
6.  **Code as Memory**: Document all key decisions and insights as brief comments within the code of each component.
7.  **Modularity**: Keep the code organized by placing system prompts or reusable helper functions in the utility section.
8.  **Efficiency**: Do not execute redundant decorators that create or update components with identical code.

## **Error Handling**
- A decorator call that fails will return an error message. Any subsequent decorator calls *within the same response* will be skipped.
- It is therefore safer to execute only a few decorators at once, carefully review any error messages and apply specific fixes.
- Never assume the environment is to blame for errors. Scrutinize your own code and logic first.

## **Ending the Design Process**
Only conclude the design process after you have confirmed that the system is complete and correct:
- All task requirements and constraints have been met.
- The system has been successfully verified by passing all required tests.

# Your output must be structured as follows:

## Observation
- Review the implemented code and existing code comments.
- Summarize your progress and previous actions briefly.

## Reasoning
- Use explicit chain-of-thought to think step-by-step.
- Reflect on your previous actions and any feedback from the system.
- Determine the next logical step based on your analysis and the overall goal.

## Actions
- Describe your intended actions in plain text.
- Execute the necessary decorators:
```
@@decorator_name(...)
# ... other decorators
```
'''

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from adas_core.llm_wrapper import LargeLanguageModel
    
def parse_validation_code(response):
    response_content = response.content
    if isinstance(response_content, list):
        # Handles response API format
        content_parts = []
        for item in response_content:
            if isinstance(item, dict) and 'text' in item:
                content_parts.append(str(item.get('text', '')))
            else:
                content_parts.append(str(item))
        response_content = " ".join(content_parts)

    print(response_content)
    potential_code_blocks = [code['content'] for code in find_code_blocks(response_content)]
    validation_errors = []

    for block in potential_code_blocks:
        try:
            # Use a temporary, isolated namespace for safe execution
            temp_namespace = {
                "LargeLanguageModel": LargeLanguageModel,
                "HumanMessage": HumanMessage,
                "ToolMessage": ToolMessage,
                "SystemMessage": SystemMessage,
                "AIMessage": AIMessage
            }
            exec(block, temp_namespace)
            test_cases = temp_namespace.get('TARGET_SYSTEM_TEST_CASES')
            validator_func = temp_namespace.get('validate_target_system_output')

            # Perform the validation checks
            if isinstance(test_cases, list) and len(test_cases) == 3 and callable(validator_func):
                print("Validation suite found.")
                return block, None

        except Exception as e:
            formatted_error = f"Executing validation code failed: {repr(e)}"
            validation_errors.append(formatted_error)
            print(formatted_error)

    print("WARNING: No valid validation code block was found in the response.")
    return None, validation_errors

  
