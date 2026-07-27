agentic_system_documentation = """
# LangGraph + ADAS Core Reference

## Core Invariants & State Rules

1. **AgentState Definition**:
   - `AgentState(TypedDict)` must be defined first and include at least:
     `messages: Annotated[List[AnyMessage], add_messages]`
   - Any extra custom state keys must be declared in `AgentState` before use.

2. **Node and Router Signatures**:
   - **Strict Rule**: EVERY node and router function MUST accept **exactly one** argument named `state`.
     - Node Signature: `def my_node(state: AgentState) -> dict:`
     - Router Signature: `def my_router(state: AgentState) -> str | List[str]:`
   - Nodes return a dictionary containing state keys to update (e.g., `{"final_answer": "42"}`).
   - Routers return the name of the next node(s) to run, or `END`. Returning a `List[str]` triggers parallel branches.

3. **Graph Endpoint Markers**:
   - Use `START` and `END` from `langgraph.graph` as workflow entry/exit markers.

---

## ADAS Core Module (`adas_core.llm_wrapper`)

### `LargeLanguageModel` Class
A standardized wrapper for interacting with LLMs.
- **Initialization**: `llm = LargeLanguageModel()`
- **Tool Binding**: `llm.bind_tools(tool_objects: List[Any]) -> LargeLanguageModel`
  Informs the LLM about available tool functions.
- **Invocation**: `response = llm.invoke(messages_input: List[Any]) -> AIMessage`
  Sends requests to the model and returns an `AIMessage` (which may contain `tool_calls`).
- **Token Counter**: `LargeLanguageModel.token_counter`
  Tokenizer property used for exact token count calculations in message trimming.

### `execute_tool_calls` Function
- **Signature**: `execute_tool_calls(response: AIMessage, available_tools: Dict[str, Any]) -> Tuple[List[ToolMessage], Dict[str, Any]]`
- **Behavior**: Processes `tool_calls` inside an `AIMessage`. Returns:
  1. `tool_messages`: A `List[ToolMessage]` containing execution outputs or error messages to append to history.
  2. `tool_results`: A `Dict[str, Any]` mapping executed tool names to their raw return values.

### Standard Agent Node Pattern
```python
from adas_core.llm_wrapper import LargeLanguageModel, execute_tool_calls
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def agent_node(state: AgentState) -> dict:
    llm = LargeLanguageModel()
    
    # Bind available tools from the global `tools` dict if needed
    if "MyTool" in tools:
        llm.bind_tools([tools["MyTool"]])
    
    messages = state.get("messages", [])
    full_messages = [SystemMessage(content="Instructions...")] + messages
    
    response: AIMessage = llm.invoke(full_messages)
    tool_messages, tool_results = execute_tool_calls(response, tools)
    
    # Return updated state dictionary (append AIMessage and resulting ToolMessages)
    return {"messages": [response] + tool_messages}
```

### Direct Tool Invocation

Tools in the `tools` dictionary can also be invoked directly inside nodes:
```python
# Pass keyword arguments as a dictionary to .invoke()
result = tools["SearchTool"].invoke({"query": "LangGraph documentation"})
```

---

## Parallel Execution & State Reducers

- Default state updates replace existing values.
- If multiple parallel nodes update the same state key in a single superstep, you MUST declare a reducer in `AgentState` using `Annotated`:
```python
import operator
from typing import Annotated, TypedDict

def append_results(old: list, new: list) -> list:
    return (old or []) + (new or [])

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    logs: Annotated[List[str], append_results] # Custom reducer
    score: Annotated[int, operator.add]        # Built-in reducer
```

---

## Message Types & Context History Trimming

- **Message Types**: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` (from `langchain_core.messages`).
- **Tool Message Rule**: Every `AIMessage` containing `tool_calls` MUST be followed immediately by its corresponding `ToolMessage` objects before the next LLM call.
- **Context Window Trimming**: Use `trim_messages` from `langchain_core.messages` to prevent token overflow on long trajectories:

```python
from langchain_core.messages import trim_messages
from adas_core.llm_wrapper import LargeLanguageModel

# Example: Keep the last 16 messages
trimmed_messages_by_count = trim_messages(
    current_messages,
    max_tokens=16,
    strategy="last",
    token_counter=len # len counts messages
)

# Example: Trim to a maximum token budget (e.g., 8000 tokens) using the LLM token counter
trimmed_messages = trim_messages(
    current_messages,
    max_tokens=8000,
    strategy="last",
    token_counter=LargeLanguageModel.token_counter,
)
```
"""
