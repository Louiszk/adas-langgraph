agentic_system_documentation = """
# LangGraph Essentials for Agentic System Design

## LangGraph Agentic System Architecture

LangGraph is used to build stateful, multi-actor applications, such as agentic systems.
An agentic system in LangGraph consists of a directed graph with nodes and edges, where:
- **Nodes**: These are Python functions that process and may modify the system's shared state.
- **Edges**: These define the sequence of execution, directing the flow of data and control between nodes.
- **Tools**: Standalone functions that perform specific tasks. Tools are not nodes themselves, but can be invoked from within nodes.
- The system has an entry point marker (`START`) and an exit point marker (`END`). `END` signifies the termination of a specific execution path.

### State (`AgentState`)

The state is a central object that holds all the information the system needs while it runs. Nodes in the graph read from and write to this state.

```python
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    final_answer: str  # Other necessary state fields for the task
```

#### State Management

- The `messages` attribute is included by default for robust conversational history.
- The `add_messages` function is a built-in LangGraph reducer that correctly appends new messages to the existing list, instead of overwriting it.
- You can define custom state attributes with their Python type annotations.
- All state attributes must be defined when the system's AgentState is declared.

### Tools

Tools are standalone functions designed to perform specific actions.
Tools are NOT nodes in the graph. They can be invoked in two primary ways:

**By AI Agents (LLM-driven)**:
When a node uses a LargeLanguageModel, the LLM can decide to use available tools.
For an LLM to understand and use a tool effectively, the tool must have clear type annotations and a comprehensive docstring.

```python
# 'tools' is a dictionary of all available tool objects
relevant_tools = [tools["Tool1"], tools["Tool2"]] 
llm.bind_tools(relevant_tools)
response = llm.invoke(some_messages)
tool_messages, tool_results = execute_tool_calls(response, tools)
```

**Programmatically within Nodes**:
You can also directly invoke a tool's functionality within any node.

```python
# .invoke() expects a dictionary of the tool's keyword arguments
result_from_tool1 = tools["Tool1"].invoke({"kwarg1": "some_value"})
```

### Nodes

Nodes are Python functions that receive the current `AgentState` as their only input and return a dictionary containing the state keys they wish to update.
LangGraph merges this dictionary into the overall state. Any valid Python code can be executed within a node.

**Example of a node that uses an LLM**:
This pattern is common for creating agents that can reason, plan, or interact with tools.

```python
def agent_node(state: AgentState) -> dict:
    # Use the LargeLanguageModel class, a wrapper around ChatOpenAI
    llm = LargeLanguageModel(temperature=0.4)
    llm.bind_tools([tools["SomeTool"]])
    
    messages = state.get("messages", [])
    full_messages = [SystemMessage(content="Your instructions...")] + messages
    
    response = llm.invoke(full_messages)
    tool_messages, tool_results = execute_tool_calls(response, tools)
    
    # Append the agent's response and the tool messages to the state
    return {"messages": [response] + tool_messages}
```

### Edges

Edges define the control flow and potential for parallelism in the graph.

#### Standard Edges

These create direct, unconditional connections.
- **Sequential Flow**: A single edge `graph.add_edge("node_A", "node_B")` ensures `node_B` runs after `node_A`.
- **Parallel Execution**: You can define multiple standard edges from the same source node.
  This will create multiple execution paths that run concurrently in "supersteps."
  A superstep is a processing phase where all active nodes at that level execute in parallel. The graph waits for all nodes in the current superstep to complete before advancing to the next superstep.
  If one parallel path reaches the END marker, any other active parallel paths will continue their execution until they also reach END.
- **Synchronization**: You can also synchronize parallel paths by directing them to reach the same node within the same superstep.

#### Conditional Edges (Routers)

These allow for dynamic routing. A "router" is a function that takes the current `AgentState` and returns the name(s) of the next node(s) to execute, or `END` to terminate the path.
- **Single Path**: Returning a single string name directs the flow to that node.
- **Parallel Paths**: Returning a `List[str]` of node names will trigger parallel execution of all nodes in the list.

```python
def router_function(state) -> str | List[str]:
    last_message = state.get("messages", [None])[-1]
    if not last_message or "error" in last_message.content:
        return "ErrorHandlerNode"
    elif "complete" in last_message.content:
        return END
    else:
        return ["ProcessingAgent1", "ProcessingAgent2"]
```

It is also possible for a source node to have both standard edges and a conditional edge.
In this case, LangGraph will execute the target(s) of the standard edge(s) concurrently with the target(s) returned by the conditional router.

### State Reducers (State Management for Parallelism)

By default, LangGraph updates state by merging dictionaries. This means a node's output for a key replaces the existing value.
During parallel execution, if two nodes try to update the same key in the same superstep, this will cause an error.

A **reducer** function solves this by defining how to combine the old and new values.
- **Examples**: The built-in `add_messages` reducer appends new messages instead of replacing the list. `operator.add` can be used as a reducer to combine values by addition instead of replacement.
- **Custom Reducers**: You can write your own reducer functions (e.g., to add numbers, merge dictionaries) and apply them to a state key using `typing.Annotated`.
  This is the primary mechanism for safely managing shared state during parallel execution.

```python
def sum_values(old_value: int, new_value: int) -> int:
    return (old_value or 0) + (new_value or 0)

class AgentState(TypedDict):
    # Any update to 'counter' will now be added to the existing value.
    counter: Annotated[int, sum_values]
```

## Message Types and Context Window

### Message Types

- `SystemMessage`: Provides initial instructions or context to the AI model.
- `HumanMessage`: Represents user input and can be used to provide specialized instructions or feedback.
- `AIMessage`: Represents a response from the AI model. It can contain `content` (text) and/or `tool_calls`.
- `ToolMessage`: Contains the result of a tool execution. These are returned by `execute_tool_calls` and should not be constructed manually.

The final message in an invoke must always be a HumanMessage or ToolMessage.
To invoke an LLM with a list of messages, every AIMessage that contains `tool_calls` must be followed by its corresponding ToolMessage.
You can ensure this by always appending the tool messages generated by `execute_tool_calls` to your state's messages.

### Context Window Management

Passing the entire history of messages to each LLM call can be inefficient. The `trim_messages` utility from LangChain can be used to manage the context window.

```python
from langchain_core.messages import trim_messages

# Example: Keep the last 16 messages
trimmed_messages_by_count = trim_messages(
    current_messages,
    max_tokens=16,
    strategy="last",
    token_counter=len # len counts messages
)

# Example: Trim to a maximum of 8000 tokens, keeping last messages.
trimmed_messages_by_token = trim_messages(
    current_messages,
    max_tokens=8000,
    strategy="last",
    token_counter=LargeLanguageModel.token_counter,
)
```

## ADAS Core Module (`adas_core.llm_wrapper`)

### `LargeLanguageModel` Class

This is a standardized wrapper for interacting with LLMs.
- Initialization: `llm = LargeLanguageModel(temperature: float = 0.2)`. The underlying model is managed by the module.
- Tool Binding: `llm.bind_tools(tool_objects: List[Any])`. It informs the LLM about the tools available to call.
- Invocation: `response = llm.invoke(messages_input: List[Any])`. It sends a request to the LLM and gets an `AIMessage` in response, which may contain tool calls.

### `execute_tool_calls` Function

This helper function processes tool calls generated by an LLM: `execute_tool_calls(response: AIMessage, available_tools: Dict)`
It returns a list of `ToolMessage` objects for the conversation history and a dictionary mapping tool names to their results.
"""
