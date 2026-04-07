from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, AnyMessage
from adas_core.llm_wrapper import LargeLanguageModel
from langchain_core.tools import tool
from typing import List, TypedDict

class AgentState(TypedDict):
    messages: List[AnyMessage]
    problem: str
    solution: str
    code_execution_error: bool

@tool
def python_interpreter(code: str) -> str:
    """Executes Python code. You can import built-in libraries.
    
    Args:
        code: Valid Python code to solve the problem.
              IMPORTANT: The final numerical answer must be assigned to a variable named `result`.
    """
    try:
        namespace = {}
        exec(code, namespace)
        
        result = namespace.get("result")
        if result is None:
            return "Error: Code executed, but variable 'result' was not defined."
        
        return str(result)
        
    except Exception as e:
        return f"Execution Error: {str(e)}"


def agent_node(state):
    llm = LargeLanguageModel(temperature=0)
    llm.bind_tools([python_interpreter], parallel_tool_calls=False)
    
    system_prompt = """
    You are an expert mathematician and python programmer.
    You utilize the python_interpreter tool to calculate the answer to math word problems.
    You always assign the final answer to a variable named `result` in your code.
    If you observe an error, you reason about the cause and fix the code flawlessly.
    """
    
    system_message = SystemMessage(content=system_prompt)
    messages: List[AnyMessage] = state.get("messages", [])
    
    if len(messages) == 0:
        messages.append(HumanMessage(content=state["problem"]))
        
    response = llm.invoke([system_message] + messages)
    messages.append(response)
    
    return {"messages": messages}


def tool_node(state):
    messages = state["messages"]
    last_message: AIMessage = messages[-1]
    tool_output = ""
    
    for tool_call in last_message.tool_calls:
        tool_output = python_interpreter.invoke(tool_call['args'])
        messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
        break

    code_execution_error = "Error" in tool_output
        
    return {"messages": messages, "code_execution_error": code_execution_error, "solution": tool_output}

def execution_router(state):
    if state.get("code_execution_error", False):
        if len([msg for msg in state["messages"] if isinstance(msg, AIMessage)]) <= 2:
            return "AgentNode"
    
    return END

graph = StateGraph(AgentState)

graph.add_node("AgentNode", agent_node)
graph.add_node("ToolNode", tool_node)

graph.add_edge(START, "AgentNode")
graph.add_edge("AgentNode", "ToolNode")
graph.add_conditional_edges("ToolNode", execution_router)

workflow = graph.compile()