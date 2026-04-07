from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from adas_core.llm_wrapper import LargeLanguageModel
from langchain_core.tools import tool
from typing import List, Any, TypedDict
import re

class AgentState(TypedDict):
    messages: List[Any]
    problem: str
    solution: str

graph = StateGraph(AgentState)

def base_node(state):
    llm = LargeLanguageModel(temperature=0)
    
    system_prompt = "You will solve math word problems."
    system_prompt += "\nWrite your final numerical answer on the last line, with at least three decimal places of precision."

    full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["problem"])]
    response = llm.invoke(full_messages)
    response_text = response.content
    print(response_text)
    
    clean_text = response_text.replace(',', '')
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", clean_text)
    final_answer = numbers[-1] if numbers else "No answer found"
    
    new_state = state.copy()
    new_state["solution"] = final_answer
    
    return new_state

graph.add_node("GSMHardBaseline", base_node)

graph.add_edge(START, "GSMHardBaseline")
graph.add_edge("GSMHardBaseline", END)

workflow = graph.compile()