from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Any, TypedDict
from adas_core.llm_wrapper import LargeLanguageModel

class AgentState(TypedDict):
    messages: List[Any]
    claim: str
    prediction: str

graph = StateGraph(AgentState)

def base_node(state):
    llm = LargeLanguageModel(temperature=0)
    system_prompt = """
        You will evaluate factual claims.
        
        For your analysis, classify the given claim into one of these categories:
        - SUPPORTS: The claim is supported by factual evidence
        - REFUTES: The claim contradicts factual evidence
        - NOT ENOUGH INFO: There is insufficient evidence to determine if the claim is supported or refuted
        
        Write your final prediction in the last line using exactly one of these three labels: `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
    """

    full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["claim"])]
    response = llm.invoke(full_messages)
    
    response_text = response.content
    print(response_text)
    
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    final_answer = "NO ANSWER"
    
    last_lines = "\n".join(response_text.split("\n")[-2:])
    
    for label in labels:
        if label in last_lines:
            final_answer = label
            break
    
    state["prediction"] = final_answer
    
    return state

graph.add_node("FEVERBaseline", base_node)

graph.add_edge(START, "FEVERBaseline")
graph.add_edge("FEVERBaseline", END)

workflow = graph.compile()