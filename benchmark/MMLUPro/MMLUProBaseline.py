from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from typing import List, TypedDict
from adas_core.llm_wrapper import LargeLanguageModel
import re


class AgentState(TypedDict):
    messages: List[AnyMessage]
    question: str
    options: List[str]
    solution: str


graph = StateGraph(AgentState)


def agent_node(state):
    llm = LargeLanguageModel(temperature=0)
    system_prompt = """
        You will solve multiple-choice questions in computer science.
        
        Each question comes with a list of options. Select the best option that answers the question.
        
        Write your final answer only as a single letter (A, B, C, ..., J) on the last line (e.g.: "The answer is X").
    """

    question = state["question"]
    options = state["options"]

    formatted_options = ""
    for i, option in enumerate(options):
        option_letter = chr(65 + i)
        formatted_options += f"{option_letter}: {option}\n"

    problem_text = f"{question}\n\n{formatted_options}"

    full_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=problem_text),
    ]
    response = llm.invoke(full_messages)

    response_text = response.content
    print(response_text)

    last_lines = "\n".join(response_text.split("\n")[-2:])
    matches = list(re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", last_lines))
    final_answer = matches[-1].group(1) if matches else "X"

    new_state = state.copy()
    new_state["solution"] = final_answer

    return new_state


graph.add_node("MMLUProBaseline", agent_node)

graph.add_edge(START, "MMLUProBaseline")
graph.add_edge("MMLUProBaseline", END)

workflow = graph.compile()
