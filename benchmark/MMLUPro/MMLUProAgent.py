from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, TypedDict, Dict, Annotated
from adas_core.llm_wrapper import LargeLanguageModel
import re
import operator
import random

voter_prompt = """
You are a Computer Science expert solving multiple-choice questions.
You have been given a SUBSET of the available options. The correct answer may or may not be in your list.

1. Analyze the question and available options.
2. Think step-by-step separately for each option.
3. If you are confident the correct answer is within your list, select it.
4. If none of your options seem correct, you must conclude that the answer is missing.

Provide your final answer only as a single letter strictly on the last line:
- Example if you find the answer: "The final answer is X"
- Example if no option fits: "The final answer is None"
"""

context_judge_prompt = """
You are a Computer Science expert acting as a final judge.
Several experts have analyzed subsets of the options and found potential answers.
However, they disagree. You must resolve the conflict.

1. Review the Question and ALL Options.
2. Review the Reasoning from the specific experts.
3. Think step-by-step to decide on the single correct option.
4. Provide the single correct uppercase letter on the last line.
Example: The correct answer is X
"""

blind_judge_prompt = """
You are a Computer Science expert solving multiple-choice questions.

1. Analyze the Question and ALL Options.
2. Think step-by-step separately for each option.
3. Provide the single correct uppercase letter on the last line.
Example: The correct answer is X
"""


class AgentState(TypedDict):
    question: str
    options: List[str]
    solution: str
    subsets: List[List[int]]
    voter_outputs: Annotated[List[Dict[str, str]], operator.add]


graph = StateGraph(AgentState)


def create_subsets_node(state):
    options = state["options"]
    num_options = len(options)
    indices = list(range(num_options))

    run1_indices = indices.copy()
    random.shuffle(run1_indices)
    subset_1 = run1_indices[:5]
    subset_2 = run1_indices[5:]

    return {"subsets": [subset_1, subset_2, indices, indices]}


def run_voter(state: AgentState, subset_index: int, voter_name: str) -> Dict[str, str]:
    options_list = state["options"]
    target_indices = state["subsets"][subset_index]
    all_letters = [chr(65 + i) for i in range(len(options_list))]

    formatted_options = ""
    for idx in target_indices:
        if idx < len(options_list) and options_list[idx]:
            formatted_options += f"{all_letters[idx]}: {options_list[idx]}\n"

    problem_text = f"--- Question ---\n{state['question']}\n\n--- Option Subset ---\n{formatted_options}\n\n"

    llm = LargeLanguageModel(temperature=0)
    response = llm.invoke([SystemMessage(content=voter_prompt), HumanMessage(content=problem_text)])
    response_text = response.content

    matches = re.findall(r"final answer is (None|[A-J])", response_text, re.IGNORECASE)
    if matches:
        decision = matches[-1].capitalize()
    else:
        last_lines = "\n".join(response_text.split("\n")[-2:])
        letter_matches = list(re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", last_lines))
        decision = letter_matches[-1].group(1) if letter_matches else "None"

    return {"voter_name": voter_name, "decision": decision, "reasoning": response_text}


def voter_1(state):
    return {"voter_outputs": [run_voter(state, 0, "Expert 1")]}


def voter_2(state):
    return {"voter_outputs": [run_voter(state, 1, "Expert 2")]}


def voter_3(state):
    return {"voter_outputs": [run_voter(state, 2, "Expert 3")]}


def voter_4(state):
    return {"voter_outputs": [run_voter(state, 3, "Expert 4")]}


def finalize_node(state):
    votes = state["voter_outputs"]
    votes.sort(key=lambda x: x["voter_name"])

    print(f"Decisions: {[v['voter_name'] + ': ' + v['decision'] for v in votes]}")

    decisions = [vote["decision"] for vote in votes]
    valid_decisions = [d for d in decisions if d != "None"]

    final_answer = "A"

    if not valid_decisions:
        options_list = state["options"]
        all_letters = [chr(65 + i) for i in range(len(options_list))]
        formatted_all = "".join([f"{all_letters[i]}: {opt}\n" for i, opt in enumerate(options_list) if opt])

        judge_text = f"--- Question ---\n{state['question']}\n\n--- All Options ---\n{formatted_all}"

        judge_llm = LargeLanguageModel(temperature=0)
        response = judge_llm.invoke(
            [
                SystemMessage(content=blind_judge_prompt),
                HumanMessage(content=judge_text),
            ]
        )

        last_lines = "\n".join(response.content.split("\n")[-2:])
        matches = list(re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", last_lines))
        if matches:
            final_answer = matches[-1].group(1)

    else:
        double_one = decisions[0] != "None" and decisions[0] in decisions[2:]
        double_two = decisions[1] != "None" and decisions[1] in decisions[2:]
        double_full_experts = decisions[2] != "None" and decisions[2] == decisions[3]
        if len(valid_decisions) == 1:
            final_answer = valid_decisions[0]
        elif double_one and not double_two:
            final_answer = decisions[0]
        elif double_two and not double_one:
            final_answer = decisions[1]
        elif double_full_experts:
            final_answer = decisions[2]
        else:
            options_list = state["options"]
            all_letters = [chr(65 + i) for i in range(len(options_list))]
            formatted_all = "".join([f"{all_letters[i]}: {opt}\n" for i, opt in enumerate(options_list) if opt])

            reasoning_context = ""
            for v in votes:
                if v["decision"] != "None":
                    reasoning_context += f"--- {v['voter_name']} Reasoning ---\n{v['reasoning']}"

            judge_text = (
                f"--- Question ---\n{state['question']}\n\n--- All Options ---\n{formatted_all}\n\n{reasoning_context}"
            )

            judge_llm = LargeLanguageModel(temperature=0)
            response = judge_llm.invoke(
                [
                    SystemMessage(content=context_judge_prompt),
                    HumanMessage(content=judge_text),
                ]
            )
            print(response.content)

            last_lines = "\n".join(response.content.split("\n")[-2:])
            matches = list(re.finditer(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", last_lines))
            if matches:
                final_answer = matches[-1].group(1)
            else:
                final_answer = valid_decisions[0]

    return {"solution": final_answer}


graph.add_node("create_subsets", create_subsets_node)
graph.add_node("voter_1", voter_1)
graph.add_node("voter_2", voter_2)
graph.add_node("voter_3", voter_3)
graph.add_node("voter_4", voter_4)
graph.add_node("finalize", finalize_node)

graph.add_edge(START, "create_subsets")
graph.add_edge("create_subsets", "voter_1")
graph.add_edge("create_subsets", "voter_2")
graph.add_edge("create_subsets", "voter_3")
graph.add_edge("create_subsets", "voter_4")

graph.add_edge("voter_1", "finalize")
graph.add_edge("voter_2", "finalize")
graph.add_edge("voter_3", "finalize")
graph.add_edge("voter_4", "finalize")

graph.add_edge("finalize", END)

workflow = graph.compile()
