from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    AnyMessage,
)
from typing import List, TypedDict, Optional
from adas_core.llm_wrapper import LargeLanguageModel
from langchain_core.tools import tool
import wikipedia  # type: ignore


class AgentState(TypedDict):
    messages: List[AnyMessage]
    claim: str
    prediction: str
    available_pages: Optional[str]
    evidence: Optional[str]


@tool
def wiki_search_tool(query: str) -> str:
    """Searches Wikipedia for relevant pages.
    Args:
        query: A search query to find titles of pages.
    """
    try:
        results = wikipedia.search(query)
        if not results:
            return "No Wikipedia pages found for this query."
        quotes_results = [f'"{result}"' for result in results]
        return f"Found the following pages: {', '.join(quotes_results)}"
    except Exception as e:
        return f"Search Error: {str(e)}"


@tool
def wiki_content_tool(page_title: str) -> str:
    """Gets the summary content of a specific Wikipedia page.
    Args:
        page_title: The exact title of the page to read.
    """
    try:
        return wikipedia.summary(page_title, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation Error: '{page_title}' refers to multiple pages: {e.options}. Please pick a more specific title."
    except wikipedia.PageError:
        error_message = f"Page Error: The page '{page_title}' does not exist."
        try:
            similar_results = wikipedia.search(page_title)
            quotes_results = [f'"{result}"' for result in similar_results[:4]]
            return error_message + f" Did you mean any of these pages?: {quotes_results}"
        except Exception:
            return error_message + " Please check the spelling from the search results."
    except Exception as e:
        return f"Error fetching content: {str(e)}"


def agent_node(state):
    messages = list(state.get("messages", []))
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    evidence = state.get("evidence")
    available_pages = state.get("available_pages")

    llm = LargeLanguageModel(temperature=0)
    if iteration <= 3:
        if not available_pages:
            llm.bind_tools([wiki_search_tool])
        else:
            llm.bind_tools([wiki_search_tool, wiki_content_tool])

    system_prompt = """
    You are an expert at evaluating factual claims using Wikipedia.
    
    Your task is to classify the given claim into one of these categories:
    - `SUPPORTS`: The full claim is strictly supported by factual evidence.
    - `REFUTES`: The full claim strictly contradicts factual evidence.
    - `NOT ENOUGH INFO`: There is insufficient evidence to determine if the entire claim is supported or refuted.

    Never assume or deduce evidence. If it is not clearly stated in the evidence, it is always `NOT ENOUGH INFO`.

    1. Search Wikipedia for relevant pages.
    2. Fetch the content of the most relevant pages.
    3. Restate the claim and compare the evidence to the claim.
    4. Think step-by-step to write the correct classification on the last line using exactly one of the categories. Example: 'The final classification is `REFUTES`'
    """

    if not messages:
        messages.append(HumanMessage(content=f"Claim: {state['claim']}"))

    if isinstance(messages[-1], AIMessage):
        messages.append(
            HumanMessage(
                content=(
                    "In your last response, you did not call any tools and did not provide a final classification. "
                    "Avoid repeating this inaction."
                )
            )
        )

    if iteration > 3:
        messages.append(
            HumanMessage(content="You have reached the maximum number of turns. You must decide on a category now.")
        )

    full_messages = [SystemMessage(content=system_prompt)] + messages
    response: AIMessage = llm.invoke(full_messages)

    final_answer = "NO ANSWER"

    if evidence or iteration > 3:
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        last_lines = "\n".join(str(response.content).split("\n")[-2:])

        for label in labels:
            if label in last_lines:
                final_answer = label
                break

    print(response.content)
    if hasattr(response, "tool_calls"):
        print(response.tool_calls)

    messages.append(response)
    return {"messages": messages, "prediction": final_answer}


def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]

    new_available_pages = state.get("available_pages") or ""
    new_evidence = state.get("evidence") or ""

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        output = ""

        if tool_name == "wiki_search_tool":
            output = wiki_search_tool.invoke(tool_args)
            new_available_pages += output

        elif tool_name == "wiki_content_tool":
            output = wiki_content_tool.invoke(tool_args)
            new_evidence += output

        print("Tooloutput", output)

        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))

    return {
        "messages": messages,
        "available_pages": new_available_pages,
        "evidence": new_evidence,
    }


def router(state):
    messages = state["messages"]
    iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
    last_message = messages[-1]
    prediction = state.get("prediction")

    if last_message.tool_calls:
        return "ToolNode"

    if (prediction and prediction != "NO ANSWER") or iteration > 4:
        return END

    return "AgentNode"


graph = StateGraph(AgentState)

graph.add_node("AgentNode", agent_node)
graph.add_node("ToolNode", tool_node)

graph.add_edge(START, "AgentNode")
graph.add_conditional_edges("AgentNode", router)
graph.add_edge("ToolNode", "AgentNode")

workflow = graph.compile()
