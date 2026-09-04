from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

from adas_core.virtual_agentic_system import VirtualAgenticSystem
from adas_core.task_spec import TaskSpec


class MetaState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    target_agentic_system: VirtualAgenticSystem
    verbose_initial_test_results: HumanMessage | None
    initial_test_results: HumanMessage | None
    initial_test_passes: int
    validation_code_snippets: list[str]
    system_passed: bool
    design_completed: bool
    initial_task: str
    designer_task: HumanMessage
    remaining_steps: RemainingSteps
    max_iterations: int
    optimize: bool
    hardening_passed: bool | None
    hardening_steps: int
    task_spec: TaskSpec | dict[str, Any]
    task_dir: str
