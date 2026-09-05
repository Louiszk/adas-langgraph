from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from adas_core.automatic_validation import (
    AutomaticValidation,
    extract_validation_requirements,
    load_validation_module,
    sanitize_test_id,
)
from adas_core.task_spec import ArchitectureContract, TaskSpec, TestCaseSpec
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.graph import _get_test_case_count
from meta_system.tools import test_system as run_test_system


@pytest.fixture
def sample_task_spec() -> TaskSpec:
    return TaskSpec(
        name="DataReportAgent",
        system_goal="Analyze company finances and output summary.json",
        architecture_contract=ArchitectureContract(
            execution_mode="single_turn", state_schema={"query": "str", "status": "str"}
        ),
        dev_suite=[TestCaseSpec(id="case_1", description="Basic test", turns=[{"query": "run"}])],
    )


def generated_code() -> str:
    return """```python
VALIDATION_REQUIREMENTS = ["pydantic"]
from typing import Any

def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    return final_state.get("status") == "success", "status must be success"

VALIDATORS = {"case_1": validate_case_1}

def validate(test_case_id, final_state, workspace_dirs):
    return VALIDATORS[test_case_id](final_state, workspace_dirs)
```"""


class TestAutomaticValidation:
    def test_helpers(self):
        assert sanitize_test_id("case-1") == "case_1"
        assert sanitize_test_id("1") == "case_1"
        assert extract_validation_requirements('VALIDATION_REQUIREMENTS = ["pandas", "pytest"]') == [
            "pandas",
            "pytest",
        ]

    def test_generates_loadable_llm_authored_module(self, sample_task_spec, tmp_path):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content=generated_code())
        result = AutomaticValidation(llm=llm).generate_all(sample_task_spec, tmp_path)

        assert result.required_packages == ["pydantic"]
        module = load_validation_module(result.validation_file_path)
        assert module.validate("case_1", {"status": "success"}, {}) == (True, "status must be success")

    def test_generation_failure_is_not_replaced_with_a_fallback(self, sample_task_spec, tmp_path):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("provider unavailable")

        with pytest.raises(RuntimeError, match="provider unavailable"):
            AutomaticValidation(llm=llm).generate_all(sample_task_spec, tmp_path)
        assert not list(tmp_path.glob("*.validation.py"))

    def test_missing_required_validator_fails_loudly(self, sample_task_spec):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="```python\ndef not_a_validator(): pass\n```")
        with pytest.raises(ValueError, match="missing expected function 'validate_case_1'"):
            AutomaticValidation(llm=llm).generate_validation_module(sample_task_spec)


class TestTaskSpecRunner:
    @staticmethod
    def _system() -> VirtualAgenticSystem:
        system = VirtualAgenticSystem("ReportAgent")
        system.set_state_attributes({"query": "str", "status": "str"})
        func, parsed = system.get_function(
            "def start_node(state: dict[str, Any]) -> dict[str, Any]:\n    return {'status': 'success'}", "node"
        )
        assert func is not None
        system.create_node("start_node", "Start", func, parsed)
        system.create_edge("__start__", "start_node")
        system.create_edge("start_node", "__end__")
        return system

    @staticmethod
    def _spec() -> TaskSpec:
        return TaskSpec(
            name="SimpleTask",
            system_goal="Return success",
            architecture_contract=ArchitectureContract(state_schema={"query": "str", "status": "str"}),
            dev_suite=[TestCaseSpec(id="case_1", description="success", turns=[{"query": "run"}])],
        )

    def test_requires_a_frozen_validation_module(self, tmp_path):
        output = run_test_system(
            {"target_agentic_system": self._system(), "task_spec": self._spec(), "task_dir": str(tmp_path)}
        )
        assert "EVALUATOR_ERROR" in output
        assert "No frozen validation module found" in output

    def test_checkpoint_selection_counts_task_spec_cases(self):
        spec = self._spec()
        assert _get_test_case_count({"task_spec": spec}) == 1
        assert _get_test_case_count({"task_spec": spec.model_dump()}) == 1

    def test_uses_frozen_validation_module(self, tmp_path):
        (tmp_path / "SimpleTask.validation.py").write_text(
            "def validate_case_1(final_state, workspace_dirs):\n"
            "    return final_state.get('status') == 'success', 'status must be success'\n"
            "VALIDATORS = {'case_1': validate_case_1}\n",
            encoding="utf-8",
        )
        output = run_test_system(
            {"target_agentic_system": self._system(), "task_spec": self._spec(), "task_dir": str(tmp_path)}
        )
        assert "Overall: PASSED" in output
