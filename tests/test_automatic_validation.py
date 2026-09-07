from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from adas_core.automatic_validation import (
    AutomaticValidation,
    assemble_validation_module,
    discover_fixture_generators,
    extract_validation_requirements,
    load_validation_module,
    sanitize_test_id,
)
from adas_core.task_spec import ArchitectureContract, TaskSpec, TestCaseSpec
from adas_core.virtual_agentic_system import VirtualAgenticSystem
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

    def test_includes_fixture_generators_context(self, sample_task_spec, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir(parents=True)
        gen_script = fixtures_dir / "generate_test_csv.py"
        gen_script.write_text("def generate_files(p): pass\n", encoding="utf-8")

        discovered = discover_fixture_generators(tmp_path)
        assert "generate_test_csv.py" in discovered

        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content=generated_code())
        AutomaticValidation(llm=llm).generate_all(sample_task_spec, tmp_path)

        # Verify LLM prompt included fixture generation code
        invoked_messages = llm.invoke.call_args[0][0]
        user_message_content = invoked_messages[1].content
        assert "FIXTURE GENERATION CODE:" in user_message_content
        assert "generate_test_csv.py" in user_message_content

    def test_assemble_validation_module_rejects_sanitized_collisions(self, sample_task_spec):
        case_a = TestCaseSpec(id="case-a", description="case a", turns=[{"q": "1"}])
        case_b = TestCaseSpec(id="case_a", description="case a dup", turns=[{"q": "2"}])
        snippets = [
            (case_a, "def validate_case_a(state, ws): return True, 'ok'"),
            (case_b, "def validate_case_a(state, ws): return True, 'ok'"),
        ]
        with pytest.raises(ValueError, match="Duplicate sanitized test case id 'case_a'"):
            assemble_validation_module(sample_task_spec, snippets, [])

    def test_generates_per_case_with_scoped_fixtures(self, tmp_path):
        from adas_core.task_spec import FileFixtureSpec, TestFixturesSpec

        task_spec = TaskSpec(
            name="ScopedAnalystAgent",
            system_goal="Analyze data scoped per case",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"status": "str"}),
            test_fixtures=TestFixturesSpec(
                files=[
                    FileFixtureSpec(id="sales_csv", path="sales.csv", description="Sales CSV"),
                    FileFixtureSpec(id="inventory_json", path="inventory.json", description="Inventory JSON"),
                ]
            ),
            dev_suite=[
                TestCaseSpec(
                    id="case_1", description="Sales only", turns=[{"query": "sales"}], fixture_ids=["sales_csv"]
                ),
                TestCaseSpec(
                    id="case_2",
                    description="Inventory only",
                    turns=[{"query": "inventory"}],
                    fixture_ids=["inventory_json"],
                ),
                TestCaseSpec(
                    id="case_3", description="No fixtures", turns=[{"query": "pure_reasoning"}], fixture_ids=[]
                ),
            ],
        )

        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir(parents=True)
        (fixtures_dir / "generate_sales_csv.py").write_text("def generate_files(p): pass  # sales\n", encoding="utf-8")
        (fixtures_dir / "generate_inventory_json.py").write_text(
            "def generate_files(p): pass  # inventory\n", encoding="utf-8"
        )

        code_case_1 = """```python
VALIDATION_REQUIREMENTS = ["pandas"]
def validate_case_1(final_state, workspace_dirs):
    return True, "case 1 passed"
```"""
        code_case_2 = """```python
VALIDATION_REQUIREMENTS = ["pydantic"]
def validate_case_2(final_state, workspace_dirs):
    return True, "case 2 passed"
```"""
        code_case_3 = """```python
VALIDATION_REQUIREMENTS = []
def validate_case_3(final_state, workspace_dirs):
    return True, "case 3 passed"
```"""

        llm = MagicMock()
        llm.invoke.side_effect = [
            AIMessage(content=code_case_1),
            AIMessage(content=code_case_2),
            AIMessage(content=code_case_3),
        ]

        result = AutomaticValidation(llm=llm).generate_all(task_spec, tmp_path)

        # 1. Verify LLM was invoked exactly once per test case (3 times)
        assert llm.invoke.call_count == 3

        # 2. Verify Case 1 prompt only had sales fixture code
        call_1_prompt = llm.invoke.call_args_list[0][0][0][1].content
        assert "generate_sales_csv.py" in call_1_prompt
        assert "generate_inventory_json.py" not in call_1_prompt

        # 3. Verify Case 2 prompt only had inventory fixture code
        call_2_prompt = llm.invoke.call_args_list[1][0][0][1].content
        assert "generate_inventory_json.py" in call_2_prompt
        assert "generate_sales_csv.py" not in call_2_prompt

        # 4. Verify Case 3 prompt had no fixture code
        call_3_prompt = llm.invoke.call_args_list[2][0][0][1].content
        assert "No input fixtures are mounted for this test case" in call_3_prompt
        assert "generate_sales_csv.py" not in call_3_prompt
        assert "generate_inventory_json.py" not in call_3_prompt

        # 5. Verify requirements aggregation
        assert sorted(result.required_packages) == ["pandas", "pydantic"]

        # 6. Verify module execution and dispatch
        module = load_validation_module(result.validation_file_path)
        assert module.validate("case_1", {}, {}) == (True, "case 1 passed")
        assert module.validate("case_2", {}, {}) == (True, "case 2 passed")
        assert module.validate("case_3", {}, {}) == (True, "case 3 passed")
        assert module.validate("non_existent", {}, {}) == (False, "No validator found for test case 'non_existent'")

    def test_filter_fixture_generators_exact_match_no_substring_false_positives(self):
        from adas_core.automatic_validation import filter_fixture_generators_for_case
        from adas_core.task_spec import FileFixtureSpec, TestFixturesSpec

        spec = TaskSpec(
            name="OrderAgent",
            system_goal="Process orders",
            architecture_contract=ArchitectureContract(state_schema={"q": "str"}),
            test_fixtures=TestFixturesSpec(
                files=[
                    FileFixtureSpec(id="order", path="order.csv", description="Order"),
                    FileFixtureSpec(id="reorder", path="reorder.csv", description="Reorder"),
                ]
            ),
            dev_suite=[
                TestCaseSpec(id="case_order", description="Order only", turns=[{"q": "1"}], fixture_ids=["order"]),
            ],
        )

        generators = {
            "generate_order.py": "def order(): pass",
            "generate_reorder.py": "def reorder(): pass",
        }

        filtered = filter_fixture_generators_for_case(spec, spec.dev_suite[0], generators)
        # Must match generate_order.py exactly, and NOT generate_reorder.py via substring
        assert "generate_order.py" in filtered
        assert "generate_reorder.py" not in filtered


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
