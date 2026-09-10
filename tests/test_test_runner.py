"""Unit tests for adas_core.test_runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adas_core.task_spec import ArchitectureContract, FileFixtureSpec, TaskSpec, TestCaseSpec, TestFixturesSpec
from adas_core.test_runner import execute_test_suite
from adas_core.virtual_agentic_system import VirtualAgenticSystem


def _create_dummy_system(name: str = "TestSystem") -> VirtualAgenticSystem:
    system = VirtualAgenticSystem(name)
    system.set_state_attributes({"query": "str", "result": "str"})
    func, parsed = system.get_function(
        "def processor(state: dict[str, Any]) -> dict[str, Any]:\n"
        "    return {'result': f\"processed_{state.get('query', '')}\"}\n",
        "node",
    )
    assert func is not None
    system.create_node("processor", "Processes input query", func, parsed)
    system.create_edge("__start__", "processor")
    system.create_edge("processor", "__end__")
    return system


class TestExecuteTestSuite:
    def test_all_passing(self, tmp_path):
        system = _create_dummy_system("PassSystem")
        test_cases = [
            TestCaseSpec(id="case_1", description="test 1", turns=[{"query": "alpha"}]),
            TestCaseSpec(id="case_2", description="test 2", turns=[{"query": "beta"}]),
        ]

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return final_state.get("result") == "processed_alpha", "ok alpha"

            @staticmethod
            def validate_case_2(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return final_state.get("result") == "processed_beta", "ok beta"

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=MockValidation(),
            workspace_root=tmp_path / "workspace",
        )

        assert result.all_passed is True
        assert result.passed_count == 2
        assert result.total_tests == 2
        assert result.pass_rate == 1.0
        assert len(result.case_results) == 2
        assert result.case_results[0].passed is True
        assert result.case_results[1].passed is True
        assert "All 2 test cases passed successfully." in result.summary_lines

    def test_execute_test_suite_isolates_fixtures_and_excludes_provisioning_scripts(self, tmp_path):
        system = _create_dummy_system("IsolationSystem")
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "data.csv").write_text("id,val\n1,10", encoding="utf-8")
        (fixtures_dir / "generate_data.py").write_text("# script", encoding="utf-8")
        (fixtures_dir / "unrelated.txt").write_text("unrelated", encoding="utf-8")

        spec = TaskSpec(
            name="IsoTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"query": "str", "result": "str"},
            ),
            test_fixtures=TestFixturesSpec(
                files=[FileFixtureSpec(id="data_csv", path="data.csv")],
            ),
            dev_suite=[
                TestCaseSpec(id="case_1", description="test", turns=[{"query": "run"}]),
            ],
        )

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                in_dir = Path(workspace_dirs["input"])
                assert (in_dir / "data.csv").exists()
                assert not (in_dir / "generate_data.py").exists()
                assert not (in_dir / "unrelated.txt").exists()
                return True, "verified isolation"

        result = execute_test_suite(
            system=system,
            test_cases=spec.dev_suite,
            validation_module=MockValidation(),
            workspace_root=tmp_path / "workspace",
            fixtures_dir=fixtures_dir,
            task_spec=spec,
        )

        assert result.all_passed is True
        assert result.passed_count == 1
        assert result.case_results[0].message == "verified isolation"

    def test_structural_errors(self):
        system = _create_dummy_system("BrokenSystem")
        system.create_node("orphan", "Orphan", lambda s: s, "def orphan(s): return s")
        test_cases = [TestCaseSpec(id="c1", description="desc", turns=[{"query": "q"}])]

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=None,
        )

        assert result.all_passed is False
        assert len(result.structural_errors) > 0
        assert result.passed_count == 0

    def test_empty_test_suite(self):
        system = _create_dummy_system("EmptySuiteSystem")
        result = execute_test_suite(
            system=system,
            test_cases=[],
            validation_module=None,
        )

        assert result.all_passed is True
        assert result.total_tests == 0
        assert result.passed_count == 0
        assert result.pass_rate == 1.0

    def test_stop_on_first_failure(self, tmp_path):
        system = _create_dummy_system("FailEarlySystem")
        test_cases = [
            TestCaseSpec(id="case_1", description="failing first", turns=[{"query": "fail"}]),
            TestCaseSpec(id="case_2", description="second", turns=[{"query": "ok"}]),
        ]

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return False, "Failed intentionally"

            @staticmethod
            def validate_case_2(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return True, "Should not reach"

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=MockValidation(),
            stop_on_first_failure=True,
            workspace_root=tmp_path / "workspace",
        )

        assert result.all_passed is False
        assert result.passed_count == 0
        # Only case 1 was executed because stop_on_first_failure=True
        assert len(result.case_results) == 1
        assert result.case_results[0].test_case_id == "case_1"
        assert result.case_results[0].passed is False

    def test_dispatch_fallback_to_validate_dispatcher(self, tmp_path):
        system = _create_dummy_system("DispatchSystem")
        test_cases = [
            TestCaseSpec(id="case_dispatch", description="desc", turns=[{"query": "xyz"}]),
        ]

        class MockValidation:
            @staticmethod
            def validate(case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                assert case_id == "case_dispatch"
                return True, "Dispatched successfully"

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=MockValidation(),
            workspace_root=tmp_path / "workspace",
        )

        assert result.all_passed is True
        assert result.passed_count == 1
        assert result.case_results[0].message == "Dispatched successfully"

    def test_validate_graph_exception_handled(self):
        system = _create_dummy_system("ExceptionSystem")
        system.validate_graph = lambda: (_ for _ in ()).throw(RuntimeError("corrupted AST"))  # type: ignore
        test_cases = [TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])]

        result = execute_test_suite(system=system, test_cases=test_cases, validation_module=None)
        assert result.all_passed is False
        assert result.passed_count == 0
        assert any("corrupted AST" in err for err in result.structural_errors)

    def test_test_system_catches_unhandled_exception(self, tmp_path):
        from unittest.mock import patch
        from meta_system.tools import test_system
        from adas_core.task_spec import TaskSpec, ArchitectureContract

        system = _create_dummy_system("CrashSystem")
        spec = TaskSpec(
            name="CrashTask",
            system_goal="Test crashes",
            architecture_contract=ArchitectureContract(state_schema={"query": "str"}),
            dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"query": "1"}])],
        )
        (tmp_path / "CrashTask.validation.py").write_text("VALIDATORS = {}\n", encoding="utf-8")

        state = {
            "target_agentic_system": system,
            "task_spec": spec,
            "task_dir": str(tmp_path),
        }

        with patch("meta_system.tools.execute_test_suite", side_effect=RuntimeError("catastrophic failure")):
            output = test_system(state)

        assert "ERROR: running the test_system tool:" in output
        assert "catastrophic failure" in output
        assert "Overall: FAILED" in output
        assert state.get("system_passed") is False
