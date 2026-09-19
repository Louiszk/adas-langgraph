"""Unit tests for adas_core.test_runner."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from unittest.mock import patch

from adas_core.chat_model import UsageRecorder
from adas_core.task_spec import (
    ArchitectureContract,
    ExternalDatabaseSeedSpec,
    FileFixtureSpec,
    MockServiceFixtureSpec,
    ResourceEntry,
    ResourceManifest,
    TaskSpec,
    TestCaseSpec,
    TestFixturesSpec,
)
from adas_core.test_runner import execute_test_suite
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.tools import test_system as run_test_system


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
    def test_execute_test_suite_cleans_external_seed_after_validator_failure(self, tmp_path, monkeypatch):
        events = tmp_path / "seed-events.txt"
        scripts = tmp_path / "setup_scripts"
        scripts.mkdir()
        (scripts / "seed_external_orders_seed.py").write_text(
            "from pathlib import Path\n"
            f"EVENTS = Path({str(events)!r})\n"
            "def _record(value):\n"
            "    EVENTS.write_text((EVENTS.read_text() if EVENTS.exists() else '') + value + '\\n')\n"
            "def seed_external_database(connection_config, namespace):\n"
            "    _record('seed:' + namespace)\n"
            "def cleanup_external_database(connection_config, namespace):\n"
            "    _record('cleanup:' + namespace)\n",
            encoding="utf-8",
        )
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        monkeypatch.setenv("EVALUATION_DB_URI", "postgres://secret@example/test")
        spec = TaskSpec(
            name="ExternalSeedRunner",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"query": "str"}),
            resource_manifest=ResourceManifest(
                available_resources=[ResourceEntry(name="evaluation_db", type="database")]
            ),
            test_fixtures=TestFixturesSpec(
                external_database_seeds=[
                    ExternalDatabaseSeedSpec(
                        name="orders_seed",
                        resource_name="evaluation_db",
                        db_type="postgres",
                        driver="psycopg",
                        connection_env={"uri": "EVALUATION_DB_URI"},
                        namespace_kind="schema",
                        namespace="adas_test_orders",
                        description="Seed deterministic order rows.",
                    )
                ]
            ),
            dev_suite=[
                TestCaseSpec(id="seeded", description="desc", fixture_ids=["orders_seed"], turns=[{"query": "q"}])
            ],
        )

        class Validator:
            @staticmethod
            def validate_seeded(final_state, workspace_dirs):
                return False, "intentional validator failure"

        result = execute_test_suite(
            _create_dummy_system("ExternalSeedSystem"),
            spec.dev_suite,
            Validator(),
            workspace_root=tmp_path / "workspace",
            fixtures_dir=fixtures_dir,
            task_spec=spec,
            stop_on_first_failure=True,
        )

        assert not result.all_passed
        assert events.read_text(encoding="utf-8").splitlines() == ["seed:adas_test_orders", "cleanup:adas_test_orders"]

    def test_execute_test_suite_runs_selected_process_fixture(self, tmp_path):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        fixtures_dir = tmp_path / "custom-process-fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "mock_weather.py").write_text(
            "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self): self.send_response(200); self.end_headers()\n"
            "    def log_message(self, *args): pass\n"
            f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n",
            encoding="utf-8",
        )
        system = VirtualAgenticSystem("ProcessFixtureSystem")
        system.set_state_attributes({"query": "str", "result": "str"})
        func, parsed = system.get_function(
            "import os\n"
            "def processor(state: dict[str, Any]) -> dict[str, Any]:\n"
            "    return {'result': os.environ.get('WEATHER_URL', '')}\n",
            "node",
        )
        assert func is not None
        system.create_node("processor", "Reads fixture URL", func, parsed)
        system.create_edge("__start__", "processor")
        system.create_edge("processor", "__end__")
        spec = TaskSpec(
            name="ProcessFixtureTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"query": "str"}),
            test_fixtures=TestFixturesSpec(
                mock_services=[MockServiceFixtureSpec(name="weather", port=port, base_url_env="WEATHER_URL")]
            ),
            dev_suite=[
                TestCaseSpec(id="selected", description="desc", fixture_ids=["weather"], turns=[{"query": "q"}])
            ],
        )

        class Validator:
            @staticmethod
            def validate_selected(final_state, workspace_dirs):
                return final_state["result"] == f"http://127.0.0.1:{port}", "fixture URL was injected"

        result = execute_test_suite(
            system,
            spec.dev_suite,
            Validator(),
            workspace_root=tmp_path / "workspace",
            fixtures_dir=fixtures_dir,
            task_spec=spec,
        )
        assert result.all_passed

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
            output = run_test_system(state)

        assert "ERROR: running the test_system tool:" in output
        assert "catastrophic failure" in output
        assert "Overall: FAILED" in output
        assert state.get("system_passed") is False

    def test_execute_test_suite_tracks_executed_count_and_summed_iterations(self, tmp_path):
        system = _create_dummy_system("IterationCountSystem")
        test_cases = [
            TestCaseSpec(id="case_1", description="first", turns=[{"query": "1"}]),
            TestCaseSpec(id="case_2", description="second", turns=[{"query": "2"}]),
            TestCaseSpec(id="case_3", description="third", turns=[{"query": "3"}]),
        ]

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return False, "Failed at 1"

            @staticmethod
            def validate_case_2(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return True, "ok 2"

            @staticmethod
            def validate_case_3(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return True, "ok 3"

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=MockValidation(),
            stop_on_first_failure=True,
            workspace_root=tmp_path / "workspace",
        )

        # Early stopping at case 1 -> executed_count must be 1, total_tests 3
        assert result.executed_count == 1
        assert result.total_tests == 3
        assert len(result.case_results) == 1
        assert result.total_iterations == result.case_results[0].total_iterations
        assert result.max_iterations == result.case_results[0].total_iterations

    def test_test_system_bases_averages_on_executed_count(self, tmp_path):
        system = _create_dummy_system("EarlyStopAveragesSystem")
        spec = TaskSpec(
            name="EarlyStopTask",
            system_goal="Test early stopping averages",
            architecture_contract=ArchitectureContract(state_schema={"query": "str", "result": "str"}),
            dev_suite=[
                TestCaseSpec(id="case_1", description="first fails", turns=[{"query": "fail"}]),
                TestCaseSpec(id="case_2", description="second not run", turns=[{"query": "skip"}]),
                TestCaseSpec(id="case_3", description="third not run", turns=[{"query": "skip"}]),
                TestCaseSpec(id="case_4", description="fourth not run", turns=[{"query": "skip"}]),
            ],
        )
        (tmp_path / "EarlyStopTask.validation.py").write_text(
            "def validate_case_1(final_state, workspace_dirs):\n"
            "    return False, 'Forced failure'\n"
            "VALIDATORS = {'case_1': validate_case_1}\n",
            encoding="utf-8",
        )

        state = {
            "target_agentic_system": system,
            "task_spec": spec,
            "task_dir": str(tmp_path),
            "messages": [],
        }

        output = run_test_system(state)
        assert "Overall: FAILED" in output
        assert state["test_metrics"]["executed_count"] == 1
        assert state["test_metrics"]["total"] == 4
        assert state["test_metrics"]["passed"] == 0
        assert "Avg. Graph Iterations:" in output

    def test_incomplete_usage_marks_telemetry_none(self, tmp_path):
        system = _create_dummy_system("IncompleteUsageSystem")
        test_cases = [TestCaseSpec(id="case_1", description="first", turns=[{"query": "1"}])]

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return True, "ok"

        before_stats = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "incomplete_usage_count": 0,
        }
        after_stats = {
            "llm_calls": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "duration_seconds": 0.1,
            "incomplete_usage_count": 1,
        }

        with patch.object(UsageRecorder, "get_aggregate", side_effect=[before_stats, after_stats]):
            result = execute_test_suite(
                system=system,
                test_cases=test_cases,
                validation_module=MockValidation(),
                workspace_root=tmp_path / "workspace",
            )

            assert result.token_usage["total_tokens"] is None
            assert result.token_usage["input_tokens"] is None
            assert result.token_usage["output_tokens"] is None

    def test_zero_usage_remains_zero(self, tmp_path):
        system = _create_dummy_system("ZeroUsageSystem")
        test_cases = [TestCaseSpec(id="case_1", description="first", turns=[{"query": "1"}])]

        class MockValidation:
            @staticmethod
            def validate_case_1(final_state: dict[str, Any], workspace_dirs: dict[str, str]):
                return True, "ok"

        result = execute_test_suite(
            system=system,
            test_cases=test_cases,
            validation_module=MockValidation(),
            workspace_root=tmp_path / "workspace",
        )

        assert result.token_usage["llm_calls"] == 0
        assert result.token_usage["input_tokens"] == 0
        assert result.token_usage["output_tokens"] == 0
        assert result.token_usage["total_tokens"] == 0
