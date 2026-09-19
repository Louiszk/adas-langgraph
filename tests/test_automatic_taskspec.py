"""Unit and integration tests for adas_core.automatic_taskspec."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from adas_core.automatic_taskspec import (
    AutomaticTaskSpec,
    build_model_catalog_context,
    find_existing_task_spec_file,
    format_assistant_message_for_display,
    run_interactive_wizard,
)
from adas_core.task_spec import TaskSpec
from create_taskspec import main, parse_args

SAMPLE_VALID_TASKSPEC_DICT = {
    "schema_version": "1.0",
    "name": "MathSolver",
    "system_goal": "Solve arithmetic problems accurately.",
    "architecture_contract": {
        "execution_mode": "single_turn",
        "state_schema": {"query": "str", "answer": "str"},
    },
    "available_models": [{"provider": "openai", "model_name": "gpt-4o"}],
    "resource_manifest": {"available_resources": [], "available_api_keys": []},
    "required_packages": ["numpy"],
    "test_fixtures": {"files": [], "databases": [], "mcps": [], "mock_services": [], "custom_fixtures": []},
    "dev_suite": [
        {
            "id": "case_1_add",
            "description": "2 + 2",
            "turns": [{"query": "What is 2+2?"}],
            "expected_outputs": ["answer"],
            "deterministic_criteria": "4 in output",
            "llm_judge_needed": False,
            "modalities": ["text"],
        }
    ],
}


def test_format_assistant_message_for_display():
    content = (
        "I've drafted the schema for you!\n"
        f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```\n"
        "Please review the state schema and let me know if you want changes."
    )
    saved_path = Path("specs/mathsolver/task.json")
    formatted = format_assistant_message_for_display(content, saved_path=saved_path)

    assert "I've drafted the schema for you!" in formatted
    assert "Please review the state schema" in formatted
    assert "[Draft TaskSpec persisted to: specs/mathsolver/task.json]" in formatted
    assert '"schema_version"' not in formatted  # Huge json is cleanly replaced


class TestModelCatalogContext:
    def test_build_model_catalog_context_includes_models_and_constraints(self):
        catalog = build_model_catalog_context()
        assert "gpt-4o" in catalog
        assert "gpt-5.6-luna" in catalog
        assert "gpt-5.4" in catalog
        assert "o3" in catalog
        assert "Supports Web Search" in catalog
        assert "VISION EVALUATION" in catalog
        assert "REASONING EFFORT" in catalog


class TestAutomaticTaskSpecSynthesis:
    def test_generate_task_spec_success(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```")

        gen = AutomaticTaskSpec(llm=mock_llm)
        spec = gen.generate_task_spec("Solve arithmetic problems")

        assert isinstance(spec, TaskSpec)
        assert spec.name == "MathSolver"
        assert spec.architecture_contract.execution_mode == "single_turn"
        assert len(spec.dev_suite) == 1
        assert spec.dev_suite[0].id == "case_1_add"
        assert mock_llm.invoke.call_count == 1

    def test_generate_task_spec_retries_on_validation_error(self):
        mock_llm = MagicMock()
        # Attempt 1: missing required system_goal
        invalid_dict = dict(SAMPLE_VALID_TASKSPEC_DICT)
        del invalid_dict["system_goal"]

        # Attempt 2: valid
        mock_llm.invoke.side_effect = [
            AIMessage(content=f"```json\n{json.dumps(invalid_dict)}\n```"),
            AIMessage(content=f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```"),
        ]

        gen = AutomaticTaskSpec(llm=mock_llm)
        spec = gen.generate_task_spec("Solve arithmetic problems")

        assert isinstance(spec, TaskSpec)
        assert spec.name == "MathSolver"
        assert mock_llm.invoke.call_count == 2

    def test_generate_task_spec_with_existing_spec_refinement(self):
        mock_llm = MagicMock()
        updated_dict = dict(SAMPLE_VALID_TASKSPEC_DICT)
        updated_dict["required_packages"] = ["numpy", "scipy"]

        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json.dumps(updated_dict)}\n```")

        gen = AutomaticTaskSpec(llm=mock_llm)
        spec = gen.generate_task_spec(
            {
                "system_goal": "Add scipy to required packages",
                "existing_spec": SAMPLE_VALID_TASKSPEC_DICT,
            }
        )

        assert isinstance(spec, TaskSpec)
        assert "scipy" in spec.required_packages
        call_prompt = mock_llm.invoke.call_args[0][0][1].content
        assert "Existing TaskSpec specification to refine" in call_prompt
        assert "Add scipy to required packages" in call_prompt

    def test_find_existing_task_spec_file(self, tmp_path):
        # 1. Non-existent returns None
        assert find_existing_task_spec_file(output_dir=tmp_path, task_name="Missing") is None

        # 2. File exists in directory matching task_name
        task_dir = tmp_path / "my_agent"
        task_dir.mkdir()
        task_file = task_dir / "my_agent.task.json"
        task_file.write_text("{}", encoding="utf-8")

        found = find_existing_task_spec_file(output_dir=task_dir, task_name="my_agent")
        assert found == task_file

        # 3. Direct file path
        found_direct = find_existing_task_spec_file(output_dir=task_file)
        assert found_direct == task_file

        # A staging directory named task.json is not itself a TaskSpec file.
        (tmp_path / "task.json").mkdir()
        assert find_existing_task_spec_file(output_dir=tmp_path) is None

    def test_save_task_spec_creates_file(self, tmp_path):
        gen = AutomaticTaskSpec(llm=MagicMock())
        spec = TaskSpec.model_validate(SAMPLE_VALID_TASKSPEC_DICT)
        out_file = gen.save_task_spec(spec, output_dir=tmp_path / "tasks" / "math_solver")

        assert out_file.is_file()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded["name"] == "MathSolver"
        assert loaded["schema_version"] == "1.0"


class TestInteractiveWizard:
    def test_run_interactive_wizard_non_interactive(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```")
        gen = AutomaticTaskSpec(llm=mock_llm)

        spec = run_interactive_wizard(
            initial_prompt="Solve math problems",
            output_dir=tmp_path,
            non_interactive=True,
            generator=gen,
        )

        assert spec is not None
        assert spec.name == "MathSolver"
        assert (tmp_path / "task.json").exists()

    def test_run_interactive_wizard_grilling_and_persistence(self, monkeypatch, tmp_path):
        mock_llm = MagicMock()
        # Turn 1: Architect grills the user with questions (no JSON)
        # Turn 2: Architect outputs schema JSON
        mock_llm.invoke.side_effect = [
            AIMessage(content="Let's clarify requirements! 1. Single-turn or multi-turn? 2. What libraries?"),
            AIMessage(
                content=(
                    "Great! I've drafted the specification for MathSolver.\n"
                    f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```\n"
                    "Take a look at the state keys. Any adjustments?"
                )
            ),
        ]
        gen = AutomaticTaskSpec(llm=mock_llm)

        # Simulation:
        # 1. User answers initial prompt in input()
        # 2. User answers grilling questions
        # 3. User types "done" to finalize
        inputs = iter(
            [
                "I want a math solving assistant",
                "Single turn pipeline using numpy",
                "done",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            output_dir=tmp_path,
            generator=gen,
        )

        assert spec is not None
        assert spec.name == "MathSolver"
        target_file = tmp_path / "task.json"
        assert target_file.exists()

    def test_run_interactive_wizard_refinement_conversation(self, monkeypatch, tmp_path):
        mock_llm = MagicMock()
        refined_dict = dict(SAMPLE_VALID_TASKSPEC_DICT)
        refined_dict["name"] = "RefinedSolver"

        # Turn 1: Architect drafts initial spec
        # Turn 2: Architect updates spec to RefinedSolver based on user feedback
        mock_llm.invoke.side_effect = [
            AIMessage(content=f"Initial draft:\n```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```"),
            AIMessage(content=f"Updated draft:\n```json\n{json.dumps(refined_dict)}\n```"),
        ]
        gen = AutomaticTaskSpec(llm=mock_llm)

        inputs = iter(
            [
                "Rename to RefinedSolver",
                "done",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            initial_prompt="I want a math solver",
            output_dir=tmp_path,
            generator=gen,
        )

        assert spec is not None
        assert spec.name == "RefinedSolver"
        assert (tmp_path / "task.json").exists()

    def test_run_interactive_wizard_exit_before_spec(self, monkeypatch, tmp_path):
        gen = AutomaticTaskSpec(llm=MagicMock())
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            output_dir=tmp_path,
            generator=gen,
        )
        assert spec is None

    def test_run_interactive_wizard_commands_do_not_invoke_llm(self, monkeypatch, tmp_path, capsys):
        mock_llm = MagicMock()
        gen = AutomaticTaskSpec(llm=mock_llm)

        inputs = iter(
            [
                "help",
                "done",
                "exit",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            output_dir=tmp_path,
            generator=gen,
        )

        assert spec is None
        captured = capsys.readouterr().out
        assert "No TaskSpec has been drafted yet" in captured
        assert "Available commands:" in captured

    def test_run_interactive_wizard_loads_existing_spec_and_refines(self, monkeypatch, tmp_path, capsys):
        existing_file = tmp_path / "mathsolver.task.json"
        existing_file.write_text(json.dumps(SAMPLE_VALID_TASKSPEC_DICT, indent=2), encoding="utf-8")

        refined_dict = dict(SAMPLE_VALID_TASKSPEC_DICT)
        refined_dict["name"] = "MathSolverV2"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content=f"Refined specification:\n```json\n{json.dumps(refined_dict)}\n```"
        )
        gen = AutomaticTaskSpec(llm=mock_llm)

        inputs = iter(
            [
                "Upgrade to MathSolverV2",
                "done",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            output_dir=tmp_path,
            generator=gen,
            task_name="MathSolver",
        )

        assert spec is not None
        assert spec.name == "MathSolverV2"
        captured = capsys.readouterr().out
        assert "Found existing TaskSpec for 'MathSolver'" in captured

        # Verify LLM received existing spec context
        first_call_msgs = mock_llm.invoke.call_args_list[0][0][0]
        context_msg = first_call_msgs[1].content
        assert "An existing TaskSpec specification was loaded" in context_msg
        assert "MathSolver" in context_msg

    def test_run_interactive_wizard_reset_command(self, monkeypatch, tmp_path, capsys):
        existing_file = tmp_path / "mathsolver.task.json"
        existing_file.write_text(json.dumps(SAMPLE_VALID_TASKSPEC_DICT, indent=2), encoding="utf-8")

        mock_llm = MagicMock()
        gen = AutomaticTaskSpec(llm=mock_llm)

        inputs = iter(
            [
                "reset",
                "exit",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))

        spec = run_interactive_wizard(
            output_dir=tmp_path,
            generator=gen,
            task_name="MathSolver",
        )

        assert spec is None
        captured = capsys.readouterr().out
        assert "Resetting to a clean slate" in captured

    def test_run_interactive_wizard_non_interactive_with_existing_spec(self, tmp_path):
        existing_file = tmp_path / "mathsolver.task.json"
        existing_file.write_text(json.dumps(SAMPLE_VALID_TASKSPEC_DICT, indent=2), encoding="utf-8")

        refined_dict = dict(SAMPLE_VALID_TASKSPEC_DICT)
        refined_dict["required_packages"] = ["scipy"]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json.dumps(refined_dict)}\n```")
        gen = AutomaticTaskSpec(llm=mock_llm)

        spec = run_interactive_wizard(
            initial_prompt="Add scipy to dependencies",
            output_dir=tmp_path,
            non_interactive=True,
            task_name="MathSolver",
            generator=gen,
        )

        assert spec is not None
        assert spec.required_packages == ["scipy"]
        call_prompt = mock_llm.invoke.call_args[0][0][1].content
        assert "Existing TaskSpec specification to refine" in call_prompt


class TestCreateTaskSpecCLI:
    def test_parse_args(self):
        args = parse_args(["--name", "CustomAgent", "--goal", "Do task", "--non-interactive"])
        assert args.name == "CustomAgent"
        assert args.goal == "Do task"
        assert args.non_interactive is True

    def test_main_cli_execution(self, monkeypatch, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=f"```json\n{json.dumps(SAMPLE_VALID_TASKSPEC_DICT)}\n```")

        with patch("adas_core.automatic_taskspec.ChatModel", return_value=mock_llm):
            ret = main(
                [
                    "--name",
                    "MathSolver",
                    "--goal",
                    "Solve arithmetic problems",
                    "--output-dir",
                    str(tmp_path),
                    "--non-interactive",
                ]
            )

        assert ret == 0
        assert (tmp_path / "task.json").exists()
