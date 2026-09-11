from __future__ import annotations

import os
from typing import Any

import dill as pickle

from adas_core.candidate_selection import (
    CandidateRecord,
    candidate_rank_key,
    finalize_best_candidate,
    record_candidate,
    record_candidate_evaluation,
    save_candidate_checkpoint,
    select_best_candidate,
)
from adas_core.task_spec import ArchitectureContract, TaskSpec, TestCaseSpec
from adas_core.virtual_agentic_system import VirtualAgenticSystem
from meta_system.graph import design_completed_condition
from meta_system.tools import test_system as run_test_system


def _dummy_system(name: str = "TestSystem") -> VirtualAgenticSystem:
    system = VirtualAgenticSystem(name)
    system.set_state_attributes({"query": "str", "status": "str"})
    func, parsed = system.get_function(
        "def start_node(state: dict[str, Any]) -> dict[str, Any]:\n    return {'status': 'success'}", "node"
    )
    assert func is not None
    system.create_node("start_node", "Start", func, parsed)
    system.create_edge("__start__", "start_node")
    system.create_edge("start_node", "__end__")
    return system


class TestCandidateSelection:
    def test_record_candidate_appends_and_initializes(self):
        state: dict[str, Any] = {}
        c1: CandidateRecord = {
            "iteration": 0,
            "dev_pass_rate": 0.5,
            "passed_count": 1,
            "total_count": 2,
            "total_tokens": 100,
            "duration_seconds": 1.2,
        }
        record_candidate(state, c1)
        assert len(state["candidates"]) == 1
        assert state["candidates"][0]["iteration"] == 0

        c2: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "passed_count": 2,
            "total_count": 2,
            "total_tokens": 150,
            "duration_seconds": 1.5,
        }
        record_candidate(state, c2)
        assert len(state["candidates"]) == 2
        assert state["candidates"][1]["iteration"] == 1

    def test_save_candidate_checkpoint_creates_file(self, tmp_path):
        sys = _dummy_system("SaveTest")
        ckpt_path = save_candidate_checkpoint(sys, iteration=3, code_dir=str(tmp_path))
        assert os.path.exists(ckpt_path)
        assert "SaveTest_candidate_iter_3.pkl" in ckpt_path

    def test_record_candidate_evaluation_creates_checkpoint_and_record(self, tmp_path):
        sys = _dummy_system("EvalTest")
        state: dict[str, Any] = {}
        cand = record_candidate_evaluation(
            state=state,
            system=sys,
            iteration=2,
            passed_count=2,
            total_count=3,
            total_tokens=450,
            duration_seconds=1.8,
            llm_calls=3,
            code_dir=str(tmp_path),
        )
        assert len(state["candidates"]) == 1
        assert state["candidates"][0] == cand
        assert cand.get("iteration") == 2
        assert float(cand.get("dev_pass_rate", 0.0)) > 0.66
        assert cand.get("checkpoint_path") != ""
        assert os.path.exists(str(cand.get("checkpoint_path")))

    def test_candidate_rank_key_pass_rate_primary(self):
        c_high_pass: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 10000,
            "duration_seconds": 10.0,
        }
        c_low_pass: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 0.5,
            "total_tokens": 50,
            "duration_seconds": 0.1,
        }
        assert candidate_rank_key(c_high_pass) > candidate_rank_key(c_low_pass)
        assert select_best_candidate([c_low_pass, c_high_pass]) == c_high_pass

    def test_candidate_rank_key_tokens_secondary(self):
        c_efficient: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 500,
            "duration_seconds": 3.0,
        }
        c_expensive: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": 2000,
            "duration_seconds": 1.0,
        }
        assert candidate_rank_key(c_efficient) > candidate_rank_key(c_expensive)
        assert select_best_candidate([c_expensive, c_efficient]) == c_efficient

    def test_candidate_rank_key_duration_tertiary(self):
        c_fast: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 500,
            "duration_seconds": 1.5,
        }
        c_slow: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": 500,
            "duration_seconds": 4.5,
        }
        assert candidate_rank_key(c_fast) > candidate_rank_key(c_slow)
        assert select_best_candidate([c_slow, c_fast]) == c_fast

    def test_candidate_rank_key_iteration_quaternary(self):
        c_early: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 500,
            "duration_seconds": 2.0,
        }
        c_late: CandidateRecord = {
            "iteration": 5,
            "dev_pass_rate": 1.0,
            "total_tokens": 500,
            "duration_seconds": 2.0,
        }
        assert candidate_rank_key(c_late) > candidate_rank_key(c_early)
        assert select_best_candidate([c_early, c_late]) == c_late

    def test_select_best_candidate_empty(self):
        assert select_best_candidate([]) is None

    def test_candidate_ranking_preference_runtime_vs_tokens(self):
        c_low_tokens_slow: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 100,
            "duration_seconds": 5.0,
        }
        c_high_tokens_fast: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": 1000,
            "duration_seconds": 0.5,
        }
        # Under "tokens" preference (default), lower tokens wins
        best_tokens = select_best_candidate([c_low_tokens_slow, c_high_tokens_fast], preference="tokens")
        assert best_tokens == c_low_tokens_slow

        # Under "runtime" preference, lower latency wins
        best_runtime = select_best_candidate([c_low_tokens_slow, c_high_tokens_fast], preference="runtime")
        assert best_runtime == c_high_tokens_fast

    def test_finalize_best_candidate_restores_from_checkpoint(self, tmp_path):
        code_dir = str(tmp_path / "output")
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        system_v1 = _dummy_system("SystemV1")
        ckpt_path_1 = str(checkpoints_dir / "SystemV1_candidate_iter_0.pkl")
        with open(ckpt_path_1, "wb") as f:
            pickle.dump(system_v1, f)

        system_v2 = _dummy_system("SystemV2")
        ckpt_path_2 = str(checkpoints_dir / "SystemV2_candidate_iter_1.pkl")
        with open(ckpt_path_2, "wb") as f:
            pickle.dump(system_v2, f)

        state: dict[str, Any] = {
            "target_agentic_system": _dummy_system("BrokenSystem"),
            "candidates": [
                {
                    "iteration": 0,
                    "dev_pass_rate": 0.5,
                    "total_tokens": 100,
                    "duration_seconds": 1.0,
                    "checkpoint_path": ckpt_path_1,
                },
                {
                    "iteration": 1,
                    "dev_pass_rate": 1.0,
                    "total_tokens": 80,
                    "duration_seconds": 0.8,
                    "checkpoint_path": ckpt_path_2,
                },
            ],
        }

        final_sys = finalize_best_candidate(state, code_dir=code_dir)

        assert final_sys is not None
        assert final_sys.system_name == "SystemV2"
        assert state["target_agentic_system"].system_name == "SystemV2"
        assert state["best_candidate"] is not None
        assert state["best_candidate"].get("iteration") == 1

        # Check materialized files
        assert os.path.exists(os.path.join(code_dir, "SystemV2.pkl"))
        assert os.path.exists(os.path.join(code_dir, "SystemV2.py"))

        # Check that intermediate checkpoints were cleaned up to save disk
        assert not os.path.exists(ckpt_path_1)
        assert not os.path.exists(ckpt_path_2)

    def test_finalize_best_candidate_preserves_checkpoints_when_cleanup_disabled(self, tmp_path):
        code_dir = str(tmp_path / "output")
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        system_v1 = _dummy_system("SystemPreserved")
        ckpt_path = str(checkpoints_dir / "SystemPreserved_candidate_iter_0.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(system_v1, f)

        state: dict[str, Any] = {
            "target_agentic_system": system_v1,
            "candidates": [
                {
                    "iteration": 0,
                    "dev_pass_rate": 1.0,
                    "total_tokens": 100,
                    "duration_seconds": 1.0,
                    "checkpoint_path": ckpt_path,
                }
            ],
        }

        finalize_best_candidate(state, code_dir=code_dir, cleanup_checkpoints=False)
        assert os.path.exists(ckpt_path)

    def test_finalize_best_candidate_fallback_current_system_when_no_candidates(self, tmp_path):
        code_dir = str(tmp_path / "output")
        sys = _dummy_system("FallbackSystem")
        state: dict[str, Any] = {
            "target_agentic_system": sys,
            "candidates": [],
        }

        final_sys = finalize_best_candidate(state, code_dir=code_dir)
        assert final_sys is not None
        assert final_sys.system_name == "FallbackSystem"
        assert state["best_candidate"] is None
        assert os.path.exists(os.path.join(code_dir, "FallbackSystem.pkl"))
        assert os.path.exists(os.path.join(code_dir, "FallbackSystem.py"))

    def test_finalize_best_candidate_corrupt_checkpoint_fallback(self, tmp_path):
        code_dir = str(tmp_path / "output")
        corrupt_ckpt = tmp_path / "corrupt.pkl"
        corrupt_ckpt.write_bytes(b"bad pickle bytes")

        sys = _dummy_system("SafeFallback")
        state: dict[str, Any] = {
            "target_agentic_system": sys,
            "candidates": [
                {
                    "iteration": 1,
                    "dev_pass_rate": 1.0,
                    "total_tokens": 10,
                    "duration_seconds": 0.1,
                    "checkpoint_path": str(corrupt_ckpt),
                }
            ],
        }

        final_sys = finalize_best_candidate(state, code_dir=code_dir)
        assert final_sys is not None
        assert final_sys.system_name == "SafeFallback"
        assert os.path.exists(os.path.join(code_dir, "SafeFallback.pkl"))

    def test_finalize_best_candidate_no_system_returns_none(self):
        state: dict[str, Any] = {"candidates": []}
        assert finalize_best_candidate(state) is None

    def test_test_system_saves_checkpoint_and_records_candidate(self, tmp_path):
        (tmp_path / "SimpleTask.validation.py").write_text(
            "def validate_case_1(final_state, workspace_dirs):\n"
            "    return final_state.get('status') == 'success', 'status must be success'\n"
            "VALIDATORS = {'case_1': validate_case_1}\n",
            encoding="utf-8",
        )
        spec = TaskSpec(
            name="SimpleTask",
            system_goal="Return success",
            architecture_contract=ArchitectureContract(state_schema={"query": "str", "status": "str"}),
            dev_suite=[TestCaseSpec(id="case_1", description="success", turns=[{"query": "run"}])],
        )
        sys = _dummy_system("RecordedCandidateSystem")
        state: dict[str, Any] = {
            "target_agentic_system": sys,
            "task_spec": spec,
            "task_dir": str(tmp_path),
            "messages": [],
        }

        output = run_test_system(state)
        assert "Overall: PASSED" in output
        assert "candidates" in state
        assert len(state["candidates"]) == 1

        cand = state["candidates"][0]
        assert cand.get("iteration") == 0
        assert cand.get("dev_pass_rate") == 1.0
        assert cand.get("passed_count") == 1
        assert cand.get("total_count") == 1
        ckpt_file = str(cand.get("checkpoint_path", ""))
        assert ckpt_file != ""
        assert os.path.exists(ckpt_file)

        # Unpickle to verify checkpoint integrity
        with open(ckpt_file, "rb") as f:
            restored = pickle.load(f)
        assert restored.system_name == "RecordedCandidateSystem"

    def test_design_completed_condition_triggers_finalization(self, tmp_path):
        sys = _dummy_system("DesignDoneSystem")
        state: dict[str, Any] = {
            "target_agentic_system": sys,
            "design_completed": True,
            "messages": [],
            "candidates": [
                {
                    "iteration": 1,
                    "dev_pass_rate": 1.0,
                    "passed_count": 1,
                    "total_count": 1,
                    "total_tokens": 100,
                    "duration_seconds": 0.5,
                }
            ],
        }
        res = design_completed_condition(state)
        assert res == "__end__"
        assert state["best_candidate"] is not None
        assert state["best_candidate"].get("iteration") == 1

    def test_missing_telemetry_ranked_after_known_measurements(self):
        c_known: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 5000,
            "duration_seconds": 10.0,
        }
        c_missing_tokens: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": 10.0,
        }
        # Known tokens must beat missing tokens even if missing is from a later iteration
        assert candidate_rank_key(c_known, preference="tokens") > candidate_rank_key(
            c_missing_tokens, preference="tokens"
        )
        assert select_best_candidate([c_missing_tokens, c_known], preference="tokens") == c_known

    def test_missing_runtime_ranked_after_known_runtime(self):
        c_known: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 100,
            "duration_seconds": 5.0,
        }
        c_missing_runtime: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": 100,
            "duration_seconds": None,
        }
        # Under runtime preference, known duration beats missing duration
        assert candidate_rank_key(c_known, preference="runtime") > candidate_rank_key(
            c_missing_runtime, preference="runtime"
        )
        assert select_best_candidate([c_missing_runtime, c_known], preference="runtime") == c_known

    def test_missing_usage_ranked_after_known_measurements(self):
        c_complete: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": 2000,
            "duration_seconds": 2.0,
        }
        c_incomplete: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": 2.0,
        }
        assert candidate_rank_key(c_complete, preference="tokens") > candidate_rank_key(
            c_incomplete, preference="tokens"
        )
        assert select_best_candidate([c_incomplete, c_complete], preference="tokens") == c_complete

    def test_multiple_missing_telemetry_tiebreaking(self):
        c_missing_1: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": 5.0,
        }
        c_missing_2: CandidateRecord = {
            "iteration": 2,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": 2.0,
        }
        # Both lack token telemetry -> tie on tokens -> c_missing_2 has lower duration
        assert candidate_rank_key(c_missing_2, preference="tokens") > candidate_rank_key(
            c_missing_1, preference="tokens"
        )

        # When both lack all telemetry, later iteration wins
        c_empty_early: CandidateRecord = {
            "iteration": 1,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": None,
        }
        c_empty_late: CandidateRecord = {
            "iteration": 3,
            "dev_pass_rate": 1.0,
            "total_tokens": None,
            "duration_seconds": None,
        }
        assert candidate_rank_key(c_empty_late, preference="tokens") > candidate_rank_key(
            c_empty_early, preference="tokens"
        )

    def test_record_candidate_evaluation_defaults_to_none_for_missing_telemetry(self, tmp_path):
        sys = _dummy_system("NoneTelemetryTest")
        state: dict[str, Any] = {}
        cand = record_candidate_evaluation(
            state=state,
            system=sys,
            iteration=1,
            passed_count=1,
            total_count=2,
            code_dir=str(tmp_path),
        )
        assert cand.get("total_tokens") is None
        assert cand.get("duration_seconds") is None
        assert cand.get("llm_calls") is None
