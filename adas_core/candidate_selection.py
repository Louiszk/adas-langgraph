from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import dill as pickle

from adas_core.environment import SANDBOX_GENERATED_SYSTEMS_DIR
from adas_core.logging_config import get_logger
from adas_core.materialize import materialize_system
from adas_core.virtual_agentic_system import VirtualAgenticSystem

DEFAULT_OPTIMIZATION_METRIC: Literal["tokens", "runtime"] = "tokens"
DEFAULT_CLEANUP_CHECKPOINTS: bool = True

if TYPE_CHECKING:
    from meta_system.state import MetaState

logger = get_logger("adas_core.candidate_selection")


class CandidateRecord(TypedDict, total=False):
    iteration: int
    dev_pass_rate: float
    passed_count: int
    total_count: int
    total_tokens: int | None
    duration_seconds: float | None
    llm_calls: int | None
    checkpoint_path: str
    system_snapshot: bytes


def record_candidate(state: MetaState | dict[str, Any] | Any, candidate: CandidateRecord) -> None:
    """Append a candidate record to state['candidates'], initializing the list if needed."""
    if "candidates" not in state or not isinstance(state["candidates"], list):
        state["candidates"] = []
    state["candidates"].append(candidate)


def save_candidate_checkpoint(
    system: VirtualAgenticSystem,
    iteration: int,
    code_dir: str | None = None,
) -> str:
    """Save an iteration-scoped checkpoint of a candidate system to disk.

    Returns the absolute path to the saved checkpoint, or empty string on error.
    """
    target_dir = code_dir or SANDBOX_GENERATED_SYSTEMS_DIR
    checkpoints_dir = os.path.join(target_dir, "checkpoints")
    try:
        os.makedirs(checkpoints_dir, exist_ok=True)
        escaped_name = getattr(system, "escaped_name", "agentic_system")
        checkpoint_path = os.path.join(checkpoints_dir, f"{escaped_name}_candidate_iter_{iteration}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(system, f)
        return checkpoint_path
    except Exception as exc:
        logger.error("Error during candidate checkpoint saving: %r", exc)
        return ""


def record_candidate_evaluation(
    state: MetaState | dict[str, Any] | Any,
    system: VirtualAgenticSystem,
    iteration: int,
    passed_count: int,
    total_count: int,
    total_tokens: int | None = None,
    duration_seconds: float | None = None,
    llm_calls: int | None = None,
    code_dir: str | None = None,
) -> CandidateRecord:
    """Checkpoint system to disk and record a structured CandidateRecord into state['candidates']."""
    checkpoint_path = save_candidate_checkpoint(system, iteration=iteration, code_dir=code_dir)
    pass_rate = (passed_count / total_count) if total_count > 0 else 0.0
    candidate: CandidateRecord = {
        "iteration": iteration,
        "dev_pass_rate": pass_rate,
        "passed_count": passed_count,
        "total_count": total_count,
        "total_tokens": total_tokens,
        "duration_seconds": duration_seconds,
        "llm_calls": llm_calls,
        "checkpoint_path": checkpoint_path,
    }
    try:
        candidate["system_snapshot"] = pickle.dumps(system)
    except Exception:
        pass
    record_candidate(state, candidate)
    return candidate


def candidate_rank_key(
    candidate: CandidateRecord,
    preference: Literal["tokens", "runtime"] = "tokens",
) -> tuple[float, float, float, int]:
    """Rank key for candidates: higher is better.

    Order of priority:
    1. dev_pass_rate (higher is better)
    2. Efficiency metrics based on preference:
       - if preference == "runtime": -duration_seconds, then -total_tokens
       - if preference == "tokens": -total_tokens, then -duration_seconds
       Missing or incomplete telemetry is ranked after all known measurements.
    3. iteration (higher is better -> prefer later iterations)
    """
    pass_rate = candidate.get("dev_pass_rate")
    if pass_rate is None:
        total = candidate.get("total_count", 0)
        passed = candidate.get("passed_count", 0)
        pass_rate = (passed / total) if total and total > 0 else 0.0
    pass_rate = float(pass_rate or 0.0)

    raw_tokens = candidate.get("total_tokens")
    if raw_tokens is None:
        raw_tokens = candidate.get("tokens")

    if raw_tokens is not None:
        token_score = -float(raw_tokens)
    else:
        token_score = -float("inf")

    raw_duration = candidate.get("duration_seconds")
    if raw_duration is None:
        raw_duration = candidate.get("duration", candidate.get("latency"))

    if raw_duration is not None:
        duration_score = -float(raw_duration)
    else:
        duration_score = -float("inf")

    iteration = int(candidate.get("iteration", 0) or 0)

    if preference == "runtime":
        return (pass_rate, duration_score, token_score, iteration)
    else:
        return (pass_rate, token_score, duration_score, iteration)


def select_best_candidate(
    candidates: Sequence[CandidateRecord],
    preference: Literal["tokens", "runtime"] = "tokens",
) -> CandidateRecord | None:
    """Select the top-ranked candidate from a sequence of candidate records."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: candidate_rank_key(c, preference=preference))


def finalize_best_candidate(
    state: MetaState | dict[str, Any] | Any,
    code_dir: str | None = None,
    cleanup_checkpoints: bool | None = None,
    preference: Literal["tokens", "runtime"] | str | None = None,
) -> VirtualAgenticSystem | None:
    """Select the best candidate system, restore its checkpoint, and materialize it.

    If candidate records exist in state["candidates"], selects the best one according
    to candidate_rank_key (pass_rate > token/latency efficiency > iteration).
    Restores the system from checkpoint or snapshot, updates state["target_agentic_system"],
    saves the final pickle at <code_dir>/<escaped_name>.pkl, and materializes code files.

    Cleans up intermediate candidate checkpoints to prevent disk bloat (configurable via
    CLEANUP_CHECKPOINTS_ON_FINALIZATION or the cleanup_checkpoints argument).

    If no candidates exist, falls back to materializing the current state["target_agentic_system"].
    """
    target_dir = code_dir or state.get("generated_systems_dir") or SANDBOX_GENERATED_SYSTEMS_DIR
    raw_candidates = state.get("candidates")
    candidates: list[CandidateRecord] = raw_candidates if isinstance(raw_candidates, list) else []
    current_system: VirtualAgenticSystem | None = state.get("target_agentic_system")

    # Resolve optimization preference (tokens vs runtime)
    pref = preference or state.get("optimization_metric") or DEFAULT_OPTIMIZATION_METRIC
    ranking_pref: Literal["tokens", "runtime"] = "runtime" if pref == "runtime" else "tokens"

    best_candidate = select_best_candidate(candidates, preference=ranking_pref) if candidates else None
    state["best_candidate"] = best_candidate

    selected_system: VirtualAgenticSystem | None = None

    if best_candidate:
        checkpoint_path = best_candidate.get("checkpoint_path", "")
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "rb") as f:
                    selected_system = pickle.load(f)
                logger.info(
                    "Restored best candidate from checkpoint %s (iteration %s, dev_pass_rate=%.2f, pref=%s)",
                    os.path.basename(checkpoint_path),
                    best_candidate.get("iteration"),
                    best_candidate.get("dev_pass_rate", 0.0),
                    ranking_pref,
                )
            except Exception as exc:
                logger.error("Failed to load best candidate checkpoint %s: %r", checkpoint_path, exc)

        if selected_system is None:
            raw_snapshot = best_candidate.get("system_snapshot")
            if raw_snapshot is not None and isinstance(raw_snapshot, (bytes, bytearray)):
                try:
                    selected_system = pickle.loads(raw_snapshot)
                    logger.info("Restored best candidate from in-memory snapshot")
                except Exception as exc:
                    logger.error("Failed to deserialize candidate system snapshot: %r", exc)

    if selected_system is None:
        if current_system is not None:
            logger.info("Using current target_agentic_system for finalization (no candidate checkpoint restored)")
            selected_system = current_system
        else:
            logger.warning("No target_agentic_system available to finalize.")
            return None

    state["target_agentic_system"] = selected_system

    # Save final system pickle and materialize code
    try:
        os.makedirs(target_dir, exist_ok=True)
        escaped_name = getattr(selected_system, "escaped_name", "agentic_system")
        final_system_path = os.path.join(target_dir, f"{escaped_name}.pkl")

        checkpoint_path = best_candidate.get("checkpoint_path", "") if best_candidate else ""
        if (
            checkpoint_path
            and os.path.exists(checkpoint_path)
            and os.path.abspath(checkpoint_path) != os.path.abspath(final_system_path)
        ):
            shutil.copy2(checkpoint_path, final_system_path)
        else:
            with open(final_system_path, "wb") as f:
                pickle.dump(selected_system, f)

        materialize_system(selected_system, output_dir=target_dir)
        logger.info("Successfully finalized system '%s' in %s", escaped_name, target_dir)

        # Cleanup intermediate candidate checkpoints if enabled
        should_cleanup = (
            cleanup_checkpoints
            if cleanup_checkpoints is not None
            else (
                state.get("cleanup_checkpoints")
                if state.get("cleanup_checkpoints") is not None
                else DEFAULT_CLEANUP_CHECKPOINTS
            )
        )
        if should_cleanup:
            for cand in candidates:
                ckpt = cand.get("checkpoint_path", "")
                if ckpt and os.path.exists(ckpt) and os.path.abspath(ckpt) != os.path.abspath(final_system_path):
                    try:
                        os.remove(ckpt)
                        logger.debug("Cleaned up intermediate checkpoint: %s", ckpt)
                    except Exception as exc:
                        logger.debug("Could not remove intermediate checkpoint %s: %r", ckpt, exc)

            checkpoints_dir = os.path.join(target_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                try:
                    if not os.listdir(checkpoints_dir):
                        os.rmdir(checkpoints_dir)
                except Exception:
                    pass

    except Exception as exc:
        logger.error("Error during final system materialization: %r", exc)
        raise

    return selected_system
