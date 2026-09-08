"""Validation module for task: mmlu_pro"""

from __future__ import annotations

from typing import Any
from pathlib import Path

VALIDATION_REQUIREMENTS = []


from pathlib import Path

from typing import Any

def validate_case_0_smoke_binary_digits(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    if "solution" not in final_state:
        return False, "Assertion failed: final_state is missing the required 'solution' field"

    if final_state["solution"] != "B":
        return (
            False,
            f"Assertion failed: expected solution 'B', got {final_state['solution']!r}",
        )

    return True, "Verification passed: the solution correctly identifies two distinct values for one bit"


from typing import Any

from pathlib import Path

def validate_case_1_sha1_digest(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir  # No filesystem outputs are declared for this test case.

    expected_solution = "C"
    actual_solution = final_state.get("solution")

    if actual_solution != expected_solution:
        return (
            False,
            f"Assertion failed: expected solution '{expected_solution}', "
            f"got {actual_solution!r}",
        )

    return True, "Verification passed: SHA-1 digest size correctly identified as option C."


from typing import Any

from pathlib import Path

def validate_case_2_compiler_code_generation(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir  # No filesystem outputs are declared for this test case.

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    solution = final_state.get("solution")
    if solution != "I":
        return (
            False,
            f"Assertion failed: expected solution 'I', got {solution!r}",
        )

    return True, "Verification passed: final_state solution is exactly 'I'"


VALIDATORS = {
    "case_0_smoke_binary_digits": validate_case_0_smoke_binary_digits,
    "case_1_sha1_digest": validate_case_1_sha1_digest,
    "case_2_compiler_code_generation": validate_case_2_compiler_code_generation,
}


def validate(test_case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    validator = VALIDATORS.get(test_case_id)
    if not validator:
        return False, f"No validator found for test case '{test_case_id}'"
    return validator(final_state, workspace_dirs)
