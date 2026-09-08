"""Validation module for task: gsm_hard"""

from __future__ import annotations

from typing import Any
from pathlib import Path

VALIDATION_REQUIREMENTS = []


from pathlib import Path

from typing import Any

import math

def validate_case_0_smoke_simple_addition(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir  # No output files are expected for this test case.

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    if "solution" not in final_state:
        return False, "Assertion failed: final_state is missing the required 'solution' field"

    raw_solution = final_state["solution"]
    try:
        solution = float(raw_solution)
    except (TypeError, ValueError):
        return False, f"Assertion failed: solution is not parseable as a float: {raw_solution!r}"

    if not math.isfinite(solution):
        return False, f"Assertion failed: solution must be finite, got {solution!r}"

    expected = 8.0
    tolerance = 1e-3
    if abs(solution - expected) > tolerance:
        return (
            False,
            f"Assertion failed: expected solution within {tolerance} of "
            f"{expected:.3f}, got {solution:.6f}",
        )

    return True, f"Verification passed: solution is {solution:.3f}, matching the expected total of 8.000"


import math

from pathlib import Path

from typing import Any

VALIDATION_REQUIREMENTS: list[str] = []

def validate_case_1_program_downloads(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    """Validate the total program downloads across the three stated months."""
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir  # Reserved for outputs if the target system writes any files.

    if "solution" not in final_state:
        return False, "Assertion failed: final state is missing the 'solution' field"

    try:
        solution = float(final_state["solution"])
    except (TypeError, ValueError):
        return False, "Assertion failed: 'solution' could not be parsed as a float"

    if not math.isfinite(solution):
        return False, "Assertion failed: 'solution' must be a finite float"

    expected = 3_244_047.100
    tolerance = 1e-3

    if abs(solution - expected) > tolerance:
        return (
            False,
            f"Assertion failed: expected solution {expected:.3f} "
            f"within {tolerance}, got {solution:.6f}",
        )

    return (
        True,
        f"Verification passed: solution {solution:.3f} matches the expected "
        f"three-month total of {expected:.3f}",
    )


from pathlib import Path

from typing import Any

import math

def validate_case_2_ship_travel(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))
    _ = output_dir  # No filesystem outputs are expected for this case.

    if "solution" not in final_state:
        return False, "Assertion failed: final_state is missing the required 'solution' field"

    raw_solution = final_state["solution"]
    if isinstance(raw_solution, bool):
        return False, "Assertion failed: solution must be numeric, not boolean"

    try:
        solution = float(raw_solution)
    except (TypeError, ValueError):
        return False, "Assertion failed: solution could not be parsed as a float"

    if not math.isfinite(solution):
        return False, "Assertion failed: solution must be finite"

    # Sailing from 1 PM to 4 PM takes 3 hours. At 10 mph, the ship travels
    # 30 miles. Returning at 6 mph therefore takes 30 / 6 = 5 hours.
    expected_solution = (4.0 - 1.0) * 10.0 / 6.0

    if abs(solution - expected_solution) > 1e-3:
        return (
            False,
            f"Assertion failed: expected solution approximately {expected_solution:.3f}, "
            f"but got {solution:.6f}",
        )

    return (
        True,
        f"Verification passed: return travel time is {solution:.3f} hours",
    )


VALIDATORS = {
    "case_0_smoke_simple_addition": validate_case_0_smoke_simple_addition,
    "case_1_program_downloads": validate_case_1_program_downloads,
    "case_2_ship_travel": validate_case_2_ship_travel,
}


def validate(test_case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    validator = VALIDATORS.get(test_case_id)
    if not validator:
        return False, f"No validator found for test case '{test_case_id}'"
    return validator(final_state, workspace_dirs)
