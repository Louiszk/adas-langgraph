"""Validation module for task: fever"""

from __future__ import annotations

from typing import Any
from pathlib import Path

VALIDATION_REQUIREMENTS = []


from pathlib import Path

from typing import Any

def validate_case_0_smoke_earth_orbit(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    """Validate that the system supports the claim that Earth revolves around the Sun."""
    output_dir = Path(workspace_dirs.get("output", ".")).resolve()

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    if "prediction" not in final_state:
        return False, "Assertion failed: final_state is missing the required 'prediction' field"

    prediction = final_state["prediction"]
    if prediction != "SUPPORTS":
        return (
            False,
            f"Assertion failed: expected prediction 'SUPPORTS', got {prediction!r}",
        )

    return (
        True,
        "Verification passed: prediction is exactly 'SUPPORTS' for the claim "
        f"that Earth revolves around the Sun; output workspace resolved to {output_dir}",
    )


from pathlib import Path

from typing import Any

def validate_case_1_robert_richardson(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", ".")).resolve()

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    prediction = final_state.get("prediction")
    if prediction != "NOT ENOUGH INFO":
        return (
            False,
            f"Assertion failed: expected prediction 'NOT ENOUGH INFO', got {prediction!r}",
        )

    return (
        True,
        "Verification passed: prediction is exactly 'NOT ENOUGH INFO'",
    )


from pathlib import Path

from typing import Any

def validate_case_2_kingdom_hearts(
    final_state: dict[str, Any],
    workspace_dirs: dict[str, str],
) -> tuple[bool, str]:
    output_dir = Path(workspace_dirs.get("output", "."))

    if not isinstance(final_state, dict):
        return False, "Assertion failed: final_state must be a dictionary"

    prediction = final_state.get("prediction")
    if prediction != "REFUTES":
        return (
            False,
            f"Assertion failed: expected prediction 'REFUTES', got {prediction!r}",
        )

    return True, "Verification passed: prediction is exactly 'REFUTES'"


VALIDATORS = {
    "case_0_smoke_earth_orbit": validate_case_0_smoke_earth_orbit,
    "case_1_robert_richardson": validate_case_1_robert_richardson,
    "case_2_kingdom_hearts": validate_case_2_kingdom_hearts,
}


def validate(test_case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    validator = VALIDATORS.get(test_case_id)
    if not validator:
        return False, f"No validator found for test case '{test_case_id}'"
    return validator(final_state, workspace_dirs)
