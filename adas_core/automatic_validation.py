"""Ahead-of-time LLM-authored validation module generation."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adas_core.chat_model import ChatModel, usage_scope
from adas_core.markdown_parser import find_code_blocks
from adas_core.helpers import sanitize_test_id
from adas_core.logging_config import get_logger
from adas_core.task_spec import TaskSpec, TestCaseSpec
from meta_system.config import validation_model, validation_wrapper

logger = get_logger("adas_core.automatic_validation")


@dataclass
class ValidationGenerationResult:
    """The frozen validation artifact and its declared dependencies."""

    validation_file_path: Path
    required_packages: list[str]
    summary: str


def extract_code_block(content: str) -> str:
    """Return the first Markdown code block, or the stripped response itself."""
    blocks = find_code_blocks(content)
    return str(blocks[0]["content"]).strip() if blocks else content.strip()


def extract_validation_requirements(code: str) -> list[str]:
    """Read a literal VALIDATION_REQUIREMENTS declaration from validator source."""
    try:
        for node in ast.parse(code).body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "VALIDATION_REQUIREMENTS" for target in node.targets):
                value = ast.literal_eval(node.value)
                return [str(item) for item in value] if isinstance(value, list) else []
    except (SyntaxError, ValueError, TypeError) as exc:
        logger.debug("Could not parse VALIDATION_REQUIREMENTS: %r", exc)
    return []


BASE_VALIDATION_SYSTEM_PROMPT = """You are generating automated validation test code for an AI agentic system.

MANDATORY RULES:
1. PACKAGE DECLARATION:
   If any third-party packages are needed for validation (e.g. pandas, neo4j, duckdb, pytest), declare at top:
   VALIDATION_REQUIREMENTS = ["pkg1", "pkg2"]
   Otherwise:
   VALIDATION_REQUIREMENTS = []

2. DEDICATED VALIDATOR PER TEST CASE:
   For each test case in the suite, you MUST define a dedicated validator function named:
   `def validate_<test_id>(final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:`
   - It must return `(True, "Verification passed: <details>")` if the test case criteria are satisfied.
   - It must return `(False, "Assertion failed: <details>")` if any condition is violated.

3. FILESYSTEM & WORKSPACE CONTRACT:
   - Output files written by the target system are located in `workspace_dirs["output"]`.
   - Input fixture files are located in `workspace_dirs["input"]`.
   - Always resolve paths cleanly: `output_dir = Path(workspace_dirs.get("output", "."))`.

4. SEMANTIC & MULTIMODAL EVALUATION VIA LLMJudge:
   - LLMJudge from `adas_core.judge` is the SOLE evaluation SDK for qualitative assessments.
   - Never instantiate external model APIs directly in validator code.
   - Usage:
     `from adas_core.judge import LLMJudge`
     `judge = LLMJudge(model=judge_model_override, provider=judge_provider_override)`  # or LLMJudge()
   - Write clear, tailored evaluation prompts directly containing task context, criteria, and outputs to evaluate:
     `eval_result = judge.evaluate(prompt=f"Task: ... Criteria: ... Output: {final_state}")`
     `if not eval_result.is_pass: return False, f"LLM Judge rejected: {eval_result.reasoning}"`
   - Arbitrary Structured Extraction:
     If you need domain-specific structured metrics, scores, or flags to pass into downstream assertions:
     define a Pydantic model (e.g. `class ExtractedMetrics(BaseModel): ...`) and pass `schema=ExtractedMetrics`
     to `judge.evaluate(prompt=..., schema=ExtractedMetrics)`.
   - For multimodal/vision evaluation (e.g. when modalities includes 'vision' or output images like .png/.jpg are produced):
     Collect the image path(s) and pass them via the `images` parameter:
     `eval_result = judge.evaluate(prompt=evaluation_prompt, images=[str(output_dir / "chart.png")])`
     LLMJudge automatically encodes images and inspects them using vision capabilities.

5. DISPATCHER & REGISTRY:
   At the bottom of the module, define a dictionary `VALIDATORS` mapping the original test case ID to its validator function:
   `VALIDATORS = {"<case_id>": validate_<clean_id>, ...}`
   and a dispatcher function:
   `def validate(test_case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:`
   `    validator = VALIDATORS.get(test_case_id)`
   `    if not validator:`
   `        return False, f"No validator found for test case '{test_case_id}'"`
   `    return validator(final_state, workspace_dirs)`

6. CODE FORMAT:
   - Output valid, complete, runnable Python code only inside a single ```python code block.
   - Do not use conversational filler or explanations outside the code block.
"""


class AutomaticValidation:
    """Synthesize a complete validation module using the configured validation model."""

    def __init__(self, llm: ChatModel | None = None) -> None:
        self.llm = llm or ChatModel(
            provider=validation_wrapper,
            model=validation_model,
            name="AutomaticValidation",
            is_meta=True,
        )

    @staticmethod
    def _format_task_context(task_spec: TaskSpec, test_cases: list[TestCaseSpec]) -> str:
        cases = [
            {
                "id": case.id,
                "description": case.description,
                "turns": case.turns,
                "expected_outputs": case.expected_outputs,
                "deterministic_criteria": case.deterministic_criteria,
                "llm_judge_needed": case.llm_judge_needed,
                "judge_criteria": case.judge_criteria,
                "judge_model": case.judge_model,
                "judge_provider": case.judge_provider,
                "modalities": case.modalities,
            }
            for case in test_cases
        ]
        return (
            f"Task name: {task_spec.name}\n"
            f"System goal: {task_spec.system_goal}\n"
            f"Execution mode: {task_spec.architecture_contract.execution_mode}\n"
            f"State schema: {task_spec.architecture_contract.state_schema}\n"
            f"Fixtures: {task_spec.test_fixtures.model_dump(mode='json')}\n"
            f"Test cases: {cases}"
        )

    def generate_validation_module(
        self, task_spec: TaskSpec, test_cases: list[TestCaseSpec] | None = None
    ) -> tuple[str, list[str]]:
        """Generate and verify source for the supplied visible development cases."""
        cases = test_cases if test_cases is not None else task_spec.dev_suite
        if not cases:
            raise ValueError(f"TaskSpec '{task_spec.name}' has no test cases to generate validation for.")

        messages = [
            SystemMessage(content=BASE_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=self._format_task_context(task_spec, cases)),
        ]
        try:
            with usage_scope(system="meta", node="automatic_validation"):
                response = self.llm.invoke(messages)
            code = extract_code_block(str(response.content))
            parsed = ast.parse(code)
        except Exception as exc:
            raise RuntimeError(f"Validation generation via model failed: {exc!r}") from exc

        function_names = {
            node.name for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for case in cases:
            expected = f"validate_{sanitize_test_id(case.id)}"
            if expected not in function_names:
                raise ValueError(f"Generated validation code is missing expected function '{expected}'.")
        if "validate" not in function_names or not any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "VALIDATORS" for target in node.targets)
            for node in parsed.body
        ):
            raise ValueError(
                "Generated validation code is missing the required VALIDATORS registry or validate dispatcher."
            )
        return code, extract_validation_requirements(code)

    def generate_all(
        self,
        task_spec: TaskSpec,
        target_path_or_dir: Path | str,
        test_cases: list[TestCaseSpec] | None = None,
    ) -> ValidationGenerationResult:
        """Generate and write a frozen validation module using the validation model."""
        destination = Path(target_path_or_dir)
        if destination.is_dir() or not destination.suffix:
            destination.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", task_spec.name)
            validation_file = destination / f"{safe_name}.validation.py"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            validation_file = destination

        cases = test_cases if test_cases is not None else task_spec.dev_suite
        code, requirements = self.generate_validation_module(task_spec, cases)
        validation_file.write_text(code, encoding="utf-8")
        return ValidationGenerationResult(
            validation_file_path=validation_file,
            required_packages=requirements,
            summary=(
                f"Synthesized validation module for '{task_spec.name}' with {len(cases)} test case(s) "
                f"at {validation_file}. Required packages: {requirements}"
            ),
        )


def load_validation_module(validation_path: Path | str) -> Any:
    """Dynamically load an already-frozen validation module."""
    path = Path(validation_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Validation module not found: {path}")
    module_name = f"adas_validation_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def ensure_automatic_validation(
    task_spec: TaskSpec,
    task_dir: Path | str,
    force: bool = False,
    llm: ChatModel | None = None,
) -> ValidationGenerationResult | None:
    """Create a validator only when explicitly invoked by the validation stage."""
    root = Path(task_dir)
    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", task_spec.name)
    expected_files = [
        root / f"{safe_name}.validation.py",
        root / f"{task_spec.name}.validation.py",
        root / "validation.py",
    ]
    existing_file = next((path for path in expected_files if path.exists()), None)
    if existing_file is not None and not force:
        logger.info("Task validation module already exists at %s. Skipping generation.", existing_file)
        return None
    return AutomaticValidation(llm=llm).generate_all(task_spec, expected_files[0])


__all__ = [
    "AutomaticValidation",
    "ValidationGenerationResult",
    "ensure_automatic_validation",
    "extract_validation_requirements",
    "load_validation_module",
    "sanitize_test_id",
]
