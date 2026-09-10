"""Ahead-of-time LLM-authored validation module generation."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adas_core.chat_model import ChatModel, usage_scope
from adas_core.environment import is_provisioning_script
from adas_core.helpers import normalize_future_imports, safe_write_text, sanitize_test_id
from adas_core.logging_config import get_logger
from adas_core.markdown_parser import find_code_blocks
from adas_core.task_spec import TaskSpec, TestCaseSpec
from meta_system.config import validation_model, validation_wrapper

logger = get_logger("adas_core.automatic_validation")

VALIDATION_GENERATOR_VERSION = "1"
_SETUP_MANIFEST_FILENAME = "setup_manifest.json"


def _normalized_sha256(path: Path) -> str:
    """Hash a text artifact independent of the checkout's newline convention."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _task_spec_sha256(task_spec: TaskSpec) -> str:
    """Hash the semantic TaskSpec content used to generate a validation module."""
    payload = json.dumps(task_spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fixture_generator_hashes(root: Path) -> dict[str, str]:
    """Return hashes for every fixture generator that influences validation generation."""
    hashes: dict[str, str] = {}
    candidate_paths: list[Path] = []
    for candidate_dir in (root / "setup_scripts", root / "fixtures"):
        if candidate_dir.is_dir():
            candidate_paths.extend(candidate_dir.glob("*.py"))
    for path in sorted(candidate_paths):
        if is_provisioning_script(path):
            hashes[path.relative_to(root).as_posix()] = _normalized_sha256(path)
    return hashes


def write_validation_manifest(task_spec: TaskSpec, root: Path, validation_file: Path) -> Path:
    """Update the validation section of the task's shared setup manifest."""
    root = root.resolve()
    validation_file = validation_file.resolve()
    if not validation_file.is_relative_to(root):
        raise ValueError(f"Validation file {validation_file} is outside task directory {root}.")
    validation = {
        "schema_version": "1.0",
        "task_name": task_spec.name,
        "task_spec_hash": _task_spec_sha256(task_spec),
        "fixture_generator_hashes": _fixture_generator_hashes(root),
        "generator_version": VALIDATION_GENERATOR_VERSION,
        "validator_file": validation_file.relative_to(root).as_posix(),
        "validator_hash": _normalized_sha256(validation_file),
    }
    manifest_path = root / _SETUP_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Setup manifest {manifest_path} must contain a JSON object.")
    manifest["validation"] = validation
    safe_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", root_dir=root)
    return manifest_path


def is_validation_manifest_current(task_spec: TaskSpec, task_dir: Path | str) -> bool:
    """Return whether the frozen validator and all its generation inputs still match."""
    root = Path(task_dir).resolve()
    manifest_path = root / _SETUP_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = manifest.get("validation")
        if not isinstance(validation, dict):
            return False
        validator_name = validation.get("validator_file")
        validator_hash = validation.get("validator_hash")
        if not isinstance(validator_name, str) or not isinstance(validator_hash, str):
            return False
        validator_file = (root / validator_name).resolve()
        if not validator_file.is_relative_to(root) or not validator_file.is_file():
            return False
        return (
            validation.get("schema_version") == "1.0"
            and validation.get("task_name") == task_spec.name
            and validation.get("task_spec_hash") == _task_spec_sha256(task_spec)
            and validation.get("fixture_generator_hashes") == _fixture_generator_hashes(root)
            and validation.get("generator_version") == VALIDATION_GENERATOR_VERSION
            and validator_hash == _normalized_sha256(validator_file)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


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
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == "VALIDATION_REQUIREMENTS" for target in node.targets
                ):
                    value = ast.literal_eval(node.value)
                    return [str(item) for item in value] if isinstance(value, list) else []
            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "VALIDATION_REQUIREMENTS"
                    and node.value is not None
                ):
                    value = ast.literal_eval(node.value)
                    return [str(item) for item in value] if isinstance(value, list) else []
    except (SyntaxError, ValueError, TypeError) as exc:
        logger.debug("Could not parse VALIDATION_REQUIREMENTS: %r", exc)
    return []


CASE_VALIDATION_SYSTEM_PROMPT = """You are generating automated validation test code for an AI agentic system.

MANDATORY RULES:
1. PACKAGE DECLARATION:
   If any third-party packages are needed for validation (e.g. pandas, neo4j, duckdb, pytest), declare at top:
   VALIDATION_REQUIREMENTS = ["pkg1", "pkg2"]
   Otherwise:
   VALIDATION_REQUIREMENTS = []

2. DEDICATED VALIDATOR FUNCTION:
   You must define a dedicated validator function named:
   `def validate_{clean_id}(final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:`
   - It must return `(True, "Verification passed: <details>")` if the test case criteria are satisfied.
   - It must return `(False, "Assertion failed: <details>")` if any condition is violated.
   - Ground checks in domain truth using the provided fixture generation code for this test case.

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

5. SCOPE ASSERTIONS STRICTLY TO DECLARED OUTPUTS & GROUND TRUTH:
   - Only check outputs declared in `expected_outputs` or explicitly requested by the test case turns.
   - Ground assertions strictly in the exact columns, schemas, and planted values revealed by the fixture generator code.
   - Avoid brittle text matching or arbitrary synonym searching. Use `LLMJudge` for qualitative aspects.

6. CODE FORMAT:
   - Output valid, complete, runnable Python code only inside a single ```python code block.
   - Do not use conversational filler or explanations outside the code block.
   - Do not use wildcard imports (e.g. `from module import *`). Always import specific names explicitly (e.g. `from math import sqrt`).
"""


def filter_fixture_generators_for_case(
    task_spec: TaskSpec,
    test_case: TestCaseSpec,
    fixture_generators: dict[str, str] | None,
) -> dict[str, str]:
    """Filter discovered fixture generators to only those relevant to the specific test case using exact lookup."""
    if not fixture_generators:
        return {}

    # If fixture_ids is None, all fixtures in the suite are visible to the case
    if test_case.fixture_ids is None:
        return dict(fixture_generators)

    # If fixture_ids is an empty list, no fixtures are mounted
    if not test_case.fixture_ids:
        return {}

    target_ids = set(test_case.fixture_ids)
    relevant_scripts: dict[str, str] = {}

    # Collect exact valid candidate filenames/keys for each target fixture ID
    candidate_keys: set[str] = set()
    for fid in target_ids:
        candidate_keys.update(task_spec.test_fixtures.get_script_filenames_for_fixture(fid))

    for script_name, code in fixture_generators.items():
        if script_name in candidate_keys:
            relevant_scripts[script_name] = code

    return relevant_scripts


def _extract_target_names(target: ast.AST) -> list[str]:
    """Recursively extract variable names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_target_names(target.value)
    return []


class _SnippetNamespaceTransformer(ast.NodeTransformer):
    """Namespace top-level helper functions, classes, variables, and imports to prevent cross-case collisions."""

    def __init__(self, names_to_rename: set[str], prefix: str) -> None:
        self.names = names_to_rename
        self.prefix = prefix

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.names:
            node.id = f"{self.prefix}_{node.id}"
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self.names:
            node.arg = f"{self.prefix}_{node.arg}"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in self.names:
            node.name = f"{self.prefix}_{node.name}"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        if node.name in self.names:
            node.name = f"{self.prefix}_{node.name}"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.name in self.names:
            node.name = f"{self.prefix}_{node.name}"
        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST | list[ast.AST]:
        extra_assigns: list[ast.AST] = []
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".")[0]
            if bound_name in self.names:
                if alias.asname is not None or "." not in alias.name:
                    alias.asname = f"{self.prefix}_{bound_name}"
                else:
                    top_name = alias.name.split(".")[0]
                    assign = ast.Assign(
                        targets=[ast.Name(id=f"{self.prefix}_{top_name}", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="__import__", ctx=ast.Load()),
                            args=[ast.Constant(value=top_name)],
                            keywords=[],
                        ),
                    )
                    ast.copy_location(assign, node)
                    ast.fix_missing_locations(assign)
                    extra_assigns.append(assign)
        if extra_assigns:
            return [node, *extra_assigns]
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module == "__future__":
            return node
        for alias in node.names:
            if alias.name == "*":
                continue
            bound_name = alias.asname or alias.name
            if bound_name in self.names:
                alias.asname = f"{self.prefix}_{bound_name}"
        return node


def assemble_validation_module(
    task_spec: TaskSpec,
    case_snippets: list[tuple[TestCaseSpec, str]],
    requirements: list[str],
) -> str:
    """Assemble individual case validator snippets into a single, cohesive validation module."""
    header = [
        f'"""Validation module for task: {task_spec.name}"""',
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "from pathlib import Path",
        "",
        f"VALIDATION_REQUIREMENTS = {requirements!r}",
        "",
    ]

    body_blocks: list[str] = []
    validators_map_entries: list[str] = []
    seen_sanitized: dict[str, str] = {}

    for case, code in case_snippets:
        clean_id = sanitize_test_id(case.id)
        if clean_id in seen_sanitized:
            raise ValueError(
                f"Duplicate sanitized test case id '{clean_id}': "
                f"'{case.id}' collides with '{seen_sanitized[clean_id]}'."
            )
        seen_sanitized[clean_id] = case.id
        target_func_name = f"validate_{clean_id}"
        validators_map_entries.append(f'    "{case.id}": {target_func_name},')

        try:
            parsed = ast.parse(code)
        except SyntaxError:
            body_blocks.append(code)
            continue

        excluded_names = {"VALIDATION_REQUIREMENTS", "VALIDATORS", "validate", target_func_name}

        def _is_excluded(name: str) -> bool:
            if name in excluded_names:
                return True
            if name.startswith("__") and name.endswith("__"):
                return True
            return False

        names_to_rename: set[str] = set()
        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not _is_excluded(node.name):
                    names_to_rename.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _extract_target_names(target):
                        if not _is_excluded(name):
                            names_to_rename.add(name)
            elif isinstance(node, ast.AnnAssign):
                for name in _extract_target_names(node.target):
                    if not _is_excluded(name):
                        names_to_rename.add(name)
            elif isinstance(node, ast.AugAssign):
                for name in _extract_target_names(node.target):
                    if not _is_excluded(name):
                        names_to_rename.add(name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound_name = alias.asname or alias.name.split(".")[0]
                    if not _is_excluded(bound_name):
                        names_to_rename.add(bound_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        mod_name = f"{'.' * node.level}{node.module or ''}"
                        raise ValueError(
                            f"Wildcard import 'from {mod_name} import *' is not permitted in test case "
                            f"snippet '{case.id}'. Explicitly import needed names."
                        )
                    bound_name = alias.asname or alias.name
                    if not _is_excluded(bound_name):
                        names_to_rename.add(bound_name)

        kept_nodes: list[ast.stmt] = []
        for node in parsed.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.Assign):
                if any(
                    name in ("VALIDATION_REQUIREMENTS", "VALIDATORS")
                    for target in node.targets
                    for name in _extract_target_names(target)
                ):
                    continue
            if isinstance(node, ast.AnnAssign):
                if any(
                    name in ("VALIDATION_REQUIREMENTS", "VALIDATORS") for name in _extract_target_names(node.target)
                ):
                    continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "validate":
                continue
            kept_nodes.append(node)

        parsed.body = kept_nodes
        if names_to_rename:
            transformer = _SnippetNamespaceTransformer(names_to_rename, prefix=f"_{clean_id}")
            parsed = transformer.visit(parsed)
            ast.fix_missing_locations(parsed)

        body_blocks.append(ast.unparse(parsed))

    dispatcher = [
        "VALIDATORS = {",
        "\n".join(validators_map_entries),
        "}",
        "",
        "",
        "def validate(test_case_id: str, final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]:",
        "    validator = VALIDATORS.get(test_case_id)",
        "    if not validator:",
        "        return False, f\"No validator found for test case '{test_case_id}'\"",
        "    return validator(final_state, workspace_dirs)",
        "",
    ]

    full_code = "\n".join(header) + "\n\n" + "\n\n\n".join(body_blocks) + "\n\n\n" + "\n".join(dispatcher)
    normalized = normalize_future_imports(full_code)
    compile(normalized, "<validation>", "exec")
    return normalized


def discover_fixture_generators(base_path: Path | str | None) -> dict[str, str]:
    """Search candidate fixture and setup directories for fixture generator/seed scripts and return their code."""
    if not base_path:
        return {}
    target = Path(base_path)
    candidate_groups = [
        [target / "setup_scripts", target / "fixtures"],
        [target.parent / "setup_scripts", target.parent / "fixtures"],
        [target] if target.is_dir() and target.name in ("setup_scripts", "fixtures") else [],
        [Path("task_setup/setup_scripts"), Path("task_setup/fixtures")],
        [Path("setup_scripts"), Path("fixtures")],
    ]
    generators: dict[str, str] = {}
    for group in candidate_groups:
        for cdir in group:
            if cdir and cdir.exists() and cdir.is_dir():
                for script in sorted(cdir.glob("*.py")):
                    if is_provisioning_script(script) and script.name not in generators:
                        try:
                            code = script.read_text(encoding="utf-8")
                            generators[script.name] = code if len(code) <= 8000 else code[:8000] + "\n# ... (truncated)"
                        except OSError as exc:
                            logger.debug("Could not read fixture generator %s: %r", script, exc)
        if generators:
            break
    return generators


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
    def _format_case_context(
        task_spec: TaskSpec,
        test_case: TestCaseSpec,
        case_fixture_generators: dict[str, str] | None = None,
    ) -> str:
        clean_id = sanitize_test_id(test_case.id)
        context_parts = [
            f"Task Name: {task_spec.name}",
            f"System Goal: {task_spec.system_goal}",
            f"Execution Mode: {task_spec.architecture_contract.execution_mode}",
            f"State Schema: {task_spec.architecture_contract.state_schema}",
            "",
            "TARGET TEST CASE TO VALIDATE:",
            f"- ID: {test_case.id}",
            f"- Required Function: validate_{clean_id}(final_state: dict[str, Any], workspace_dirs: dict[str, str]) -> tuple[bool, str]",
            f"- Description: {test_case.description}",
            f"- Turns: {test_case.turns}",
            f"- Expected Outputs: {test_case.expected_outputs}",
            f"- Deterministic Criteria: {test_case.deterministic_criteria}",
            f"- LLM Judge Needed: {test_case.llm_judge_needed}",
            f"- Judge Criteria: {test_case.judge_criteria}",
            f"- Modalities: {test_case.modalities}",
            f"- Active Fixture IDs: {test_case.fixture_ids}",
        ]

        if case_fixture_generators:
            generator_blocks = [
                f"### Fixture Generator Script: {name}\n```python\n{code.strip()}\n```"
                for name, code in sorted(case_fixture_generators.items())
            ]
            context_parts.extend(
                [
                    "",
                    "FIXTURE GENERATION CODE:\n"
                    "The following Python scripts generate the input fixtures mounted for this test case. "
                    "Inspect their exact column names, schemas, formulas, and planted edge cases to write grounded, accurate assertions:",
                    "\n\n".join(generator_blocks),
                ]
            )
        else:
            context_parts.extend(
                [
                    "",
                    "FIXTURE GENERATION CODE:\n"
                    "No input fixtures are mounted for this test case. Rely on final_state or outputs produced during execution.",
                ]
            )

        return "\n".join(context_parts)

    def generate_case_validator(
        self,
        task_spec: TaskSpec,
        test_case: TestCaseSpec,
        fixture_generators: dict[str, str] | None = None,
    ) -> tuple[str, list[str]]:
        """Generate and verify validation code for a single test case."""
        case_generators = filter_fixture_generators_for_case(task_spec, test_case, fixture_generators)
        clean_id = sanitize_test_id(test_case.id)
        expected_fn = f"validate_{clean_id}"

        prompt = CASE_VALIDATION_SYSTEM_PROMPT.replace("{clean_id}", clean_id)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=self._format_case_context(task_spec, test_case, case_generators)),
        ]
        try:
            with usage_scope(system="meta", node="automatic_validation"):
                response = self.llm.invoke(messages)
            code = normalize_future_imports(extract_code_block(str(response.content)))
            parsed = ast.parse(code)
        except Exception as exc:
            raise RuntimeError(f"Validation generation for '{test_case.id}' failed: {exc!r}") from exc

        function_names = {
            node.name for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if expected_fn not in function_names:
            raise ValueError(
                f"Generated validation code for '{test_case.id}' is missing expected function '{expected_fn}'."
            )

        requirements = extract_validation_requirements(code)
        return code, requirements

    def generate_validation_module(
        self,
        task_spec: TaskSpec,
        test_cases: list[TestCaseSpec] | None = None,
        fixture_generators: dict[str, str] | None = None,
    ) -> tuple[str, list[str]]:
        """Generate and assemble source for all supplied test cases, invoking the model once per case."""
        cases = test_cases if test_cases is not None else task_spec.dev_suite
        if not cases:
            raise ValueError(f"TaskSpec '{task_spec.name}' has no test cases to generate validation for.")

        all_reqs: set[str] = set()
        case_snippets: list[tuple[TestCaseSpec, str]] = []

        for case in cases:
            code, reqs = self.generate_case_validator(task_spec, case, fixture_generators=fixture_generators)
            all_reqs.update(reqs)
            case_snippets.append((case, code))

        assembled_code = assemble_validation_module(task_spec, case_snippets, sorted(list(all_reqs)))
        return assembled_code, sorted(list(all_reqs))

    def generate_all(
        self,
        task_spec: TaskSpec,
        target_path_or_dir: Path | str,
        test_cases: list[TestCaseSpec] | None = None,
        fixture_generators: dict[str, str] | None = None,
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

        resolved_generators = fixture_generators
        if resolved_generators is None:
            resolved_generators = discover_fixture_generators(destination)

        cases = test_cases if test_cases is not None else task_spec.dev_suite
        code, requirements = self.generate_validation_module(task_spec, cases, fixture_generators=resolved_generators)
        root_dir = destination if (destination.is_dir() or not destination.suffix) else destination.parent
        safe_write_text(validation_file, code, root_dir=root_dir)
        write_validation_manifest(task_spec, root_dir, validation_file)
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
    fixture_generators: dict[str, str] | None = None,
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
    if existing_file is not None and not force and is_validation_manifest_current(task_spec, root):
        logger.info("Task validation module already exists at %s. Skipping generation.", existing_file)
        return None
    resolved_generators = fixture_generators or discover_fixture_generators(root)
    return AutomaticValidation(llm=llm).generate_all(
        task_spec, expected_files[0], fixture_generators=resolved_generators
    )


__all__ = [
    "CASE_VALIDATION_SYSTEM_PROMPT",
    "AutomaticValidation",
    "ValidationGenerationResult",
    "assemble_validation_module",
    "discover_fixture_generators",
    "ensure_automatic_validation",
    "extract_validation_requirements",
    "filter_fixture_generators_for_case",
    "is_validation_manifest_current",
    "load_validation_module",
    "sanitize_test_id",
    "write_validation_manifest",
]
