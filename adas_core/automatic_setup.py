from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adas_core.chat_model import ChatModel, usage_scope
from adas_core.exceptions import FixtureExecutionError
from adas_core.helpers import normalize_fixture_path, normalize_future_imports, safe_write_text
from adas_core.logging_config import get_logger
from adas_core.markdown_parser import find_code_blocks
from adas_core.task_spec import (
    CustomFixtureSpec,
    DatabaseFixtureSpec,
    FileFixtureSpec,
    MCPFixtureSpec,
    MockServiceFixtureSpec,
    TaskSpec,
)
from meta_system.config import validation_model, validation_wrapper

logger = get_logger("adas_core.automatic_setup")


@dataclass
class SetupGenerationResult:
    """Result of running automatic setup for a task."""

    created_files: list[Path]
    preflight_script_path: Path
    discovered_packages: list[str]
    summary: str


def extract_code_block(content: str) -> str:
    """Extract code block content using find_code_blocks or return raw stripped content."""
    blocks = find_code_blocks(content)
    if blocks:
        return str(blocks[0]["content"]).strip()
    return content.strip()


def extract_setup_requirements(code: str) -> list[str]:
    """Extract SETUP_REQUIREMENTS list from generated Python code if declared."""
    try:
        parsed = ast.parse(code)
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SETUP_REQUIREMENTS":
                        val = ast.literal_eval(node.value)
                        if isinstance(val, list):
                            return [str(item) for item in val]
    except Exception as e:
        logger.debug(f"Could not parse SETUP_REQUIREMENTS from code: {e}")
    return []


BASE_GENERATION_SYSTEM_PROMPT = """You are generating automated setup and fixture code for AI agent evaluation.

MANDATORY PACKAGE DECLARATION RULE:
If ANY third-party or non-standard library packages are needed by your code (e.g. neo4j, psycopg2-binary, duckdb, fastapi, uvicorn, mcp, langchain-mcp-adapters, faker, pandas, httpx), you must declare them at the top of the file:
SETUP_REQUIREMENTS = ["package1", "package2"]
If no third-party packages are needed, declare:
SETUP_REQUIREMENTS = []

CODE CONSTRAINTS:
- Output valid, complete, runnable Python code only.
- Do not use conversational filler, markdown explanations, or commentary outside the code.
- Never hardcode absolute host paths, sandbox paths, service URLs, or connection strings. Use the function arguments and declared environment variables supplied at runtime.
"""


def _has_declared_fixtures(task_spec: TaskSpec) -> bool:
    """Return True if the TaskSpec declares any test fixtures."""
    tf = task_spec.test_fixtures
    return bool(tf.files or tf.databases or tf.mcps or tf.mock_services or tf.custom_fixtures)


# normalize_fixture_path is imported from adas_core.helpers and re-exported


class AutomaticSetup:
    """Ahead-of-time synthesizer for test fixtures and preflight scripts using setup_model."""

    def __init__(self, llm: ChatModel | None = None) -> None:
        self.llm = llm or ChatModel(
            provider=validation_wrapper,
            model=validation_model,
            name="AutomaticSetup",
            is_meta=True,
        )

    def _invoke_setup_model(self, messages: list[SystemMessage | HumanMessage]) -> Any:
        """Invoke setup generation under its durable telemetry scope."""
        with usage_scope(system="meta", node="automatic_setup"):
            return self.llm.invoke(messages)

    def _build_system_prompt(self, task_instructions: str) -> str:
        """Compose the base generation prompt with task-specific instructions."""
        return f"{BASE_GENERATION_SYSTEM_PROMPT}\n{task_instructions.strip()}"

    def _format_task_context(self, task_spec: TaskSpec) -> str:
        dev_cases = "\n".join([f"- {tc.id}: {tc.description} (Turns: {len(tc.turns)})" for tc in task_spec.dev_suite])
        return (
            f"Task Name: {task_spec.name}\n"
            f"System Goal: {task_spec.system_goal}\n"
            f"Execution Mode: {task_spec.architecture_contract.execution_mode}\n"
            f"Required Tools: {[t.name for t in task_spec.architecture_contract.required_tools]}\n"
            f"Dev Test Scenarios:\n{dev_cases}\n"
        )

    def generate_file_script(self, task_spec: TaskSpec, fixture: FileFixtureSpec) -> tuple[str, list[str]]:
        """Prompt setup_model to generate a Python script that synthesizes test file(s)."""
        instructions = (
            "TASK: Procedural File Generation\n"
            "Write a self-contained Python script to procedurally generate the requested test files.\n"
            "The script must define a function:\n"
            "`def generate_files(output_path: Path) -> None:`\n"
            "Rules:\n"
            "1. If count == 1: write the single file directly to `output_path`.\n"
            "2. If count > 1: treat `output_path` as a directory, create it if needed, and write the specified number of files inside.\n"
            "3. Include realistic column headers, data distributions, and formatting.\n"
            "4. Model edge cases as realistic data-level anomalies (null/empty values, type inconsistencies, date format variations, outliers, or missing optional fields)."
            " Preserve valid top-level syntax and container structure (e.g. all records in a JSON list must be objects/dicts, not bare strings) so standard library loaders and iterators do not crash on parse unless unparseable syntax is explicitly requested."
        )
        system_prompt = self._build_system_prompt(instructions)
        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"File Path: {fixture.path}\n"
            f"Target File Count: {fixture.count}\n"
            f"Content Type: {fixture.content_type}\n"
            f"Description & Requirements:\n{fixture.description}\n\n"
            "Write the complete Python generator script:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        reqs = extract_setup_requirements(code)
        return code, reqs

    def generate_database_seed_script(self, task_spec: TaskSpec, fixture: DatabaseFixtureSpec) -> tuple[str, list[str]]:
        """Prompt setup_model to generate a Python database seeding script."""
        instructions = (
            "TASK: Database Seeding\n"
            "Write a self-contained Python script to seed test data into the specified database.\n"
            "The script must define a function:\n"
            "`def seed_database(workspace_dirs: dict[str, str]) -> None:`\n"
            "Rules:\n"
            "1. Read connection credentials from environment variables where specified.\n"
            "2. Populate realistic tables, collections, or graph nodes honoring target counts."
        )
        system_prompt = self._build_system_prompt(instructions)
        conn_info = (
            f"Connection Environment Variables: {fixture.connection_env}"
            if fixture.connection_env
            else f"Local File Path: {fixture.file_path}"
        )
        count_info = (
            f"Target Records/Nodes Count: {fixture.count}" if fixture.count else "Target Count: Realistic seed dataset"
        )
        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"Database Name: {fixture.name}\n"
            f"Engine Type: {fixture.db_type}\n"
            f"{conn_info}\n"
            f"{count_info}\n"
            f"Schema & Data Requirements:\n{fixture.description}\n\n"
            "Write the complete Python seeding script:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        reqs = extract_setup_requirements(code)
        return code, reqs

    def generate_mcp_server_script(self, task_spec: TaskSpec, fixture: MCPFixtureSpec) -> tuple[str, list[str]]:
        """Prompt setup_model to generate a mock FastMCP server script."""
        instructions = (
            "TASK: Mock FastMCP Server\n"
            "Write a complete, executable mock MCP server script using the FastMCP framework (`from mcp.server.fastmcp import FastMCP`).\n"
            "Implement realistic mock tools using `@mcp.tool()` based on the fixture requirements.\n"
            "The script must bind exactly to host `127.0.0.1`, the declared port, and the declared endpoint path.\n"
            "It must stay in the foreground and run directly with `python fixtures/mock_<name>.py` without interactive setup.\n"
            "Include `if __name__ == '__main__': mcp.run(...)` configured for the specified Streamable HTTP port and endpoint path."
        )
        system_prompt = self._build_system_prompt(instructions)
        transport_info = f"Transport: {fixture.transport}"
        transport_info += f", Host: 127.0.0.1, Port: {fixture.port}, Endpoint Path: {fixture.endpoint_path}"

        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"MCP Server Name: {fixture.name}\n"
            f"{transport_info}\n"
            f"Requirements & Tools to Mock:\n{fixture.description}\n\n"
            "Write the complete FastMCP server script:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        reqs = extract_setup_requirements(code)
        if "mcp" not in reqs:
            reqs.append("mcp")
        return code, reqs

    def generate_mock_service_script(
        self, task_spec: TaskSpec, fixture: MockServiceFixtureSpec
    ) -> tuple[str, list[str]]:
        """Prompt setup_model to generate a lightweight FastAPI mock server."""
        instructions = (
            "TASK: Mock REST HTTP Service\n"
            "Write a self-contained FastAPI mock server script.\n"
            "Implement realistic endpoints and mock data according to the description.\n"
            "Include `if __name__ == '__main__': uvicorn.run(app, host='127.0.0.1', port=...)`.\n"
            "Use exactly the declared port, stay in the foreground, and run directly with `python fixtures/mock_<name>.py` without interactive setup."
        )
        system_prompt = self._build_system_prompt(instructions)
        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"Service Name: {fixture.name}\n"
            f"Port: {fixture.port}\n"
            f"Base URL Env: {fixture.base_url_env}\n"
            f"Endpoints & Response Requirements:\n{fixture.description}\n\n"
            "Write the complete FastAPI mock server script:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        reqs = extract_setup_requirements(code)
        if "fastapi" not in reqs:
            reqs.append("fastapi")
        if "uvicorn" not in reqs:
            reqs.append("uvicorn")
        return code, reqs

    def generate_custom_fixture_script(self, task_spec: TaskSpec, fixture: CustomFixtureSpec) -> tuple[str, list[str]]:
        """Prompt setup_model to generate a custom environment setup Python script."""
        instructions = (
            "TASK: Custom Environment Setup\n"
            "Write a Python script defining `setup_environment(workspace_dirs: dict[str, str]) -> None`\n"
            "that materializes the declared filesystem artifact (e.g. a git repository or CLI fixture). "
            "Do not start services or background processes."
        )
        instructions += (
            f"\nArtifact Relative Path: {fixture.path}\n"
            f"The script must create that exact relative path beneath workspace_dirs['ADAS_INPUT_DIR']."
        )
        system_prompt = self._build_system_prompt(instructions)

        path_info = (
            f"Artifact Path: {fixture.path}\n"
            f'Artifact Location Instruction: Create that exact relative path beneath workspace_dirs["ADAS_INPUT_DIR"].\n'
        )

        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"Fixture Name: {fixture.name}\n"
            f"{path_info}"
            f"Requirements:\n{fixture.description}\n\n"
            "Write the complete setup script:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        reqs = extract_setup_requirements(code)
        return code, reqs

    def generate_preflight_script(self, task_spec: TaskSpec, all_discovered_packages: list[str]) -> str:
        """Prompt setup_model to generate a preflight environment verification module."""
        instructions = (
            "TASK: Preflight Environment Verification\n"
            "Write a Python module `preflight.py` that verifies the environment before running tests.\n"
            "The module must define a function:\n"
            "`def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:`\n"
            "Rules for check_environment:\n"
            "1. Check that required environment variables / API keys exist (log names only, NEVER secrets).\n"
            "2. Verify declared file and embedded-database inputs only through workspace_dirs and the runtime ADAS_INPUT_DIR environment; search input directories recursively using rglob, as fixtures may have relative subdirectories. Never hardcode a host or sandbox path.\n"
            "3. For a declared HTTP, MCP, or external database resource, use only its declared environment variable when checking configuration. Do not hardcode localhost URLs, ports, endpoint paths, or connection strings: these may be replaced by a runtime resource profile.\n"
            "4. Return (True, 'Environment verified') on success, or (False, error_description) on failure."
        )
        system_prompt = self._build_system_prompt(instructions)
        api_keys = [
            f"{k.env_var} ({k.description})" for k in task_spec.resource_manifest.available_api_keys if not k.optional
        ]
        user_prompt = (
            f"{self._format_task_context(task_spec)}\n"
            f"Required API Keys: {api_keys}\n"
            f"Discovered Packages: {all_discovered_packages}\n\n"
            "Write the complete preflight.py module:"
        )

        response = self._invoke_setup_model([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        code = normalize_future_imports(extract_code_block(str(response.content)))
        try:
            compile(code, "<preflight>", "exec")
        except SyntaxError as exc:
            logger.warning("Generated preflight script failed syntax validation: %s", exc)
        return code

    def generate_all(
        self,
        task_spec: TaskSpec,
        task_dir: Path | str,
        execute_generated_code: bool = True,
    ) -> SetupGenerationResult:
        """Synthesize all declared fixtures and the preflight script for a TaskSpec."""
        root = Path(task_dir)
        setup_scripts_dir = root / "setup_scripts"
        fixtures_dir = root / "fixtures"
        has_fixtures = _has_declared_fixtures(task_spec)
        if has_fixtures:
            setup_scripts_dir.mkdir(parents=True, exist_ok=True)
            fixtures_dir.mkdir(parents=True, exist_ok=True)

        created_files: list[Path] = []
        discovered_packages: set[str] = set(task_spec.required_packages)

        # 1. Generate file fixtures procedurally via script
        for file_fix in task_spec.test_fixtures.files:
            code, reqs = self.generate_file_script(task_spec, file_fix)
            clean_rel = normalize_fixture_path(file_fix.path)
            clean_name = clean_rel.replace("/", "_").replace("\\", "_").replace(".", "_")
            generator_script_dest = setup_scripts_dir / f"generate_{clean_name}.py"
            safe_write_text(generator_script_dest, code, root_dir=setup_scripts_dir)
            created_files.append(generator_script_dest)
            discovered_packages.update(reqs)

        # 2. Generate database seed scripts
        for db_fix in task_spec.test_fixtures.databases:
            db_script_dest = setup_scripts_dir / f"seed_{db_fix.name}.py"
            code, reqs = self.generate_database_seed_script(task_spec, db_fix)
            safe_write_text(db_script_dest, code, root_dir=setup_scripts_dir)
            created_files.append(db_script_dest)
            discovered_packages.update(reqs)
            logger.info(f"Generated database seed script: {db_script_dest}")

        # 3. Generate MCP mock server scripts
        for mcp_fix in task_spec.test_fixtures.mcps:
            mcp_script_dest = fixtures_dir / f"mock_{mcp_fix.name}.py"
            code, reqs = self.generate_mcp_server_script(task_spec, mcp_fix)
            safe_write_text(mcp_script_dest, code, root_dir=fixtures_dir)
            created_files.append(mcp_script_dest)
            discovered_packages.update(reqs)
            logger.info(f"Generated MCP server script: {mcp_script_dest}")

        # 4. Generate mock service scripts
        for mock_fix in task_spec.test_fixtures.mock_services:
            mock_dest = fixtures_dir / f"mock_{mock_fix.name}.py"
            code, reqs = self.generate_mock_service_script(task_spec, mock_fix)
            safe_write_text(mock_dest, code, root_dir=fixtures_dir)
            created_files.append(mock_dest)
            discovered_packages.update(reqs)
            logger.info(f"Generated mock service script: {mock_dest}")

        # 5. Generate custom fixture scripts
        for custom_fix in task_spec.test_fixtures.custom_fixtures:
            cust_dest = setup_scripts_dir / f"setup_{custom_fix.name}.py"
            code, reqs = self.generate_custom_fixture_script(task_spec, custom_fix)
            safe_write_text(cust_dest, code, root_dir=setup_scripts_dir)
            created_files.append(cust_dest)
            discovered_packages.update(reqs)
            logger.info(f"Generated custom fixture script: {cust_dest}")

        # 6. Generate preflight.py
        all_packages = sorted(discovered_packages)
        preflight_code = self.generate_preflight_script(task_spec, all_packages)
        preflight_path = root / "preflight.py"
        safe_write_text(preflight_path, preflight_code, root_dir=root)
        created_files.append(preflight_path)
        logger.info(f"Generated preflight verification module: {preflight_path}")

        if execute_generated_code:
            self.execute_generated_artifacts(task_spec, root, created_files)

        summary = (
            f"Generated {len(created_files) - 1} fixture setup artifact(s) in {setup_scripts_dir} / {fixtures_dir} "
            f"and preflight script at {preflight_path}. "
            f"Discovered {len(all_packages)} required package(s): {all_packages}"
        )
        return SetupGenerationResult(
            created_files=created_files,
            preflight_script_path=preflight_path,
            discovered_packages=all_packages,
            summary=summary,
        )

    def execute_generated_artifacts(
        self,
        task_spec: TaskSpec,
        task_dir: Path | str,
        created_files: list[Path] | None = None,
    ) -> None:
        """Execute frozen file, embedded-database, and custom fixture scripts.

        Callers that need to install discovered dependencies first can generate
        scripts with ``execute_generated_code=False`` and invoke this method
        afterwards.  This method deliberately does not start long-running MCP
        or HTTP service fixtures; the test harness owns their lifecycle.
        """
        root = Path(task_dir)
        setup_scripts_dir = root / "setup_scripts"
        fixtures_dir = root / "fixtures"

        for file_fix in task_spec.test_fixtures.files:
            clean_rel = normalize_fixture_path(file_fix.path)
            clean_name = clean_rel.replace("/", "_").replace("\\", "_").replace(".", "_")
            script = setup_scripts_dir / f"generate_{clean_name}.py"
            if not script.is_file():
                script = fixtures_dir / f"generate_{clean_name}.py"
            dest = fixtures_dir / clean_rel
            if file_fix.count == 1:
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest.mkdir(parents=True, exist_ok=True)
            try:
                ns: dict[str, Any] = {"Path": Path}
                exec(script.read_text(encoding="utf-8"), ns)
                generator = ns.get("generate_files")
                if not callable(generator):
                    raise ValueError("missing generate_files(output_path) function")
                generator(dest)
                if created_files is not None:
                    created_files.append(dest)
                logger.info(f"Materialized file fixture via generator script: {dest} (count: {file_fix.count})")
            except Exception as exc:
                raise FixtureExecutionError(f"Error executing file generator for {file_fix.path}: {exc}") from exc

        for db_fix in task_spec.test_fixtures.databases:
            if db_fix.db_type not in ("sqlite", "duckdb"):
                continue
            script = setup_scripts_dir / f"seed_{db_fix.name}.py"
            if not script.is_file():
                script = fixtures_dir / f"seed_{db_fix.name}.py"
            try:
                ns = {"Path": Path}
                exec(script.read_text(encoding="utf-8"), ns)
                seed_database = ns.get("seed_database")
                if not callable(seed_database):
                    raise FixtureExecutionError(
                        f"Error seeding embedded database {db_fix.name}: missing seed_database(workspace_dirs) function"
                    )
                seed_database({"ADAS_INPUT_DIR": str(fixtures_dir)})
            except FixtureExecutionError:
                raise
            except Exception as exc:
                raise FixtureExecutionError(f"Error seeding embedded database {db_fix.name}: {exc}") from exc

        for custom_fix in task_spec.test_fixtures.custom_fixtures:
            script = setup_scripts_dir / f"setup_{custom_fix.name}.py"
            if not script.is_file():
                script = fixtures_dir / f"setup_{custom_fix.name}.py"
            try:
                ns = {"Path": Path}
                exec(script.read_text(encoding="utf-8"), ns)
                setup_environment = ns.get("setup_environment")
                if not callable(setup_environment):
                    raise FixtureExecutionError(
                        f"Error executing custom fixture {custom_fix.name}: missing setup_environment(workspace_dirs) function"
                    )
                setup_environment({"ADAS_INPUT_DIR": str(fixtures_dir), "ADAS_WORKSPACE_DIR": str(root)})
            except FixtureExecutionError:
                raise
            except Exception as exc:
                raise FixtureExecutionError(f"Error executing custom fixture {custom_fix.name}: {exc}") from exc


def ensure_automatic_setup(
    task_spec: TaskSpec,
    task_dir: Path | str,
    force: bool = False,
    llm: ChatModel | None = None,
) -> SetupGenerationResult | None:
    """Ensure task fixtures and preflight script exist; invoke AutomaticSetup if missing.

    Args:
        task_spec: The parsed TaskSpec.
        task_dir: The directory where fixtures/ and preflight.py should live.
        force: If True, regenerates fixtures even if they already exist.
        llm: Optional custom LLM instance (defaults to setup_model).

    Returns:
        SetupGenerationResult if setup was run, or None if already present.
    """
    root = Path(task_dir)
    fixtures_dir = root / "fixtures"
    setup_scripts_dir = root / "setup_scripts"
    preflight_path = root / "preflight.py"

    has_fixtures = _has_declared_fixtures(task_spec)
    fixtures_ready = (fixtures_dir.exists() or setup_scripts_dir.exists()) if has_fixtures else True

    if not force and preflight_path.exists() and fixtures_ready:
        logger.info(f"Task setup already complete in {root}. Skipping automatic setup.")
        return None

    logger.info(f"Automatic setup needed for '{task_spec.name}': synthesizing in {root}...")
    setup = AutomaticSetup(llm=llm)
    return setup.generate_all(task_spec, root)
