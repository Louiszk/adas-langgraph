from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adas_core.helpers import normalize_fixture_path, sanitize_test_id, validate_identifier


class FeatureNotImplementedError(NotImplementedError, ValueError):
    """Raised when a TaskSpec requests a capability unavailable in the runtime."""


class ToolRequirement(BaseModel):
    """Specification of a tool required by the target system."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique name of the tool")
    description: str = Field(..., min_length=1, description="Purpose and usage instructions for the tool")


class ModelSpec(BaseModel):
    """Specification of an allowed model available for target system use."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="openai", description="Provider name (e.g. 'openai')")
    model_name: str = Field(..., min_length=1, description="Model identifier (e.g. 'gpt-5.6-luna')")


class PersistenceContract(BaseModel):
    """Configuration for state persistence and checkpointers."""

    model_config = ConfigDict(extra="forbid")

    checkpointer: str | None = Field(default="memory", description="Checkpointer backend (e.g. 'memory')")
    requires_thread_id: bool = Field(default=True, description="Whether invocations require a thread_id")


class ArchitectureContract(BaseModel):
    """Contract defining execution mode, state typing, and required tools."""

    model_config = ConfigDict(extra="forbid")

    # TODO: 'multi_turn' execution mode is not implemented yet in the test runner/execution harness.
    # Currently only 'single_turn' is supported; full multi-turn conversational execution with
    # checkpointer persistence is planned for a future commit.
    execution_mode: Literal["single_turn", "multi_turn"] = Field(
        default="single_turn",
        description="Single-turn execution or multi-turn conversational execution (Note: multi_turn is not implemented yet)",
    )
    state_schema: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Mapping of state attribute names to type annotations (e.g. {'messages': 'Annotated[list[AnyMessage], add_messages]'})",
    )
    persistence: PersistenceContract | None = Field(
        default=None,
        description="Checkpointer configuration for stateful/multi-turn execution (TODO: not implemented yet)",
    )
    required_tools: list[ToolRequirement] = Field(
        default_factory=list, description="List of tools the target system must provide"
    )

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, v: str) -> str:
        # TODO(multi_turn): The test runner executes only one turn.
        if v != "single_turn":
            raise FeatureNotImplementedError(
                f"NOT_IMPLEMENTED: Execution mode '{v}' is not supported by the current runtime. "
                "Only 'single_turn' is currently implemented."
            )
        return v

    @model_validator(mode="after")
    def validate_multi_turn_persistence(self) -> ArchitectureContract:
        # Execution-mode validation rejects multi_turn before this model validator runs.
        return self


class ResourceEntry(BaseModel):
    """An available resource in the environment (e.g. database, input file directory)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Resource identifier")
    type: Literal["file", "directory", "database", "api", "other"] = Field(..., description="Resource category")
    path_or_uri: str | None = Field(default=None, description="Filesystem path or URI to the resource")
    description: str = Field(default="", description="Description of the resource schema or contents")


class ApiKeyRequirement(BaseModel):
    """Declaration of an expected environment variable/API key."""

    model_config = ConfigDict(extra="forbid")

    env_var: str = Field(..., min_length=1, description="Environment variable name (e.g. 'SERPER_API_KEY')")
    description: str = Field(default="", description="Description of the service unlocked by this key")
    optional: bool = Field(default=False, description="Whether this key is optional for the task")


class ResourceManifest(BaseModel):
    """Manifest of environment resources and credentials available to the agent."""

    model_config = ConfigDict(extra="forbid")

    available_resources: list[ResourceEntry] = Field(
        default_factory=list, description="Available local files, databases, or directories"
    )
    available_api_keys: list[ApiKeyRequirement] = Field(
        default_factory=list, description="Required environment variables/API keys"
    )


class FileFixtureSpec(BaseModel):
    """Specification of test file(s) to be generated for tests."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default="",
        description="Unique fixture identifier (e.g. 'sales_csv', 'customers_json'). Defaults from path if empty.",
    )
    path: str = Field(
        ..., min_length=1, description="Relative destination path or directory (e.g. 'test.csv' or 'docs/')"
    )
    count: int = Field(
        default=1, ge=1, description="Number of files to generate (1 for single file, N for batch/folder)"
    )
    description: str = Field(default="", description="Purpose, schema, or content requirements for this file")
    content_type: Literal["text", "csv", "json", "binary"] = Field(default="text", description="File format")

    @field_validator("path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        return normalize_fixture_path(v, field_name="path")

    @model_validator(mode="after")
    def set_default_id(self) -> FileFixtureSpec:
        if not self.id:
            self.id = Path(self.path).name.replace(".", "_").replace("-", "_")
        validate_identifier(self.id, field_name="file fixture id")
        return self


class DatabaseFixtureSpec(BaseModel):
    """Specification of a test database or knowledge graph (embedded or network-based)."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", description="Unique fixture identifier. Defaults to name if empty.")
    name: str = Field(..., min_length=1, description="Database identifier (e.g. 'analytics_db', 'graph_store')")
    db_type: Literal["sqlite", "duckdb", "neo4j", "postgres", "redis", "qdrant", "custom"] = Field(
        ..., description="Database engine type"
    )
    file_path: str | None = Field(default=None, description="Path for embedded/file-based DBs (e.g. 'data/test.db')")
    connection_env: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of connection parameters to env vars, e.g. {'uri': 'NEO4J_URI', 'user': 'NEO4J_USER', 'password': 'NEO4J_PASSWORD'}",
    )
    count: int | None = Field(default=None, ge=1, description="Target number of records, rows, or nodes to seed")
    description: str = Field(default="", description="Schema, entities, tables, or graph structure to populate")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_identifier(v, field_name="database fixture name")

    @field_validator("file_path")
    @classmethod
    def validate_db_file_path(cls, v: str | None) -> str | None:
        if v is not None:
            return normalize_fixture_path(v, field_name="file_path")
        return v

    @model_validator(mode="after")
    def validate_db_configuration(self) -> DatabaseFixtureSpec:
        if not self.id:
            self.id = self.name
        validate_identifier(self.id, field_name="database fixture id")
        if self.db_type in ("sqlite", "duckdb") and not self.file_path:
            self.file_path = f"data/{self.name}.{self.db_type}"
        if self.file_path is not None:
            self.file_path = normalize_fixture_path(self.file_path, field_name="file_path")
        return self


class MCPFixtureSpec(BaseModel):
    """Specification for a Streamable HTTP Model Context Protocol server fixture."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", description="Unique fixture identifier. Defaults to name if empty.")
    name: str = Field(..., min_length=1, description="MCP server identifier (e.g. 'github_mcp', 'filesystem_mcp')")
    transport: Literal["streamable-http"] = Field(
        default="streamable-http",
        description="MCP transport protocol. Streamable HTTP is the only supported protocol.",
    )
    port: int | None = Field(default=None, description="Port number for the Streamable HTTP MCP server")
    endpoint_path: str = Field(
        default="/mcp", description="HTTP endpoint path for Streamable HTTP (defaults to '/mcp')"
    )
    url_env: str = Field(
        default="MCP_SERVER_URL", description="Env var exposing the server URL to the agent for HTTP transports"
    )
    description: str = Field(
        default="", description="Tools, resources, and simulated behaviors this MCP server provides"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_identifier(v, field_name="mcp fixture name")

    @model_validator(mode="after")
    def set_default_id(self) -> MCPFixtureSpec:
        if not self.id:
            self.id = self.name
        validate_identifier(self.id, field_name="mcp fixture id")
        return self


class MockServiceFixtureSpec(BaseModel):
    """Specification of a mock HTTP service / API for tools to query."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", description="Unique fixture identifier. Defaults to name if empty.")
    name: str = Field(..., min_length=1, description="Service name (e.g. 'mock_weather_api')")
    port: int = Field(default=8000, description="Local port for the mock server")
    base_url_env: str = Field(
        default="MOCK_API_BASE_URL", description="Env var exposing the mock server URL to the agent"
    )
    description: str = Field(default="", description="Endpoints, routes, and response behavior to mock")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_identifier(v, field_name="mock service fixture name")

    @model_validator(mode="after")
    def set_default_id(self) -> MockServiceFixtureSpec:
        if not self.id:
            self.id = self.name
        validate_identifier(self.id, field_name="mock service fixture id")
        return self


class CustomFixtureSpec(BaseModel):
    """Custom fixture that materializes one filesystem artifact for case-local use.

    Custom fixtures are artifact-only. They must not start services or require a
    runtime lifecycle; use the typed MCP or mock-service fixtures once those
    process lifecycles are implemented.
    """

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", description="Unique fixture identifier. Defaults to name if empty.")
    name: str = Field(..., min_length=1, description="Fixture identifier")
    path: str = Field(
        ...,
        min_length=1,
        description="Relative file or directory artifact produced by custom setup (e.g. 'repo/' or 'config.yaml')",
    )
    description: str = Field(..., min_length=1, description="Description of the custom setup requirements")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_identifier(v, field_name="custom fixture name")

    @field_validator("path")
    @classmethod
    def validate_custom_path(cls, v: str) -> str:
        return normalize_fixture_path(v, field_name="path")

    @model_validator(mode="after")
    def set_default_id(self) -> CustomFixtureSpec:
        if not self.id:
            self.id = self.name
        validate_identifier(self.id, field_name="custom fixture id")
        return self


class TestFixturesSpec(BaseModel):
    """Declarative specification of all test fixtures."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    files: list[FileFixtureSpec] = Field(default_factory=list, description="File fixtures to generate")
    databases: list[DatabaseFixtureSpec] = Field(default_factory=list, description="Database fixtures to generate/seed")
    mcps: list[MCPFixtureSpec] = Field(default_factory=list, description="Streamable HTTP MCP server fixtures")
    mock_services: list[MockServiceFixtureSpec] = Field(
        default_factory=list, description="Mock HTTP services to run during tests"
    )
    custom_fixtures: list[CustomFixtureSpec] = Field(
        default_factory=list, description="Custom scripts that materialize declared filesystem artifacts"
    )

    @model_validator(mode="after")
    def validate_unique_fixture_ids(self) -> TestFixturesSpec:
        """Ensure all declared fixture IDs across all categories are globally unique."""
        seen: set[str] = set()
        for category, items in [
            ("files", self.files),
            ("databases", self.databases),
            ("mcps", self.mcps),
            ("mock_services", self.mock_services),
            ("custom_fixtures", self.custom_fixtures),
        ]:
            for item in items:
                if item.id in seen:
                    raise ValueError(f"Duplicate fixture id '{item.id}' in test_fixtures ({category}).")
                seen.add(item.id)
        return self

    @model_validator(mode="after")
    def reject_unimplemented_process_fixtures(self) -> TestFixturesSpec:
        # TODO(proper-fixtures): Add per-case process lifecycle management for
        # Streamable HTTP MCP servers and mock HTTP services before enabling them.
        # Until then, accepting either kind would produce misleading test results.
        if self.mcps or self.mock_services:
            raise FeatureNotImplementedError(
                "NOT_IMPLEMENTED: MCP and mock HTTP service fixtures require process lifecycle management "
                "and are not supported by the current runtime."
            )
        return self

    def all_fixture_ids(self) -> set[str]:
        """Return the set of all declared fixture IDs across all categories."""
        ids: set[str] = set()
        for f in self.files:
            ids.add(f.id)
        for d in self.databases:
            ids.add(d.id)
        for m in self.mcps:
            ids.add(m.id)
        for s in self.mock_services:
            ids.add(s.id)
        for c in self.custom_fixtures:
            ids.add(c.id)
        return ids

    def get_file_paths_for_fixture_ids(self, fixture_ids: list[str] | None) -> list[str] | None:
        """Return relative file paths matching the requested fixture IDs, or None if all should be included."""
        if fixture_ids is None:
            return None
        target_ids = set(fixture_ids)
        # Normalize target IDs where applicable so querying with raw fixture path works seamlessly
        normalized_targets = set(target_ids)
        for tid in target_ids:
            try:
                normalized_targets.add(normalize_fixture_path(tid))
            except ValueError:
                pass

        paths: list[str] = []

        for f in self.files:
            if f.id in normalized_targets or f.path in normalized_targets:
                paths.append(normalize_fixture_path(f.path))

        for db in self.databases:
            if (db.id in normalized_targets or db.name in normalized_targets) and db.file_path:
                paths.append(normalize_fixture_path(db.file_path))

        for cf in self.custom_fixtures:
            if (cf.id in normalized_targets or cf.name in normalized_targets) and cf.path:
                paths.append(normalize_fixture_path(cf.path))

        return list(dict.fromkeys(paths))

    def get_all_file_paths(self) -> list[str]:
        """Return the relative file paths of all declared filesystem artifacts."""
        paths: list[str] = []
        for f in self.files:
            paths.append(normalize_fixture_path(f.path))
        for db in self.databases:
            if db.file_path:
                paths.append(normalize_fixture_path(db.file_path))
        for cf in self.custom_fixtures:
            if cf.path:
                paths.append(normalize_fixture_path(cf.path))
        return list(dict.fromkeys(paths))

    def get_fixture_by_id(self, fixture_id: str) -> Any | None:
        """Find and return any fixture matching the given ID."""
        for f in self.files:
            if f.id == fixture_id:
                return f
        for d in self.databases:
            if d.id == fixture_id:
                return d
        for m in self.mcps:
            if m.id == fixture_id:
                return m
        for s in self.mock_services:
            if s.id == fixture_id:
                return s
        for c in self.custom_fixtures:
            if c.id == fixture_id:
                return c
        return None

    def get_script_filenames_for_fixture(self, fixture_id: str) -> list[str]:
        """Return the exact deterministic script filenames associated with a fixture ID."""
        names: list[str] = [fixture_id, f"{fixture_id}.py"]
        fix = self.get_fixture_by_id(fixture_id)
        if fix is None:
            return names

        if isinstance(fix, FileFixtureSpec):
            clean_rel = normalize_fixture_path(fix.path)
            clean_name = clean_rel.replace("/", "_").replace("\\", "_").replace(".", "_")
            names.extend([f"generate_{fix.id}.py", f"generate_{clean_name}.py"])
        elif isinstance(fix, DatabaseFixtureSpec):
            names.extend([f"seed_{fix.id}.py", f"seed_{fix.name}.py"])
        elif isinstance(fix, (MCPFixtureSpec, MockServiceFixtureSpec)):
            names.extend([f"mock_{fix.id}.py", f"mock_{fix.name}.py"])
        elif isinstance(fix, CustomFixtureSpec):
            names.extend([f"setup_{fix.id}.py", f"setup_{fix.name}.py"])

        return list(dict.fromkeys(names))


class TestCaseSpec(BaseModel):
    """Specification for a single test scenario (single-turn or multi-turn)."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique test case identifier (e.g. 'case_1_basic_math')")
    description: str = Field(..., min_length=1, description="Description of the test scenario and edge cases")
    fixture_ids: list[str] | None = Field(
        default=None,
        description="Optional list of fixture IDs (files, databases, mock services, MCPs) provisioned for this test case. If None, all fixtures are provisioned.",
    )
    turns: list[dict[str, Any]] = Field(
        ..., min_length=1, description="Sequence of input state dictionaries (one dict per turn)"
    )
    expected_outputs: list[str] = Field(
        default_factory=list, description="Expected output artifact paths or state keys"
    )
    deterministic_criteria: str = Field(
        default="", description="Plain-English description of deterministic conditions code must assert"
    )
    llm_judge_needed: bool = Field(default=False, description="Whether qualitative LLM judgment is required")
    judge_criteria: str | None = Field(
        default=None, description="Detailed rubric and criteria for LLMJudge if llm_judge_needed is True"
    )
    judge_model: str | None = Field(
        default=None, description="Optional model override for LLMJudge (defaults to validation_model)"
    )
    judge_provider: str | None = Field(
        default=None, description="Optional provider override for LLMJudge (defaults to validation_wrapper)"
    )
    modalities: list[Literal["text", "vision"]] = Field(
        default_factory=lambda: ["text"],
        description="Expected output modalities for evaluation, e.g. ['text', 'vision']",
    )

    @field_validator("id")
    @classmethod
    def validate_test_case_id(cls, v: str) -> str:
        return validate_identifier(v, field_name="test case id")

    @field_validator("turns")
    @classmethod
    def validate_turns_non_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("Test case must contain at least one turn dictionary.")
        return v

    @model_validator(mode="after")
    def validate_judge_criteria_present_if_needed(self) -> TestCaseSpec:
        if self.llm_judge_needed and not (self.judge_criteria and self.judge_criteria.strip()):
            raise ValueError(f"TestCase '{self.id}' sets llm_judge_needed=True but judge_criteria is empty.")
        if self.judge_model is not None and not self.judge_model.strip():
            raise ValueError(f"TestCase '{self.id}' specifies an empty judge_model.")
        if self.judge_provider is not None and not self.judge_provider.strip():
            raise ValueError(f"TestCase '{self.id}' specifies an empty judge_provider.")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError(f"TestCase '{self.id}' contains duplicate modalities.")
        if "vision" in self.modalities and not self.llm_judge_needed:
            raise ValueError(f"TestCase '{self.id}' requires vision evaluation but llm_judge_needed is False.")
        if self.fixture_ids is not None and len(set(self.fixture_ids)) != len(self.fixture_ids):
            raise ValueError(f"TestCase '{self.id}' contains duplicate fixture_ids.")
        return self


SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0",)


class TaskSpec(BaseModel):
    """Complete, versioned specification for a target agentic system task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", description="Specification schema version")
    name: str = Field(..., min_length=1, description="Name of the target system to design")
    system_goal: str = Field(..., min_length=1, description="Primary goal and problem statement for the Meta-Agent")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported schema_version '{v}'. Supported versions: {list(SUPPORTED_SCHEMA_VERSIONS)}")
        return v

    @field_validator("name")
    @classmethod
    def validate_task_spec_name(cls, v: str) -> str:
        return validate_identifier(v, field_name="TaskSpec name")

    architecture_contract: ArchitectureContract = Field(..., description="Execution mode, schema, and persistence")
    available_models: list[ModelSpec] = Field(
        default_factory=lambda: [ModelSpec(provider="openai", model_name="gpt-5.6-luna")],
        description="Allowed models available for target system nodes within ChatModel.",
    )
    resource_manifest: ResourceManifest = Field(
        default_factory=ResourceManifest, description="Available resources and API keys"
    )
    required_packages: list[str] = Field(
        default_factory=list,
        description="Python packages to install upfront in the sandbox (e.g. ['neo4j', 'fastapi', 'psycopg2-binary'])",
    )
    test_fixtures: TestFixturesSpec = Field(
        default_factory=TestFixturesSpec, description="Data fixtures required for tests"
    )
    dev_suite: list[TestCaseSpec] = Field(
        default_factory=list, description="Visible test cases used for iterative refinement"
    )

    @field_validator("required_packages")
    @classmethod
    def validate_required_packages(cls, v: list[str]) -> list[str]:
        from adas_core.environment import validate_package_requirement

        for pkg in v:
            if not validate_package_requirement(pkg):
                raise ValueError(f"Invalid package requirement '{pkg}'. Must be a valid PEP 508 requirement.")
        return v

    @field_validator("dev_suite")
    @classmethod
    def validate_unique_test_case_ids(cls, v: list[TestCaseSpec]) -> list[TestCaseSpec]:
        seen_ids: set[str] = set()
        seen_sanitized: dict[str, str] = {}
        for tc in v:
            if tc.id in seen_ids:
                raise ValueError(f"Duplicate test case id '{tc.id}' in dev_suite.")
            seen_ids.add(tc.id)
            clean = sanitize_test_id(tc.id)
            if clean in seen_sanitized:
                raise ValueError(
                    f"Duplicate sanitized test case id '{clean}' in dev_suite: "
                    f"'{tc.id}' collides with '{seen_sanitized[clean]}'."
                )
            seen_sanitized[clean] = tc.id
        return v

    @model_validator(mode="after")
    def validate_judge_model_capabilities(self) -> TaskSpec:
        """Fail early for judge overrides that cannot evaluate declared modalities."""
        # Delayed import avoids the TaskSpec <-> ChatModel module dependency at import time.
        from adas_core.chat_model import ModelRegistry
        from meta_system.config import validation_wrapper

        for test_case in self.dev_suite:
            if not test_case.judge_model:
                continue
            provider = test_case.judge_provider or validation_wrapper
            if not ModelRegistry.is_registered_model(provider, test_case.judge_model):
                raise ValueError(
                    f"TestCase '{test_case.id}' specifies unregistered judge_model "
                    f"'{test_case.judge_model}' for provider '{provider}'. "
                    f"Add it to ModelRegistry before using it in a TaskSpec."
                )
            capabilities = ModelRegistry.get_capabilities(provider, test_case.judge_model)
            if "vision" in test_case.modalities and not capabilities.supports_vision:
                raise ValueError(
                    f"TestCase '{test_case.id}' requires vision evaluation but judge_model "
                    f"'{test_case.judge_model}' ({provider}) is not vision-capable."
                )
        return self

    @model_validator(mode="after")
    def validate_test_case_fixture_ids(self) -> TaskSpec:
        """Ensure all fixture_ids referenced by test cases exist in test_fixtures."""
        all_ids = self.test_fixtures.all_fixture_ids()
        for test_case in self.dev_suite:
            if test_case.fixture_ids is not None:
                for fid in test_case.fixture_ids:
                    if fid not in all_ids:
                        raise ValueError(
                            f"TestCase '{test_case.id}' references unknown fixture_id '{fid}'. "
                            f"Available fixture IDs: {sorted(all_ids)}"
                        )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to standard dictionary."""
        return self.model_dump(mode="json")

    def to_design_context(self) -> str:
        """Render the generalization-focused task contract supplied to the meta-agent.

        This intentionally excludes the concrete development cases.
        TaskSpec also has no holdout fields, so this representation cannot expose a private evaluation suite.
        """
        context = self.to_dict()
        context.pop("dev_suite", None)
        return json.dumps(context, indent=2, sort_keys=True)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save task specification to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        """Parse and validate from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> TaskSpec:
        """Parse and validate from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> TaskSpec:
        """Load, parse, and validate from a JSON file."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"TaskSpec file not found: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class HoldoutSuiteSpec(BaseModel):
    """Isolated specification for holdout test cases (stored in private_tasks/)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", description="Holdout schema version")
    task_name: str = Field(..., min_length=1, description="Associated task name")
    holdout_suite: list[TestCaseSpec] = Field(
        ..., min_length=1, description="List of private, unseen test cases for final acceptance"
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported schema_version '{v}'. Supported versions: {list(SUPPORTED_SCHEMA_VERSIONS)}")
        return v

    @field_validator("holdout_suite")
    @classmethod
    def validate_unique_holdout_ids(cls, v: list[TestCaseSpec]) -> list[TestCaseSpec]:
        seen_ids: set[str] = set()
        seen_sanitized: dict[str, str] = {}
        for tc in v:
            if tc.id in seen_ids:
                raise ValueError(f"Duplicate test case id '{tc.id}' in holdout_suite.")
            seen_ids.add(tc.id)
            clean = sanitize_test_id(tc.id)
            if clean in seen_sanitized:
                raise ValueError(
                    f"Duplicate sanitized test case id '{clean}' in holdout_suite: "
                    f"'{tc.id}' collides with '{seen_sanitized[clean]}'."
                )
            seen_sanitized[clean] = tc.id
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoldoutSuiteSpec:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> HoldoutSuiteSpec:
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> HoldoutSuiteSpec:
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"HoldoutSuite file not found: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
