from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    execution_mode: Literal["single_turn", "multi_turn"] = Field(
        default="single_turn", description="Single-turn execution or multi-turn conversational execution"
    )
    state_schema: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Mapping of state attribute names to type annotations (e.g. {'messages': 'Annotated[list[AnyMessage], add_messages]'})",
    )
    persistence: PersistenceContract | None = Field(
        default=None, description="Checkpointer configuration for stateful/multi-turn execution"
    )
    required_tools: list[ToolRequirement] = Field(
        default_factory=list, description="List of tools the target system must provide"
    )

    @model_validator(mode="after")
    def validate_multi_turn_persistence(self) -> ArchitectureContract:
        if self.execution_mode == "multi_turn" and self.persistence is None:
            self.persistence = PersistenceContract(checkpointer="memory", requires_thread_id=True)
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

    path: str = Field(
        ..., min_length=1, description="Relative destination path or directory (e.g. 'test.csv' or 'docs/')"
    )
    count: int = Field(
        default=1, ge=1, description="Number of files to generate (1 for single file, N for batch/folder)"
    )
    description: str = Field(default="", description="Purpose, schema, or content requirements for this file")
    content_type: Literal["text", "csv", "json", "binary"] = Field(default="text", description="File format")


class DatabaseFixtureSpec(BaseModel):
    """Specification of a test database or knowledge graph (embedded or network-based)."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def validate_db_configuration(self) -> DatabaseFixtureSpec:
        if self.db_type in ("sqlite", "duckdb") and not self.file_path:
            self.file_path = f"data/{self.name}.{self.db_type}"
        return self


class MCPFixtureSpec(BaseModel):
    """Specification for a Model Context Protocol (MCP) server fixture."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="MCP server identifier (e.g. 'github_mcp', 'filesystem_mcp')")
    transport: Literal["stdio", "streamable_http", "sse"] = Field(
        default="stdio",
        description="MCP transport protocol: 'stdio' for subprocess pipes, 'streamable_http' for modern HTTP, or 'sse' for legacy HTTP",
    )
    command: str | None = Field(
        default=None, description="Executable command for stdio transport (e.g. 'python fixtures/mock_mcp.py')"
    )
    args: list[str] = Field(default_factory=list, description="Command line arguments for stdio transport")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables passed to the MCP server process"
    )
    port: int | None = Field(default=None, description="Port number if transport is 'streamable_http' or 'sse'")
    endpoint_path: str = Field(
        default="/mcp", description="HTTP endpoint path for Streamable HTTP (defaults to '/mcp')"
    )
    url_env: str = Field(
        default="MCP_SERVER_URL", description="Env var exposing the server URL to the agent for HTTP transports"
    )
    description: str = Field(
        default="", description="Tools, resources, and simulated behaviors this MCP server provides"
    )


class MockServiceFixtureSpec(BaseModel):
    """Specification of a mock HTTP service / API for tools to query."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Service name (e.g. 'mock_weather_api')")
    port: int = Field(default=8000, description="Local port for the mock server")
    base_url_env: str = Field(
        default="MOCK_API_BASE_URL", description="Env var exposing the mock server URL to the agent"
    )
    description: str = Field(default="", description="Endpoints, routes, and response behavior to mock")


class CustomFixtureSpec(BaseModel):
    """Escape hatch for custom environment setup (e.g. git repos, process mocks)."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Fixture identifier")
    description: str = Field(..., min_length=1, description="Description of the custom setup requirements")


class TestFixturesSpec(BaseModel):
    """Declarative specification of all test fixtures."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    files: list[FileFixtureSpec] = Field(default_factory=list, description="File fixtures to generate")
    databases: list[DatabaseFixtureSpec] = Field(default_factory=list, description="Database fixtures to generate/seed")
    mcps: list[MCPFixtureSpec] = Field(
        default_factory=list, description="MCP server fixtures (stdio, Streamable HTTP, or legacy SSE)"
    )
    mock_services: list[MockServiceFixtureSpec] = Field(
        default_factory=list, description="Mock HTTP services to run during tests"
    )
    custom_fixtures: list[CustomFixtureSpec] = Field(
        default_factory=list, description="Custom fixture scripts (git repos, CLI mocks, etc.)"
    )


class TestCaseSpec(BaseModel):
    """Specification for a single test scenario (single-turn or multi-turn)."""

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique test case identifier (e.g. 'case_1_basic_math')")
    description: str = Field(..., min_length=1, description="Description of the test scenario and edge cases")
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
        return self


class TaskSpec(BaseModel):
    """Complete, versioned specification for a target agentic system task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", description="Specification schema version")
    name: str = Field(..., min_length=1, description="Name of the target system to design")
    system_goal: str = Field(..., min_length=1, description="Primary goal and problem statement for the Meta-Agent")
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

    @field_validator("dev_suite")
    @classmethod
    def validate_unique_test_case_ids(cls, v: list[TestCaseSpec]) -> list[TestCaseSpec]:
        seen_ids = set()
        for tc in v:
            if tc.id in seen_ids:
                raise ValueError(f"Duplicate test case id '{tc.id}' in dev_suite.")
            seen_ids.add(tc.id)
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

    @field_validator("holdout_suite")
    @classmethod
    def validate_unique_holdout_ids(cls, v: list[TestCaseSpec]) -> list[TestCaseSpec]:
        seen_ids = set()
        for tc in v:
            if tc.id in seen_ids:
                raise ValueError(f"Duplicate test case id '{tc.id}' in holdout_suite.")
            seen_ids.add(tc.id)
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
