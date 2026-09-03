import pytest
from pydantic import ValidationError

from adas_core.task_spec import (
    ApiKeyRequirement,
    ArchitectureContract,
    CustomFixtureSpec,
    DatabaseFixtureSpec,
    FileFixtureSpec,
    HoldoutSuiteSpec,
    MCPFixtureSpec,
    MockServiceFixtureSpec,
    ResourceEntry,
    ResourceManifest,
    TestCaseSpec,
    TestFixturesSpec,
    ToolRequirement,
    TaskSpec,
)


class TestTaskSpecModel:
    def test_minimal_valid_task_spec(self):
        spec = TaskSpec(
            name="SimpleMathAgent",
            system_goal="Solve basic arithmetic word problems.",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"query": "str", "answer": "str"},
            ),
            dev_suite=[
                TestCaseSpec(
                    id="case_1_addition",
                    description="Test simple addition 2 + 2",
                    turns=[{"query": "What is 2 + 2?"}],
                    expected_outputs=["answer"],
                    deterministic_criteria="Output answer contains '4'",
                )
            ],
        )

        assert spec.schema_version == "1.0"
        assert spec.name == "SimpleMathAgent"
        assert len(spec.dev_suite) == 1
        assert spec.dev_suite[0].turns[0]["query"] == "What is 2 + 2?"

    def test_multi_turn_auto_configures_persistence(self):
        spec = TaskSpec(
            name="ConversationalMathAgent",
            system_goal="Multi-turn math tutor.",
            architecture_contract=ArchitectureContract(
                execution_mode="multi_turn",
                state_schema={"messages": "Annotated[list[AnyMessage], add_messages]"},
            ),
            dev_suite=[
                TestCaseSpec(
                    id="case_1_two_turn",
                    description="Follow-up question",
                    turns=[
                        {"messages": [("human", "Let x = 5.")]},
                        {"messages": [("human", "What is x * 2?")]},
                    ],
                )
            ],
        )

        assert spec.architecture_contract.persistence is not None
        assert spec.architecture_contract.persistence.checkpointer == "memory"
        assert spec.architecture_contract.persistence.requires_thread_id is True

    def test_full_fixtures_and_required_packages_serialization(self, tmp_path):
        spec = TaskSpec(
            name="GraphAndApiAgent",
            system_goal="Query a Neo4j knowledge graph and call a mock REST API.",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"query": "str", "response": "str"},
                required_tools=[
                    ToolRequirement(name="cypher_query", description="Run Cypher queries on Neo4j"),
                    ToolRequirement(name="fetch_endpoint", description="Query local mock API"),
                ],
            ),
            resource_manifest=ResourceManifest(
                available_resources=[
                    ResourceEntry(
                        name="output_dir",
                        type="directory",
                        path_or_uri="data/output",
                        description="Location for outputs",
                    )
                ],
                available_api_keys=[
                    ApiKeyRequirement(env_var="NEO4J_PASSWORD", description="Neo4j auth", optional=False)
                ],
            ),
            required_packages=["neo4j>=5.0", "fastapi", "httpx"],
            test_fixtures=TestFixturesSpec(
                files=[
                    FileFixtureSpec(
                        path="input.csv",
                        content_type="csv",
                        description="Customer accounts to ingest",
                    )
                ],
                databases=[
                    DatabaseFixtureSpec(
                        name="graph_store",
                        db_type="neo4j",
                        connection_env={"uri": "NEO4J_URI", "user": "NEO4J_USER", "password": "NEO4J_PASSWORD"},
                        description="Nodes for Company, Product, and Vulnerability",
                    ),
                    DatabaseFixtureSpec(
                        name="local_cache",
                        db_type="sqlite",
                        file_path="data/cache.db",
                        description="SQLite cache table",
                    ),
                ],
                mcps=[
                    MCPFixtureSpec(
                        name="github_mcp",
                        transport="stdio",
                        command="python",
                        args=["fixtures/mock_github_mcp.py"],
                        description="Simulated GitHub issues and PR tools",
                    ),
                    MCPFixtureSpec(
                        name="remote_docs_mcp",
                        transport="streamable_http",
                        port=8090,
                        endpoint_path="/mcp",
                        url_env="DOCS_MCP_URL",
                        description="Streamable HTTP MCP server for documentation lookups",
                    ),
                ],
                mock_services=[
                    MockServiceFixtureSpec(
                        name="weather_mock",
                        port=8080,
                        base_url_env="MOCK_WEATHER_URL",
                        description="Serves /v1/current and /v1/forecast mock JSONs",
                    )
                ],
                custom_fixtures=[
                    CustomFixtureSpec(
                        name="git_repo",
                        description="Initialize git repo with 2 branches and a conflicting commit",
                    )
                ],
            ),
            dev_suite=[
                TestCaseSpec(
                    id="case_1_graph_lookup",
                    description="Find products related to Acme",
                    turns=[{"query": "Find Acme vulnerabilities"}],
                    llm_judge_needed=True,
                    judge_criteria="Checks if vulnerability CVE-2024-001 is mentioned.",
                )
            ],
        )

        # Verify default file_path on SQLite when omitted
        auto_sqlite = DatabaseFixtureSpec(name="quick_db", db_type="sqlite")
        assert auto_sqlite.file_path == "data/quick_db.sqlite"

        # JSON roundtrip
        json_str = spec.to_json()
        restored = TaskSpec.from_json(json_str)
        assert restored.name == spec.name
        assert restored.required_packages == ["neo4j>=5.0", "fastapi", "httpx"]
        assert len(restored.test_fixtures.databases) == 2
        assert restored.test_fixtures.databases[0].db_type == "neo4j"
        assert restored.test_fixtures.databases[0].connection_env["uri"] == "NEO4J_URI"
        assert len(restored.test_fixtures.mcps) == 2
        assert restored.test_fixtures.mcps[0].transport == "stdio"
        assert restored.test_fixtures.mcps[1].transport == "streamable_http"
        assert restored.test_fixtures.mcps[1].endpoint_path == "/mcp"
        assert restored.test_fixtures.mock_services[0].port == 8080
        assert restored.test_fixtures.custom_fixtures[0].name == "git_repo"

        # File save/load roundtrip
        file_path = tmp_path / "task.json"
        spec.save(file_path)
        assert file_path.exists()

        from_file_spec = TaskSpec.from_file(file_path)
        assert from_file_spec.name == spec.name
        assert from_file_spec.test_fixtures.files[0].path == "input.csv"

    def test_forbid_extra_fields(self):
        with pytest.raises(ValidationError):
            TaskSpec(
                name="AgentWithExtraField",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(
                    execution_mode="single_turn",
                    state_schema={"q": "str"},
                ),
                dev_suite=[],
                unknown_field="not_allowed",  # type: ignore
            )

    def test_empty_turns_rejected(self):
        with pytest.raises(ValidationError):
            TestCaseSpec(
                id="empty_case",
                description="Has no turns",
                turns=[],
            )

    def test_llm_judge_requires_judge_criteria(self):
        # Setting llm_judge_needed=True without judge_criteria must fail
        with pytest.raises(ValidationError):
            TestCaseSpec(
                id="judge_case_missing_criteria",
                description="Judge needed without criteria",
                turns=[{"input": "test"}],
                llm_judge_needed=True,
                judge_criteria=None,
            )

    def test_duplicate_dev_suite_ids_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec(
                name="DuplicateIdsAgent",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(
                    execution_mode="single_turn",
                    state_schema={"q": "str"},
                ),
                dev_suite=[
                    TestCaseSpec(id="case_1", description="First", turns=[{"q": "1"}]),
                    TestCaseSpec(id="case_1", description="Duplicate", turns=[{"q": "2"}]),
                ],
            )


class TestHoldoutSuiteSpecModel:
    def test_valid_holdout_suite_spec(self, tmp_path):
        holdout = HoldoutSuiteSpec(
            schema_version="1.0",
            task_name="SimpleMathAgent",
            holdout_suite=[
                TestCaseSpec(
                    id="holdout_1_division",
                    description="Hidden division test",
                    turns=[{"query": "What is 10 / 2?"}],
                    expected_outputs=["answer"],
                    deterministic_criteria="Output must contain '5'",
                ),
                TestCaseSpec(
                    id="holdout_2_negative",
                    description="Hidden negative addition",
                    turns=[{"query": "What is -3 + 5?"}],
                    expected_outputs=["answer"],
                    deterministic_criteria="Output must contain '2'",
                ),
            ],
        )

        assert len(holdout.holdout_suite) == 2
        file_path = tmp_path / "holdout.json"
        holdout.save(file_path)

        loaded = HoldoutSuiteSpec.from_file(file_path)
        assert loaded.task_name == "SimpleMathAgent"
        assert loaded.holdout_suite[1].id == "holdout_2_negative"

    def test_empty_holdout_suite_rejected(self):
        with pytest.raises(ValidationError):
            HoldoutSuiteSpec(
                task_name="EmptyHoldout",
                holdout_suite=[],
            )

    def test_duplicate_holdout_ids_rejected(self):
        with pytest.raises(ValidationError):
            HoldoutSuiteSpec(
                task_name="DupHoldout",
                holdout_suite=[
                    TestCaseSpec(id="dup", description="A", turns=[{"x": 1}]),
                    TestCaseSpec(id="dup", description="B", turns=[{"x": 2}]),
                ],
            )
