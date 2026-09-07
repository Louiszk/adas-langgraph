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
    ModelSpec,
    ResourceEntry,
    ResourceManifest,
    TaskSpec,
    TestCaseSpec,
    TestFixturesSpec,
    ToolRequirement,
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
        context = spec.to_design_context()
        assert '"schema_version": "1.0"' in context
        assert '"dev_suite"' not in context
        assert "case_1_addition" not in context
        assert '"holdout_suite"' not in context

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
                        count=1,
                        content_type="csv",
                        description="Customer accounts to ingest",
                    )
                ],
                databases=[
                    DatabaseFixtureSpec(
                        name="graph_store",
                        db_type="neo4j",
                        connection_env={"uri": "NEO4J_URI", "user": "NEO4J_USER", "password": "NEO4J_PASSWORD"},
                        count=500,
                        description="Nodes for Company, Product, and Vulnerability",
                    ),
                    DatabaseFixtureSpec(
                        name="local_cache",
                        db_type="sqlite",
                        file_path="data/cache.db",
                        count=50,
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
                        transport="streamable-http",
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
        assert restored.test_fixtures.files[0].count == 1
        assert len(restored.test_fixtures.databases) == 2
        assert restored.test_fixtures.databases[0].db_type == "neo4j"
        assert restored.test_fixtures.databases[0].count == 500
        assert restored.test_fixtures.databases[0].connection_env["uri"] == "NEO4J_URI"
        assert len(restored.test_fixtures.mcps) == 2
        assert restored.test_fixtures.mcps[0].transport == "stdio"
        assert restored.test_fixtures.mcps[1].transport == "streamable-http"
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

    def test_default_available_models(self):
        spec = TaskSpec(
            name="DefaultModelAgent",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"q": "str"},
            ),
            dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
        )
        assert len(spec.available_models) == 1
        assert spec.available_models[0].provider == "openai"
        assert spec.available_models[0].model_name == "gpt-5.6-luna"

        context = spec.to_design_context()
        assert "available_models" in context
        assert "gpt-5.6-luna" in context

    def test_custom_available_models(self):
        spec = TaskSpec(
            name="MultiModelAgent",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"q": "str"},
            ),
            available_models=[
                ModelSpec(provider="openai", model_name="gpt-4o-mini"),
                ModelSpec(provider="openai", model_name="gpt-5.6-luna"),
            ],
            dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
        )
        assert len(spec.available_models) == 2
        assert spec.available_models[0].model_name == "gpt-4o-mini"
        assert spec.available_models[1].model_name == "gpt-5.6-luna"

        context = spec.to_design_context()
        assert "gpt-4o-mini" in context
        assert "gpt-5.6-luna" in context

    def test_model_spec_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            ModelSpec(provider="openai", model_name="gpt-4o", extra_field="forbidden")  # type: ignore

    def test_model_spec_forbids_legacy_wrapper_field(self):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate({"wrapper": "openai", "model_name": "gpt-4o"})

    def test_fixture_ids_scoping_and_validation(self):
        # Valid fixture IDs
        spec = TaskSpec(
            name="ScopedFixtureAgent",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"q": "str"},
            ),
            test_fixtures=TestFixturesSpec(
                files=[
                    FileFixtureSpec(id="clean_csv", path="clean.csv"),
                    FileFixtureSpec(id="dirty_json", path="dirty.json"),
                ],
                mock_services=[MockServiceFixtureSpec(id="api_mock", name="api_mock", port=8080)],
            ),
            dev_suite=[
                TestCaseSpec(id="case_1", description="Clean only", fixture_ids=["clean_csv"], turns=[{"q": "1"}]),
                TestCaseSpec(id="case_2", description="All", fixture_ids=["clean_csv", "api_mock"], turns=[{"q": "2"}]),
            ],
        )
        assert spec.test_fixtures.all_fixture_ids() == {"clean_csv", "dirty_json", "api_mock"}
        assert spec.test_fixtures.get_file_paths_for_fixture_ids(["clean_csv"]) == ["clean.csv"]
        assert spec.test_fixtures.get_file_paths_for_fixture_ids(None) is None

        # Unknown fixture_id raises ValidationError
        with pytest.raises(ValidationError, match="references unknown fixture_id 'nonexistent'"):
            TaskSpec(
                name="BadFixtureRefAgent",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(
                    execution_mode="single_turn",
                    state_schema={"q": "str"},
                ),
                test_fixtures=TestFixturesSpec(
                    files=[FileFixtureSpec(id="clean_csv", path="clean.csv")],
                ),
                dev_suite=[TestCaseSpec(id="c1", description="d", fixture_ids=["nonexistent"], turns=[{"q": "1"}])],
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

    def test_test_case_spec_judge_model_and_modalities(self):
        tc_default = TestCaseSpec(
            id="c_default",
            description="Default text case",
            turns=[{"q": "hello"}],
        )
        assert tc_default.judge_model is None
        assert tc_default.modalities == ["text"]

        tc_custom = TestCaseSpec(
            id="c_custom",
            description="Vision evaluation case",
            turns=[{"q": "generate plot"}],
            llm_judge_needed=True,
            judge_criteria="The plot must display all sales bars correctly.",
            judge_model="o3-mini",
            modalities=["text", "vision"],
        )
        assert tc_custom.judge_model == "o3-mini"
        assert tc_custom.modalities == ["text", "vision"]

    def test_vision_judge_requires_a_vision_capable_override(self):
        with pytest.raises(ValidationError, match="not vision-capable"):
            TaskSpec(
                name="VisionTask",
                system_goal="Assess a chart",
                architecture_contract=ArchitectureContract(state_schema={"query": "str"}),
                dev_suite=[
                    TestCaseSpec(
                        id="vision",
                        description="Assess a chart",
                        turns=[{"query": "go"}],
                        llm_judge_needed=True,
                        judge_criteria="The chart is readable.",
                        judge_model="o3-mini",
                        modalities=["vision"],
                    )
                ],
            )

    def test_unknown_modalities_are_rejected(self):
        with pytest.raises(ValidationError):
            TestCaseSpec.model_validate({"id": "audio", "description": "Audio", "turns": [{}], "modalities": ["audio"]})

    def test_judge_provider_override_validation(self):
        with pytest.raises(ValidationError, match="specifies an empty judge_provider"):
            TestCaseSpec(
                id="case_prov",
                description="desc",
                turns=[{"query": "run"}],
                llm_judge_needed=True,
                judge_criteria="criteria",
                judge_provider="   ",
            )

        tc = TestCaseSpec(
            id="case_prov",
            description="desc",
            turns=[{"query": "run"}],
            llm_judge_needed=True,
            judge_criteria="criteria",
            judge_model="gpt-4o",
            judge_provider="openai",
        )
        assert tc.judge_provider == "openai"

    def test_judge_provider_unregistered_model_rejected(self):
        with pytest.raises(ValidationError, match="specifies unregistered judge_model"):
            TaskSpec(
                name="CustomProvTask",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(state_schema={"query": "str"}),
                dev_suite=[
                    TestCaseSpec(
                        id="case_1",
                        description="desc",
                        turns=[{"query": "run"}],
                        llm_judge_needed=True,
                        judge_criteria="criteria",
                        judge_model="unknown-custom-model",
                        judge_provider="openai",
                    )
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

    def test_duplicate_dev_suite_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate test case id 'case_1' in dev_suite"):
            TaskSpec(
                name="DupDevSuite",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(state_schema={"q": "str"}),
                dev_suite=[
                    TestCaseSpec(id="case_1", description="A", turns=[{"q": "1"}]),
                    TestCaseSpec(id="case_1", description="B", turns=[{"q": "2"}]),
                ],
            )

    def test_duplicate_fixture_ids_in_test_fixtures_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate fixture id 'shared_id' in test_fixtures"):
            TestFixturesSpec(
                files=[FileFixtureSpec(id="shared_id", path="test.csv", description="CSV")],
                databases=[DatabaseFixtureSpec(id="shared_id", name="db", db_type="sqlite", description="DB")],
            )

    def test_duplicate_fixture_ids_in_test_case_rejected(self):
        with pytest.raises(ValidationError, match="contains duplicate fixture_ids"):
            TestCaseSpec(
                id="case_1",
                description="Duplicate fixture ref",
                turns=[{"q": "1"}],
                fixture_ids=["f1", "f1"],
            )

    def test_get_file_paths_for_fixture_ids_batch_and_directory(self):
        tf = TestFixturesSpec(
            files=[
                FileFixtureSpec(id="single_file", path="data/input/sales.csv", description="Single file"),
                FileFixtureSpec(id="batch_dir", path="reports/", count=5, description="Batch of report files"),
                FileFixtureSpec(id="nested_dir", path="input/metrics/monthly/", count=3, description="Nested batch"),
            ]
        )

        assert tf.get_file_paths_for_fixture_ids(["single_file"]) == ["sales.csv"]
        assert tf.get_file_paths_for_fixture_ids(["batch_dir"]) == ["reports"]
        assert tf.get_file_paths_for_fixture_ids(["nested_dir"]) == ["metrics/monthly"]
        assert tf.get_file_paths_for_fixture_ids(["single_file", "batch_dir"]) == ["sales.csv", "reports"]
        assert tf.get_file_paths_for_fixture_ids(None) is None

    def test_get_file_paths_includes_embedded_databases_and_custom_fixtures(self):
        tf = TestFixturesSpec(
            files=[FileFixtureSpec(id="sales_csv", path="sales.csv", description="Sales")],
            databases=[
                DatabaseFixtureSpec(
                    id="sqlite_cache", name="cache_db", db_type="sqlite", file_path="data/cache.db", description="Cache"
                ),
                DatabaseFixtureSpec(id="remote_neo4j", name="graph_db", db_type="neo4j", description="Remote Neo4j"),
            ],
            custom_fixtures=[
                CustomFixtureSpec(id="sample_repo", name="sample_repo", path="repo/", description="Git repo"),
                CustomFixtureSpec(id="no_path_custom", name="no_path_custom", description="No path custom"),
            ],
        )

        # Scoped to sqlite db only
        assert tf.get_file_paths_for_fixture_ids(["sqlite_cache"]) == ["data/cache.db"]
        # Scoped to custom fixture with directory path
        assert tf.get_file_paths_for_fixture_ids(["sample_repo"]) == ["repo"]
        # Combined files, sqlite db, and custom path
        assert tf.get_file_paths_for_fixture_ids(["sales_csv", "sqlite_cache", "sample_repo"]) == [
            "sales.csv",
            "data/cache.db",
            "repo",
        ]
        # Remote db and custom without path yield no file paths
        assert tf.get_file_paths_for_fixture_ids(["remote_neo4j", "no_path_custom"]) == []

    def test_get_file_paths_fails_loudly_on_invalid_fixture_path(self):
        tf = TestFixturesSpec(
            files=[FileFixtureSpec(id="sales_csv", path="sales.csv", description="Sales")]
        )
        # Mutating the path post-construction must fail loudly rather than silently omitting it
        object.__setattr__(tf.files[0], "path", "../../outside.txt")
        with pytest.raises(ValueError, match="path traversal"):
            tf.get_file_paths_for_fixture_ids(["sales_csv"])

        with pytest.raises(ValueError, match="path traversal"):
            tf.get_script_filenames_for_fixture("sales_csv")

    def test_fixture_specs_reject_unsafe_paths(self):
        # FileFixtureSpec
        with pytest.raises(ValidationError, match="path traversal"):
            FileFixtureSpec(path="../../secret.txt")
        with pytest.raises(ValidationError, match="absolute paths are not allowed"):
            FileFixtureSpec(path="/abs/path.csv")
        with pytest.raises(ValidationError, match="absolute paths are not allowed"):
            FileFixtureSpec(path="C:/Windows/System32")
        with pytest.raises(ValidationError, match="empty or root normalized path"):
            FileFixtureSpec(path=".")

        # DatabaseFixtureSpec
        with pytest.raises(ValidationError, match="path traversal"):
            DatabaseFixtureSpec(name="mydb", db_type="sqlite", file_path="../mydb.sqlite")
        with pytest.raises(ValidationError, match="absolute paths are not allowed"):
            DatabaseFixtureSpec(name="mydb", db_type="sqlite", file_path="/var/lib/mydb.sqlite")

        # CustomFixtureSpec
        with pytest.raises(ValidationError, match="path traversal"):
            CustomFixtureSpec(name="mycustom", description="test", path="../../repo")
        with pytest.raises(ValidationError, match="absolute paths are not allowed"):
            CustomFixtureSpec(name="mycustom", description="test", path="/etc/git")

    def test_task_spec_rejects_colliding_sanitized_test_case_ids(self):
        # case-a and case_a both sanitize to case_a
        cases = [
            TestCaseSpec(id="case-a", description="Case A", turns=[{"input": "a"}]),
            TestCaseSpec(id="case_a", description="Case A duplicate", turns=[{"input": "b"}]),
        ]
        with pytest.raises(ValidationError, match="Duplicate sanitized test case id 'case_a'"):
            TaskSpec(
                name="CollisionTest",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(
                    execution_mode="single_turn", state_schema={"status": "str"}
                ),
                dev_suite=cases,
            )

    def test_holdout_suite_rejects_colliding_sanitized_test_case_ids(self):
        # eval-1 and eval_1 both sanitize to eval_1
        from adas_core.task_spec import HoldoutSuiteSpec

        cases = [
            TestCaseSpec(id="eval-1", description="Eval 1", turns=[{"input": "1"}]),
            TestCaseSpec(id="eval_1", description="Eval 1 duplicate", turns=[{"input": "2"}]),
        ]
        with pytest.raises(ValidationError, match="Duplicate sanitized test case id 'eval_1'"):
            HoldoutSuiteSpec(
                task_name="CollisionTest",
                holdout_suite=cases,
            )
