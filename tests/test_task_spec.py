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

    def test_multi_turn_is_rejected_until_execution_is_implemented(self):
        with pytest.raises(ValidationError, match="NOT_IMPLEMENTED: Execution mode 'multi_turn'"):
            ArchitectureContract(
                execution_mode="multi_turn",
                state_schema={"messages": "Annotated[list[AnyMessage], add_messages]"},
            )

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
                custom_fixtures=[
                    CustomFixtureSpec(
                        name="git_repo",
                        path="repo/",
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
        assert restored.test_fixtures.custom_fixtures[0].name == "git_repo"

        # File save/load roundtrip
        file_path = tmp_path / "task.json"
        spec.save(file_path)
        assert file_path.exists()

        from_file_spec = TaskSpec.from_file(file_path)
        assert from_file_spec.name == spec.name
        assert from_file_spec.test_fixtures.files[0].path == "input.csv"

    def test_process_fixtures_are_rejected_until_lifecycle_exists(self):
        with pytest.raises(ValidationError, match="NOT_IMPLEMENTED: MCP and mock HTTP service fixtures"):
            TestFixturesSpec(mcps=[MCPFixtureSpec(name="docs_mcp", port=8090)])

        with pytest.raises(ValidationError, match="NOT_IMPLEMENTED: MCP and mock HTTP service fixtures"):
            TestFixturesSpec(mock_services=[MockServiceFixtureSpec(name="weather_mock", port=8080)])

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
                custom_fixtures=[
                    CustomFixtureSpec(
                        id="api_mock", name="api_mock", path="api_mock/", description="Placeholder fixture"
                    )
                ],
            ),
            dev_suite=[
                TestCaseSpec(id="case_1", description="Clean only", fixture_ids=["clean_csv"], turns=[{"q": "1"}]),
                TestCaseSpec(id="case_2", description="All", fixture_ids=["clean_csv", "api_mock"], turns=[{"q": "2"}]),
            ],
        )
        assert spec.test_fixtures.all_fixture_ids() == {"clean_csv", "dirty_json", "api_mock"}
        assert spec.test_fixtures.get_file_paths_for_fixture_ids(["clean_csv"]) == ["clean.csv"]
        assert spec.test_fixtures.get_file_paths_for_fixture_ids(None) is None
        assert spec.test_fixtures.get_all_file_paths() == ["clean.csv", "dirty.json", "api_mock"]

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

    def test_task_spec_validates_required_packages(self):
        spec = TaskSpec(
            name="ValidPkgAgent",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            required_packages=["neo4j>=5.0", "fastapi", "psycopg2-binary"],
            dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
        )
        assert spec.required_packages == ["neo4j>=5.0", "fastapi", "psycopg2-binary"]

        with pytest.raises(ValidationError, match="Invalid package requirement"):
            TaskSpec(
                name="BadPkgAgent",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
                required_packages=["bad; rm -rf /"],
                dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
            )

    def test_task_spec_rejects_unsupported_schema_version(self):
        with pytest.raises(ValidationError, match="Unsupported schema_version '2.0'"):
            TaskSpec(
                schema_version="2.0",
                name="FutureTask",
                system_goal="Goal",
                architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
                dev_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
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
                CustomFixtureSpec(
                    id="custom_config", name="custom_config", path="config/custom.yaml", description="Config"
                ),
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
        # Remote databases have no file artifact; custom fixtures always declare one.
        assert tf.get_file_paths_for_fixture_ids(["remote_neo4j"]) == []
        assert tf.get_file_paths_for_fixture_ids(["custom_config"]) == ["config/custom.yaml"]

    def test_custom_fixture_requires_artifact_path(self):
        with pytest.raises(ValidationError, match="path"):
            CustomFixtureSpec.model_validate({"name": "missing_artifact", "description": "No artifact path"})

    def test_get_file_paths_fails_loudly_on_invalid_fixture_path(self):
        tf = TestFixturesSpec(files=[FileFixtureSpec(id="sales_csv", path="sales.csv", description="Sales")])
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

    def test_holdout_suite_rejects_unsupported_schema_version(self):
        from adas_core.task_spec import HoldoutSuiteSpec

        with pytest.raises(ValidationError, match="Unsupported schema_version '0.9'"):
            HoldoutSuiteSpec(
                schema_version="0.9",
                task_name="OldTask",
                holdout_suite=[TestCaseSpec(id="c1", description="d", turns=[{"q": "1"}])],
            )


class TestExampleSpecs:
    def test_example_specs_conform_to_schema(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        example_specs_dir = repo_root / "example_specs"
        assert example_specs_dir.is_dir(), "example_specs directory should exist"

        spec_files = sorted(example_specs_dir.glob("*/task.json"))
        assert len(spec_files) >= 1, f"Expected at least 1 example spec, found {len(spec_files)}"

        expected_dirs = {"data_analyst"}
        found_dirs = {p.parent.name for p in spec_files}
        assert expected_dirs.issubset(found_dirs), f"Missing expected example specs: {expected_dirs - found_dirs}"

        for spec_path in spec_files:
            spec = TaskSpec.from_file(spec_path)
            assert spec.name, f"Spec at {spec_path} must have a non-empty name"
            assert spec.system_goal, f"Spec at {spec_path} must have a system_goal"
            assert spec.architecture_contract.execution_mode in {"single_turn", "multi_turn"}
            assert len(spec.dev_suite) > 0, f"Spec at {spec_path} must have at least one test case in dev_suite"


class TestBenchmarkSpecs:
    def test_benchmark_specs_conform_to_schema(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        benchmark_dir = repo_root / "benchmark"
        assert benchmark_dir.is_dir(), "benchmark directory should exist"

        spec_files = sorted(benchmark_dir.glob("*/spec/task.json"))
        assert len(spec_files) == 3, f"Expected 3 benchmark specs, found {len(spec_files)}"

        expected_benchmarks = {"FEVER", "GSMHard", "MMLUPro"}
        found_benchmarks = {p.parent.parent.name for p in spec_files}
        assert expected_benchmarks == found_benchmarks, f"Mismatch in benchmark specs: {found_benchmarks}"

        for spec_path in spec_files:
            spec = TaskSpec.from_file(spec_path)
            assert spec.name, f"Spec at {spec_path} must have a non-empty name"
            assert spec.system_goal, f"Spec at {spec_path} must have a system_goal"
            assert spec.architecture_contract.execution_mode in {"single_turn", "multi_turn"}
            assert len(spec.dev_suite) >= 3, (
                f"Spec at {spec_path} must have at least 3 dev test cases including smoke test"
            )
            assert spec.dev_suite[0].id.startswith("case_0_smoke"), f"First case in {spec_path} should be smoke test"

    def test_benchmark_setup_manifests_are_current(self):
        from pathlib import Path

        from adas_core.automatic_validation import is_validation_manifest_current
        from create_setup import setup_manifest_is_current

        repo_root = Path(__file__).resolve().parent.parent
        benchmark_dir = repo_root / "benchmark"
        spec_files = sorted(benchmark_dir.glob("*/spec/task.json"))
        assert len(spec_files) == 3, f"Expected 3 benchmark specs, found {len(spec_files)}"

        for spec_path in spec_files:
            spec_dir = spec_path.parent
            manifest_file = spec_dir / "setup_manifest.json"
            preflight_file = spec_dir / "preflight.py"
            validation_files = list(spec_dir.glob("*.validation.py"))

            assert manifest_file.is_file(), f"Missing setup_manifest.json for {spec_path}"
            assert preflight_file.is_file(), f"Missing preflight.py for {spec_path}"
            assert len(validation_files) == 1, (
                f"Expected exactly 1 *.validation.py file in {spec_dir}, found {len(validation_files)}"
            )
            validation_file = validation_files[0]
            assert setup_manifest_is_current(spec_path), (
                f"Setup manifest at {manifest_file} is not current for {spec_path}. "
                "Task spec content hash does not match setup_manifest.json."
            )
            task_spec = TaskSpec.from_file(spec_path)
            assert is_validation_manifest_current(task_spec, spec_dir), (
                f"Validation section in {manifest_file} is missing or stale for {spec_path}."
            )

            # Compile preflight and validator scripts to ensure valid Python syntax
            import py_compile

            from adas_core.automatic_validation import load_validation_module

            py_compile.compile(str(preflight_file), doraise=True)
            py_compile.compile(str(validation_file), doraise=True)

            module = load_validation_module(validation_file)
            assert hasattr(module, "VALIDATORS") and isinstance(module.VALIDATORS, dict), (
                f"{validation_file} must export a VALIDATORS dictionary"
            )
            assert hasattr(module, "validate") and callable(module.validate), (
                f"{validation_file} must export a callable validate() function"
            )
            for test_case in task_spec.dev_suite:
                assert test_case.id in module.VALIDATORS, (
                    f"Test case '{test_case.id}' in {spec_path} is missing from VALIDATORS in {validation_file}"
                )
                assert callable(module.VALIDATORS[test_case.id]), (
                    f"Validator for '{test_case.id}' in {validation_file} must be callable"
                )
