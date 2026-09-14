import textwrap
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from adas_core.automatic_setup import (
    AutomaticSetup,
    ensure_automatic_setup,
    extract_setup_requirements,
)
from adas_core.task_spec import (
    ApiKeyRequirement,
    ArchitectureContract,
    CustomFixtureSpec,
    DatabaseFixtureSpec,
    ExternalDatabaseSeedSpec,
    FileFixtureSpec,
    MCPFixtureSpec,
    MockServiceFixtureSpec,
    ResourceEntry,
    ResourceManifest,
    TaskSpec,
    TestCaseSpec,
    TestFixturesSpec,
)


def test_extract_setup_requirements():
    code = """
SETUP_REQUIREMENTS = ["neo4j>=5.0", "fastapi"]

def seed():
    pass
"""
    reqs = extract_setup_requirements(code)
    assert reqs == ["neo4j>=5.0", "fastapi"]

    no_reqs_code = "def foo(): pass"
    assert extract_setup_requirements(no_reqs_code) == []


class TestAutomaticSetup:
    def test_generate_all_writes_process_scripts_to_fixtures(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="SETUP_REQUIREMENTS = []\n"),
            AIMessage(content="SETUP_REQUIREMENTS = []\n"),
            AIMessage(
                content=(
                    "def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:\n"
                    "    return True, 'ok'\n"
                )
            ),
        ]
        spec = TaskSpec(
            name="ProcessFixtureTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            test_fixtures=TestFixturesSpec(
                mcps=[MCPFixtureSpec(name="tools", port=8101)],
                mock_services=[MockServiceFixtureSpec(name="weather", port=8102)],
            ),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": "value"}])],
        )

        AutomaticSetup(llm=mock_llm).generate_all(spec, tmp_path, execute_generated_code=False)

        assert (tmp_path / "fixtures" / "mock_tools.py").is_file()
        assert (tmp_path / "fixtures" / "mock_weather.py").is_file()

    def test_generate_all_fixtures_and_preflight(self, tmp_path):
        mock_llm = MagicMock()

        # Define canned responses for LLM calls in order:
        # 1. file content (test.csv)
        # 2. database script (sqlite)
        # 3. custom fixture script (git_repo)
        # 4. preflight script
        mock_llm.invoke.side_effect = [
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = []

def generate_files(output_path: Path) -> None:
    output_path.write_text("id,revenue\\n1,150.0\\n2,250.0\\n")
```"""
            ),
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = []

def seed_database(workspace_dirs: dict[str, str]) -> None:
    pass
```"""
            ),
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = ["gitpython"]

def setup_environment(workspace_dirs: dict[str, str]) -> None:
    pass
```"""
            ),
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = ["gitpython"]

def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    return True, "Environment verified"
```"""
            ),
        ]

        spec = TaskSpec(
            name="EnterpriseSecAgent",
            system_goal="Analyze company vulnerabilities across graph and git repo.",
            architecture_contract=ArchitectureContract(
                execution_mode="single_turn",
                state_schema={"query": "str"},
            ),
            resource_manifest=ResourceManifest(
                available_resources=[ResourceEntry(name="out", type="directory")],
                available_api_keys=[ApiKeyRequirement(env_var="NEO4J_PASSWORD", description="Neo4j")],
            ),
            required_packages=["pydantic"],
            test_fixtures=TestFixturesSpec(
                files=[FileFixtureSpec(path="data/test.csv", content_type="csv", description="Financials")],
                databases=[
                    DatabaseFixtureSpec(
                        name="sec_graph",
                        db_type="sqlite",
                        description="Vulnerability graph",
                    )
                ],
                custom_fixtures=[
                    CustomFixtureSpec(
                        name="git_repo",
                        path="repo/",
                        description="Git repository with branches",
                    )
                ],
            ),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"query": "check"}])],
        )

        setup = AutomaticSetup(llm=mock_llm)
        result = setup.generate_all(spec, tmp_path)

        # Verify files were created
        assert (tmp_path / "setup_scripts" / "generate_data_test_csv.py").exists()
        assert not (tmp_path / "fixtures" / "generate_data_test_csv.py").exists()
        assert (tmp_path / "fixtures" / "data" / "test.csv").exists()
        assert (tmp_path / "fixtures" / "data" / "test.csv").read_text() == "id,revenue\n1,150.0\n2,250.0\n"
        assert (tmp_path / "setup_scripts" / "seed_sec_graph.py").exists()
        assert not (tmp_path / "fixtures" / "seed_sec_graph.py").exists()
        assert (tmp_path / "setup_scripts" / "setup_git_repo.py").exists()
        assert not (tmp_path / "fixtures" / "setup_git_repo.py").exists()
        assert (tmp_path / "preflight.py").exists()

        # Verify discovered packages union
        assert "pydantic" in result.discovered_packages
        assert "gitpython" in result.discovered_packages

    def test_generate_external_database_seed_script_without_executing_it(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content=(
                    "SETUP_REQUIREMENTS = ['psycopg']\n"
                    "def seed_external_database(connection_config, namespace): pass\n"
                    "def cleanup_external_database(connection_config, namespace): pass\n"
                )
            ),
            AIMessage(content="def check_environment(workspace_dirs): return True, 'ok'\n"),
        ]
        spec = TaskSpec(
            name="ExternalSeedSetup",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            resource_manifest=ResourceManifest(
                available_resources=[ResourceEntry(name="evaluation_db", type="database")]
            ),
            test_fixtures=TestFixturesSpec(
                external_database_seeds=[
                    ExternalDatabaseSeedSpec(
                        name="orders_seed",
                        resource_name="evaluation_db",
                        db_type="postgres",
                        driver="psycopg",
                        connection_env={"uri": "EVALUATION_DB_URI"},
                        namespace_kind="schema",
                        namespace="adas_test_orders",
                        description="Seed deterministic orders.",
                    )
                ]
            ),
        )

        result = AutomaticSetup(llm=mock_llm).generate_all(spec, tmp_path)

        assert (tmp_path / "setup_scripts" / "seed_external_orders_seed.py").is_file()
        assert "psycopg" in result.discovered_packages

    def test_ensure_automatic_setup_skips_when_present(self, tmp_path):
        # Create fixtures dir and preflight.py to simulate existing setup
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        preflight_file = tmp_path / "preflight.py"
        preflight_file.write_text("# existing preflight")

        spec = TaskSpec(
            name="ExistingTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "int"}),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": 1}])],
        )

        mock_llm = MagicMock()
        # Should return None without calling LLM
        res = ensure_automatic_setup(spec, tmp_path, llm=mock_llm)
        assert res is None
        mock_llm.invoke.assert_not_called()

    def test_ensure_automatic_setup_skips_when_zero_fixtures_and_no_fixtures_dir(self, tmp_path):
        # Only preflight exists; fixtures/ directory was never created because none are needed
        preflight_file = tmp_path / "preflight.py"
        preflight_file.write_text("# existing preflight")

        spec = TaskSpec(
            name="ZeroFixturesTask",
            system_goal="Conversational agent with no external files or databases",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "int"}),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": 1}])],
        )

        mock_llm = MagicMock()
        res = ensure_automatic_setup(spec, tmp_path, llm=mock_llm)
        assert res is None
        mock_llm.invoke.assert_not_called()
        assert not (tmp_path / "fixtures").exists()

    def test_generate_custom_fixture_script_instructs_artifact_path(self):
        spec = TaskSpec(
            name="CustomFixtureTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": "val"}])],
        )
        custom_fixture = CustomFixtureSpec(
            name="git_repo",
            path="repo/",
            description="Initialize git repo with master branch",
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="""```python
SETUP_REQUIREMENTS = ["gitpython"]

def setup_environment(workspace_dirs: dict[str, str]) -> None:
    pass
```"""
        )

        setup = AutomaticSetup(llm=mock_llm)
        code, reqs = setup.generate_custom_fixture_script(spec, custom_fixture)

        assert "gitpython" in reqs
        invoked_messages = mock_llm.invoke.call_args[0][0]
        system_msg = invoked_messages[0].content
        user_msg = invoked_messages[1].content

        assert custom_fixture.path in system_msg
        assert "workspace_dirs['ADAS_INPUT_DIR']" in system_msg
        assert custom_fixture.path in user_msg
        assert 'workspace_dirs["ADAS_INPUT_DIR"]' in user_msg

    def test_generate_preflight_script_normalizes_future_imports(self):
        spec = TaskSpec(
            name="FutureImportTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": "val"}])],
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content=textwrap.dedent(
                """
                ```python
                SETUP_REQUIREMENTS = ["pandas"]

                from __future__ import annotations

                def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
                    return True, "OK"
                ```
                """
            ).strip()
        )
        setup = AutomaticSetup(llm=mock_llm)
        code = setup.generate_preflight_script(spec, ["pandas"])

        compiled = compile(code, "<preflight>", "exec")
        assert compiled is not None
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        assert lines[0] == "from __future__ import annotations"
        assert lines[1] == 'SETUP_REQUIREMENTS = ["pandas"]'

    def test_generate_preflight_script_requires_runtime_resource_contracts(self):
        spec = TaskSpec(
            name="RuntimeContractTask",
            system_goal="Goal",
            architecture_contract=ArchitectureContract(execution_mode="single_turn", state_schema={"q": "str"}),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"q": "value"}])],
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="def check_environment(workspace_dirs): return True, 'ok'")

        AutomaticSetup(llm=mock_llm).generate_preflight_script(spec, [])

        system_prompt = mock_llm.invoke.call_args.args[0][0].content
        assert "Never hardcode a host or sandbox path" in system_prompt
        assert "runtime resource profile" in system_prompt

    def test_create_setup_cli_verify_flag(self, tmp_path):
        import json

        import create_setup
        from adas_core.task_spec import TaskSpec

        spec_file = tmp_path / "task.json"
        spec_data = {
            "name": "VerifySetupTask",
            "system_goal": "Goal",
            "architecture_contract": {
                "execution_mode": "single_turn",
                "state_schema": {"messages": "list[dict]"},
                "persistence": {},
                "required_tools": [],
            },
            "resource_manifest": {"available_resources": [], "available_api_keys": []},
            "dev_suite": [
                {
                    "id": "case_1",
                    "description": "Test case 1",
                    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                }
            ],
        }
        spec = TaskSpec.model_validate(spec_data)
        spec.save(spec_file)

        exit_code = create_setup.main(["--task-spec", str(spec_file), "--verify"])
        assert exit_code == 1

        task_hash = create_setup._file_hash(spec_file)
        manifest = {
            "schema_version": "1.0",
            "fixture_lifecycle_version": create_setup.FIXTURE_LIFECYCLE_VERSION,
            "task_name": "VerifySetupTask",
            "files": {
                "task.json": task_hash,
            },
        }
        (tmp_path / "setup_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        exit_code = create_setup.main(["--task-spec", str(spec_file), "--verify"])
        assert exit_code == 0
