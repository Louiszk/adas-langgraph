import textwrap
from unittest.mock import MagicMock
import pytest

from langchain_core.messages import AIMessage

from adas_core.automatic_setup import (
    AutomaticSetup,
    ensure_automatic_setup,
    extract_code_block,
    extract_setup_requirements,
    normalize_fixture_path,
)
from adas_core.task_spec import (
    ApiKeyRequirement,
    ArchitectureContract,
    CustomFixtureSpec,
    DatabaseFixtureSpec,
    FileFixtureSpec,
    MCPFixtureSpec,
    MockServiceFixtureSpec,
    ResourceEntry,
    ResourceManifest,
    TaskSpec,
    TestCaseSpec,
    TestFixturesSpec,
)


class TestAutomaticSetupHelpers:
    def test_extract_code_block(self):
        assert extract_code_block("```csv\ncol1,col2\n1,2\n```") == "col1,col2\n1,2"
        assert extract_code_block("```python\nprint('hi')\n```") == "print('hi')"
        assert extract_code_block("plain text") == "plain text"

    def test_extract_setup_requirements(self):
        code = """
SETUP_REQUIREMENTS = ["neo4j>=5.0", "fastapi"]

def seed():
    pass
"""
        reqs = extract_setup_requirements(code)
        assert reqs == ["neo4j>=5.0", "fastapi"]

        no_reqs_code = "def foo(): pass"
        assert extract_setup_requirements(no_reqs_code) == []

    def test_normalize_fixture_path_valid(self):
        assert normalize_fixture_path("test.csv") == "test.csv"
        assert normalize_fixture_path("data/input/sales.csv") == "sales.csv"
        assert normalize_fixture_path("input/sub/report.pdf") == "sub/report.pdf"
        assert normalize_fixture_path("sandbox/workspace/input/metrics.json") == "metrics.json"
        assert normalize_fixture_path("sandbox/workspace/data/input/orders.csv") == "orders.csv"
        assert normalize_fixture_path("reports/") == "reports"
        assert normalize_fixture_path("./reports/q1.csv") == "reports/q1.csv"

    def test_normalize_fixture_path_rejects_unsafe(self):
        # Empty paths
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_fixture_path("")
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_fixture_path("   ")

        # Absolute paths
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path("/etc/passwd")
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path("C:/Windows/System32")
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            normalize_fixture_path("\\\\server\\share\\file.csv")

        # Path traversal
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("../../outside.txt")
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("input/../../outside.txt")
        with pytest.raises(ValueError, match="path traversal"):
            normalize_fixture_path("sub/../..")

        # Empty normalized / root paths
        with pytest.raises(ValueError, match="empty or root normalized path|resolves to root"):
            normalize_fixture_path(".")
        with pytest.raises(ValueError, match="empty or root normalized path|resolves to root"):
            normalize_fixture_path("./")
        with pytest.raises(ValueError, match="empty or root normalized path|resolves to root"):
            normalize_fixture_path("input/")
        with pytest.raises(ValueError, match="empty or root normalized path|resolves to root"):
            normalize_fixture_path("input")
        with pytest.raises(ValueError, match="empty or root normalized path|resolves to root"):
            normalize_fixture_path("sandbox/workspace/input/")


class TestAutomaticSetup:
    def test_generate_all_fixtures_and_preflight(self, tmp_path):
        mock_llm = MagicMock()

        # Define canned responses for LLM calls in order:
        # 1. file content (test.csv)
        # 2. database script (neo4j)
        # 3. mcp script (github_mcp)
        # 4. mock service script (weather_mock)
        # 5. custom fixture script (git_repo)
        # 6. preflight script
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
SETUP_REQUIREMENTS = ["neo4j>=5.0"]

def seed_database(workspace_dirs: dict[str, str]) -> None:
    pass
```"""
            ),
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = ["mcp", "langchain-mcp-adapters"]
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github_mock")
```"""
            ),
            AIMessage(
                content="""```python
SETUP_REQUIREMENTS = ["fastapi", "uvicorn"]
from fastapi import FastAPI
app = FastAPI()
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
SETUP_REQUIREMENTS = ["neo4j>=5.0", "fastapi", "mcp", "gitpython"]

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
                        db_type="neo4j",
                        connection_env={"uri": "NEO4J_URI", "password": "NEO4J_PASSWORD"},
                        description="Vulnerability graph",
                    )
                ],
                mcps=[
                    MCPFixtureSpec(
                        name="github_mcp",
                        transport="stdio",
                        command="python",
                        description="Mock GitHub PRs",
                    )
                ],
                mock_services=[
                    MockServiceFixtureSpec(
                        name="weather_mock",
                        port=8001,
                        description="Mock API",
                    )
                ],
                custom_fixtures=[
                    CustomFixtureSpec(
                        name="git_repo",
                        description="Git repository with branches",
                    )
                ],
            ),
            dev_suite=[TestCaseSpec(id="c1", description="desc", turns=[{"query": "check"}])],
        )

        setup = AutomaticSetup(llm=mock_llm)
        result = setup.generate_all(spec, tmp_path)

        # Verify files were created
        assert (tmp_path / "fixtures" / "generate_data_test_csv.py").exists()
        assert (tmp_path / "fixtures" / "data" / "test.csv").exists()
        assert (tmp_path / "fixtures" / "data" / "test.csv").read_text() == "id,revenue\n1,150.0\n2,250.0\n"
        assert (tmp_path / "fixtures" / "seed_sec_graph.py").exists()
        assert (tmp_path / "fixtures" / "mock_github_mcp.py").exists()
        assert (tmp_path / "fixtures" / "mock_weather_mock.py").exists()
        assert (tmp_path / "fixtures" / "setup_git_repo.py").exists()
        assert (tmp_path / "preflight.py").exists()

        # Verify discovered packages union
        assert "pydantic" in result.discovered_packages
        assert "neo4j>=5.0" in result.discovered_packages
        assert "fastapi" in result.discovered_packages
        assert "mcp" in result.discovered_packages
        assert "gitpython" in result.discovered_packages

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

        assert "repo/" in system_msg
        assert "workspace_dirs['ADAS_INPUT_DIR']" in system_msg
        assert "repo/" in user_msg
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
