import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from adas_core.environment import (
    ADAS_INPUT_DIR_ENV,
    ADAS_OUTPUT_DIR_ENV,
    ADAS_WORKSPACE_DIR_ENV,
    copy_directory_contents,
    ensure_packages_installed,
    is_package_excluded,
    is_package_installed,
    is_provisioning_script,
    isolated_case_workspace,
    normalize_package_name,
    run_preflight_check,
    validate_package_requirement,
)


class TestEnvironmentPackageProvisioning:
    def test_normalize_package_name(self):
        assert normalize_package_name("neo4j>=5.0,<6.0") == "neo4j"
        assert normalize_package_name("uvicorn[standard]>=0.20.0") == "uvicorn"
        assert normalize_package_name("psycopg2_binary") == "psycopg2-binary"
        assert normalize_package_name("duckdb~=0.9.0") == "duckdb"
        assert normalize_package_name("PyTest_Mock>=1.0") == "pytest-mock"
        assert normalize_package_name("langchain.core") == "langchain-core"

    def test_validate_package_requirement(self):
        assert validate_package_requirement("neo4j") is True
        assert validate_package_requirement("neo4j>=5.0,<6.0") is True
        assert validate_package_requirement("  uvicorn[standard]>=0.20.0  ") is True
        assert validate_package_requirement("psycopg2-binary") is True
        assert validate_package_requirement("duckdb~=0.9.0") is True

        assert validate_package_requirement("pkg; rm -rf /") is False
        assert validate_package_requirement("pkg && curl evil.com") is False
        assert validate_package_requirement("-r requirements.txt") is False
        assert validate_package_requirement("") is False

        with pytest.raises(ValueError, match="Invalid package requirement"):
            validate_package_requirement("bad; injection", raise_on_error=True)

    def test_is_package_excluded(self):
        # Exact canonical match
        assert is_package_excluded("pip") is True
        # Case insensitivity
        assert is_package_excluded("PIP") is True
        # Hyphen and underscore equivalence
        assert is_package_excluded("python_dotenv") is True
        assert is_package_excluded("python-dotenv") is True
        assert is_package_excluded("grpcio_status") is True

        # Unrelated distribution sharing substring must NOT be excluded
        assert is_package_excluded("pip-tools") is False
        assert is_package_excluded("pipenv") is False
        assert is_package_excluded("wheel-inspect") is False
        assert is_package_excluded("mypkg-docker") is False

        # Custom excluded list
        assert is_package_excluded("langchain_core", excluded_packages=["langchain-core"]) is True
        assert is_package_excluded("langgraph_checkpoint", excluded_packages=["langgraph"]) is False

    def test_is_package_installed(self):
        # pytest and pydantic are definitely installed in this venv
        assert is_package_installed("pytest") is True
        assert is_package_installed("pydantic") is True
        assert is_package_installed("nonexistent_package_xyz_9999") is False

    def test_is_provisioning_script(self):
        assert is_provisioning_script("generate_data.py") is True
        assert is_provisioning_script("seed_db.py") is True
        assert is_provisioning_script("mock_service.py") is True
        assert is_provisioning_script("setup_repo.py") is True
        assert is_provisioning_script("sub/generate_data.py") is True
        assert is_provisioning_script("data.csv") is False
        assert is_provisioning_script("seed.txt") is False
        assert is_provisioning_script("generator.py") is False
        assert is_provisioning_script("setup.cfg") is False

    def test_ensure_packages_installed_rejects_invalid_names(self):
        with pytest.raises(ValueError, match="Invalid package requirement"):
            ensure_packages_installed(["safe-pkg", "bad; rm -rf /"])

        with pytest.raises(ValueError, match="Invalid package requirement"):
            ensure_packages_installed(["pkg && curl evil.com"])

    @patch("subprocess.run")
    def test_ensure_packages_installed_skips_when_all_installed(self, mock_run):
        installed = ensure_packages_installed(["pytest", "pydantic"])
        assert installed == []
        mock_run.assert_not_called()

    @patch("adas_core.environment.is_package_installed", side_effect=lambda name: name == "pydantic")
    @patch("subprocess.run")
    def test_ensure_packages_installed_installs_missing(self, mock_run, mock_check):
        mock_run.return_value = MagicMock(stdout="Successfully installed")
        installed = ensure_packages_installed(["pydantic", "neo4j>=5.0"])
        assert installed == ["neo4j>=5.0"]
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "pip", "install"]
        assert "neo4j>=5.0" in cmd

    @patch("adas_core.environment.is_package_installed", return_value=False)
    @patch("subprocess.run")
    def test_ensure_packages_installed_raises_on_pip_error(self, mock_run, mock_check):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["pip", "install"], stderr="Package not found"
        )
        with pytest.raises(RuntimeError, match="Failed to install packages"):
            ensure_packages_installed(["failing-pkg"])


class TestDirectoryIsolationAndWorkspace:
    def test_copy_directory_contents_preserves_structure_and_skips_caches(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "data.csv").write_text("id,val\n1,10", encoding="utf-8")
        subdir = src / "nested"
        subdir.mkdir()
        (subdir / "info.json").write_text('{"key": "value"}', encoding="utf-8")

        # Cache dirs that should be skipped
        pycache = src / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-311.pyc").write_text("bytecode", encoding="utf-8")
        (src / "stray.pyc").write_text("bytecode", encoding="utf-8")

        dest = tmp_path / "dest"
        copied = copy_directory_contents(src, dest)
        assert copied == 2

        assert (dest / "data.csv").exists()
        assert (dest / "data.csv").read_text(encoding="utf-8") == "id,val\n1,10"
        assert (dest / "nested" / "info.json").exists()
        assert not (dest / "__pycache__").exists()
        assert not (dest / "stray.pyc").exists()

    def test_copy_directory_contents_skips_provisioning_scripts_by_default(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "data.csv").write_text("id,val\n1,10", encoding="utf-8")
        (src / "generate_data.py").write_text("# generator", encoding="utf-8")
        (src / "seed_db.py").write_text("# seeder", encoding="utf-8")
        (src / "mock_api.py").write_text("# mock", encoding="utf-8")
        (src / "setup_env.py").write_text("# setup", encoding="utf-8")
        (src / "normal.py").write_text("# normal", encoding="utf-8")

        dest = tmp_path / "dest"
        copied = copy_directory_contents(src, dest)
        assert copied == 2
        assert (dest / "data.csv").exists()
        assert (dest / "normal.py").exists()
        assert not (dest / "generate_data.py").exists()
        assert not (dest / "seed_db.py").exists()
        assert not (dest / "mock_api.py").exists()
        assert not (dest / "setup_env.py").exists()

        dest_all = tmp_path / "dest_all"
        copied_all = copy_directory_contents(src, dest_all, exclude_provisioning_scripts=False)
        assert copied_all == 6
        assert (dest_all / "generate_data.py").exists()

    def test_isolated_case_workspace_creates_dirs_and_sets_env(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "seed.txt").write_text("initial seed data")

        # Save initial environment
        initial_ws = os.environ.get(ADAS_WORKSPACE_DIR_ENV)
        initial_in = os.environ.get(ADAS_INPUT_DIR_ENV)
        initial_out = os.environ.get(ADAS_OUTPUT_DIR_ENV)

        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_101",
            case_id="case_1",
            fixtures_dir=fixtures_dir,
            clean_up=False,
        ) as ws:
            assert ws["workspace"].exists()
            assert ws["input"].exists()
            assert ws["output"].exists()

            # Verify environment variables point to these exact paths
            assert os.environ[ADAS_WORKSPACE_DIR_ENV] == str(ws["workspace"])
            assert os.environ[ADAS_INPUT_DIR_ENV] == str(ws["input"])
            assert os.environ[ADAS_OUTPUT_DIR_ENV] == str(ws["output"])

            # Verify fixture was seeded into input directory
            assert (ws["input"] / "seed.txt").exists()
            assert (ws["input"] / "seed.txt").read_text() == "initial seed data"

            # Simulate agent writing an output file
            (ws["output"] / "result.txt").write_text("agent solution")

        # After exiting the block, environment variables must be restored
        assert os.environ.get(ADAS_WORKSPACE_DIR_ENV) == initial_ws
        assert os.environ.get(ADAS_INPUT_DIR_ENV) == initial_in
        assert os.environ.get(ADAS_OUTPUT_DIR_ENV) == initial_out

        # Case directory still exists since clean_up was False
        assert ws["workspace"].exists()
        assert (ws["output"] / "result.txt").read_text() == "agent solution"

    def test_isolated_case_workspace_clean_up(self, tmp_path):
        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_clean",
            case_id="case_clean",
            clean_up=True,
        ) as ws:
            case_dir = ws["workspace"]
            assert case_dir.exists()

        # Should be removed after exiting
        assert not case_dir.exists()

    def test_isolated_case_workspace_filters_allowed_files(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "clean.csv").write_text("clean data")
        (fixtures_dir / "dirty.json").write_text("corrupted data")

        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_filter",
            case_id="case_filtered",
            fixtures_dir=fixtures_dir,
            allowed_files=["clean.csv"],
            clean_up=False,
        ) as ws:
            assert (ws["input"] / "clean.csv").exists()
            assert not (ws["input"] / "dirty.json").exists()

    def test_isolated_case_workspace_filters_allowed_directory_batch_fixtures(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        reports_dir = fixtures_dir / "reports"
        reports_dir.mkdir()
        (reports_dir / "q1.csv").write_text("q1 data", encoding="utf-8")
        (reports_dir / "q2.csv").write_text("q2 data", encoding="utf-8")

        other_dir = fixtures_dir / "unrelated_dir"
        other_dir.mkdir()
        (other_dir / "other.csv").write_text("other data", encoding="utf-8")

        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_batch_filter",
            case_id="case_batch_filtered",
            fixtures_dir=fixtures_dir,
            allowed_files=["reports/"],
            clean_up=False,
        ) as ws:
            assert (ws["input"] / "reports").is_dir()
            assert (ws["input"] / "reports" / "q1.csv").exists()
            assert (ws["input"] / "reports" / "q1.csv").read_text(encoding="utf-8") == "q1 data"
            assert (ws["input"] / "reports" / "q2.csv").exists()
            assert not (ws["input"] / "unrelated_dir").exists()

    def test_isolated_case_workspace_excludes_provisioning_scripts_when_allowed_files_none(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "data.csv").write_text("data")
        (fixtures_dir / "generate_data.py").write_text("# generator")
        (fixtures_dir / "seed_db.py").write_text("# seeder")

        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_prov_none",
            case_id="case_1",
            fixtures_dir=fixtures_dir,
            allowed_files=None,
            clean_up=False,
        ) as ws:
            assert (ws["input"] / "data.csv").exists()
            assert not (ws["input"] / "generate_data.py").exists()
            assert not (ws["input"] / "seed_db.py").exists()

    def test_isolated_case_workspace_copies_explicitly_allowed_python_fixture(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "clean.csv").write_text("clean")
        (fixtures_dir / "generate_data.py").write_text("# generator")

        with isolated_case_workspace(
            base_dir=tmp_path / "runs",
            run_id="run_prov_allowed",
            case_id="case_1",
            fixtures_dir=fixtures_dir,
            allowed_files=["clean.csv", "generate_data.py"],
            clean_up=False,
        ) as ws:
            assert (ws["input"] / "clean.csv").exists()
            assert (ws["input"] / "generate_data.py").exists()

    def test_isolated_case_workspace_rejects_unsafe_allowed_files(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "safe.csv").write_text("safe")

        # Rejection of directory traversal
        with (
            pytest.raises(ValueError, match="path traversal"),
            isolated_case_workspace(
                base_dir=tmp_path / "runs",
                run_id="run_bad_traversal",
                case_id="case_1",
                fixtures_dir=fixtures_dir,
                allowed_files=["../../outside.txt"],
            ),
        ):
            pass

        # Rejection of absolute paths
        with (
            pytest.raises(ValueError, match="absolute paths are not allowed"),
            isolated_case_workspace(
                base_dir=tmp_path / "runs",
                run_id="run_bad_abs",
                case_id="case_1",
                fixtures_dir=fixtures_dir,
                allowed_files=["/etc/passwd"],
            ),
        ):
            pass

        # Rejection of Windows drive absolute paths
        with (
            pytest.raises(ValueError, match="absolute paths are not allowed"),
            isolated_case_workspace(
                base_dir=tmp_path / "runs",
                run_id="run_bad_drive",
                case_id="case_1",
                fixtures_dir=fixtures_dir,
                allowed_files=["C:/Windows/System32"],
            ),
        ):
            pass

        # Rejection of empty paths
        with (
            pytest.raises(ValueError, match="path cannot be empty"),
            isolated_case_workspace(
                base_dir=tmp_path / "runs",
                run_id="run_bad_empty",
                case_id="case_1",
                fixtures_dir=fixtures_dir,
                allowed_files=[""],
            ),
        ):
            pass

        # Rejection of root/current directory paths
        with (
            pytest.raises(ValueError, match="empty or root normalized path"),
            isolated_case_workspace(
                base_dir=tmp_path / "runs",
                run_id="run_bad_dot",
                case_id="case_1",
                fixtures_dir=fixtures_dir,
                allowed_files=["."],
            ),
        ):
            pass


class TestPreflightExecution:
    def test_run_preflight_check_passes_when_file_missing(self, tmp_path):
        missing_script = tmp_path / "nonexistent_preflight.py"
        is_ok, msg = run_preflight_check(missing_script)
        assert is_ok is True
        assert "No preflight" in msg

    def test_run_preflight_check_executes_function_successfully(self, tmp_path):
        preflight = tmp_path / "preflight.py"
        preflight.write_text(
            """
def check_environment(workspace_dirs: dict[str, str]) -> tuple[bool, str]:
    if "input" in workspace_dirs:
        return True, "Environment verified successfully"
    return False, "Missing input dir"
""",
            encoding="utf-8",
        )
        is_ok, msg = run_preflight_check(preflight, {"input": "/path/input"})
        assert is_ok is True
        assert msg == "Environment verified successfully"

        is_fail, fail_msg = run_preflight_check(preflight, {"other": "/path/other"})
        assert is_fail is False
        assert fail_msg == "Missing input dir"

    def test_run_preflight_check_handles_exceptions_gracefully(self, tmp_path):
        preflight = tmp_path / "preflight_crash.py"
        preflight.write_text(
            """
def check_environment(workspace_dirs):
    raise ConnectionRefusedError("Database port unreachable")
""",
            encoding="utf-8",
        )
        is_ok, msg = run_preflight_check(preflight, {})
        assert is_ok is False
        assert "ConnectionRefusedError" in msg

    def test_run_preflight_check_rejects_invalid_return_type(self, tmp_path):
        preflight = tmp_path / "preflight_invalid.py"
        preflight.write_text("def check_environment(workspace_dirs):\n    return 'okay'\n", encoding="utf-8")

        is_ok, msg = run_preflight_check(preflight, {})
        assert is_ok is False
        assert "unexpected type" in msg
