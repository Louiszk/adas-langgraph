# Automated Design of Agentic Systems (ADAS) in LangGraph

A framework for the automated design, testing, and optimization of graph-structured agentic systems.

Manual engineering of complex, multi-agent workflows is time-intensive and limits the exploration of effective architectures. This project provides a **meta-system** that iteratively builds, tests, and refines target agentic systems using the [LangGraph](https://github.com/langchain-ai/langgraph) library. By operating on a code-based search space, the meta-system can autonomously discover novel control flows, integrate custom tools, and install external dependencies.

## Key Features & Findings

* **Modular Component Editing:** Instead of whole-file replacements or unified diffs, this framework uses component-level modifications. The meta-agent uses `manage_node`, `manage_tool`, `manage_conditional_edge`, `manage_edge`, and `manage_utilities` to make targeted changes to a virtual system.
* **Explicit Lifecycle and Routing:** Nodes, tools, conditional edges, and utilities use `create`, `update`, and `delete` actions. Conditional edges require an explicit `path_map` from each condition-function return value to a destination node or `END`.
* **Safe Graph and Utility Changes:** Deleting a node also removes standard edges connected to it and conditional edges that route to it. Utility deletion identifies the exact top-level `function`, `class`, or `assignment`, so same-named definitions can be removed unambiguously.
* **Automated Validation Guardrails:** Relies on programmatic test validation and structural graph checks rather than purely subjective LLM-as-a-judge approaches. This prevents premature finalization, effectively catches structural flaws (like dead ends, invalid path-map destinations, or infinite loops), and improves target system accuracy.

## Repository Structure

* `adas_core/`: The core logic, including the `VirtualAgenticSystem` representation, AST-based materialization, task specification schemas, and custom LLM wrappers.
* `meta_system/`: The implementation of the meta-agent, its management tools (`ManageNode`, `ManageTool`, `ManageConditionalEdge`, `ManageEdge`, `ManageUtilities`), and evaluation prompts.
* `example_specs/`: Task specifications (e.g., `data_analyst`) with schemas, contracts, and test fixtures.
* `generated_systems/`: The output directory where the meta-system saves the successfully built and compiled LangGraph target systems.
* `benchmark/`: Parallelized benchmarking suites (FEVER, GSM-Hard, MMLU-Pro) to evaluate target system accuracy and resource consumption.
* `sandbox/`: Docker/Podman integration using `llm-sandbox` to safely execute and evaluate generated code in isolated environments.
* `scripts/`: Central execution orchestrator and HPC/SLURM batch execution scripts.

---

## Quick Setup

### Environment Setup

1. Clone the repository.
2. Copy the example environment file:
   ```bash
   cp .env_copy .env
   ```
3. Edit `.env` with your API keys:
   ```
   OPENAI_API_KEY=sk-...
   ```

### Virtual Environment (recommended)

```bash
# Create virtual environment
python -m venv adasvenv

# Activate on Linux/Mac
source adasvenv/bin/activate
# OR on Windows
# .\adasvenv\Scripts\Activate.ps1

# Install runtime dependencies
pip install -r requirements.txt
# OR install development dependencies (testing, linting, formatting)
# pip install -r requirements-dev.txt
```

### Docker Setup (for sandbox execution)

The system uses Docker or Podman to create a sandbox environment for secure code execution. Make sure Docker is installed and running:

```bash
docker --version
```

### Security & Sandboxing Disclaimer

> [!WARNING]
> While execution occurs inside Docker/Podman containers, the sandbox is a cooperative boundary:
> - **Unrestricted Network Egress:** Outbound internet access is enabled by default.
> - **Credential Exposure:** The `.env` file and environment variables are readable by sandboxed code.
> - **Model Restrictions:** `ChatModel` allowlists and token budgets operate at the application layer and can be bypassed if generated code accesses `os.environ` or external SDKs directly.
>
> **Best Practice:** Use dedicated evaluation-only API keys with strict spend caps, keep `.env` minimal, and never store production credentials.

## Running the System

ADAS provides a 5-stage lifecycle for defining, provisioning, validating, designing, and executing agentic systems.

### 1. Define Task Specification (`create_taskspec.py`)
Synthesize a schema-validated task specification (`task.json`) defining the agent's goals, architecture contract, resource requirements, fixtures, and evaluation suite:
```bash
# Interactive CLI:
python create_taskspec.py

# Or non-interactive from command line / prompt:
python create_taskspec.py --name DataAnalyst --goal "Analyze CSV data and output summary" --non-interactive
```
Pre-built specifications are available in the `example_specs/` and `benchmark/` directories:
- `example_specs/data_analyst/task.json`: Automatic data analyst generating pandas & matplotlib workflows.
- `benchmark/GSMHard/spec/task.json`: Multi-step mathematical reasoning benchmark.
- `benchmark/FEVER/spec/task.json`: Factual claim verification with Wikipedia retrieval.
- `benchmark/MMLUPro/spec/task.json`: Computer science multiple-choice reasoning benchmark.

### 2. Synthesize Fixtures & Setup (`create_setup.py`)
Materialize deterministic sandbox fixtures (files, sqlite tables, mock endpoints) and preflight verification scripts based on `task.json`:
```bash
python create_setup.py --task-spec example_specs/data_analyst/task.json
```

### Resources versus Fixtures

Use `resource_manifest` for user-owned resources: files, directories, HTTP APIs, Streamable HTTP MCP servers, and external databases. Declare paths or connection details as resources and required credentials as API keys; the preflight step verifies availability, while ADAS does not start or mock them.

Use `test_fixtures` for deterministic, harness-owned evaluation inputs. ADAS can generate file data, create and seed embedded SQLite/DuckDB database files, and run mock HTTP or Streamable HTTP MCP services for a test case. External databases such as Postgres, Neo4j, Redis, and Qdrant cannot be mocked by this fixture mechanism; they must be available beforehand. An `external_database_seeds` entry may seed one only for development tests, and only through a declared database resource, named connection environment variables (never credentials), and an isolated `adas_test_*` schema/database/namespace that the lifecycle drops after every case. Direct `invoke_target` never seeds external databases by default.

For a direct invocation, use a runtime profile to replace selected fixture providers without changing the target system. The profile keys are declared fixture IDs: use `external` with a URL for HTTP/MCP fixtures, or `local_file` with a host file/directory path for file and embedded SQLite/DuckDB fixtures. ADAS stages local paths into the isolated workspace at the fixture's declared path.

```bash
python invoke_target.py --system-name my_system --task-spec task.json --runtime-config my_environment.json --state '{"query": "..."}'
```

```json
{
  "overrides": {
    "transit_api": {"provider": "external", "url": "https://staging.transit.example/api"},
    "stations_csv": {"provider": "local_file", "source": "C:/data/stations.csv"}
  }
}
```

Resources not listed in the profile continue to use their generated fixtures. Credentials remain normal environment/secret configuration, not profile values.

### 3. Generate Frozen Validation Module (`create_validation.py`)
Synthesize standalone, frozen evaluation code (`<task>.validation.py`) implementing deterministic checks and LLM-as-a-judge rubrics:
```bash
python create_validation.py --task-spec example_specs/data_analyst/task.json
```

### 4. Run Meta-System Design Optimization (`invoke_design.py`)
Run the autonomous meta-agent loop to design, iterate, and optimize a target LangGraph system inside an isolated container sandbox:
```bash
python invoke_design.py --task-spec example_specs/data_analyst/task.json --system-name data_analyst_iter1_gpt
```
*Dependencies are installed into a persisted local sandbox image once per dependency version, then reused by subsequent runs.*

**Options:**
* `--task-spec`: Required path to validated `task.json`.
* `--system-name`: Target system output identifier (defaults to TaskSpec name).
* `--auto-setup`: Automatically generate frozen fixtures and preflight artifacts if missing or stale.
* `--optimize-system`: Specify existing target system name to optimize/refine.
* `--reinstall`: Force re-installation of dependencies.

### 5. Execute Target System (`invoke_target.py`)
Execute a materialized target system in the sandbox with custom state and task fixtures:
```bash
# Invoke with custom state and task fixtures:
python invoke_target.py --system-name data_analyst_iter1_gpt --task-spec example_specs/data_analyst/task.json --state '{"analysis_task": "Analyze sales.csv"}'

# Or invoke using a JSON state file:
python invoke_target.py --system-name data_analyst_iter1_gpt --task-spec example_specs/data_analyst/task.json --state-file path/to/state.json

# --state and --state-file are mutually exclusive; one is required.
```

---

### Running Scripts & Batch Orchestration

The repository includes a central Python engine (`scripts/orchestrator.py`) and thin SLURM shell wrappers in the `scripts/` directory to automate batch benchmarks, iterative system design, and parallel target execution across Docker and Podman environments.

#### 1. Python Orchestrator (`scripts/orchestrator.py`)
The orchestrator manages base and temporary container builds via the Docker Python SDK, dependency parsing from JSON metrics files, execution timeout enforcement, and CSV/text result aggregation.

Run via standard module invocation (`python -m scripts.orchestrator`) or direct script execution (`python scripts/orchestrator.py`):

* **Run Benchmarks:**
  ```bash
  python -m scripts.orchestrator --task benchmark --benchmark mmlu --type ablationC --iterations 1-16
  ```
* **Iterative System Design:**
  ```bash
  python -m scripts.orchestrator --task design --task-spec benchmark/GSMHard/spec/task.json --benchmark gsm --type ablationC --iterations 1-10
  ```
* **Run Target Systems:**
  ```bash
  python -m scripts.orchestrator --task target --system-names data_analyst_gpt5_v0 --state '{"messages": []}'
  # Or with a task specification and state file:
  python -m scripts.orchestrator --task target --task-spec example_specs/data_analyst/task.json --system-names data_analyst_iter1_gpt --state-file path/to/state.json
  ```

#### 2. HPC / SLURM Wrappers
Thin wrappers isolate SLURM `#SBATCH` directives and environment activation from application logic:
* `scripts/slurm_benchmark.sh`: Submits benchmark jobs via `sbatch scripts/slurm_benchmark.sh`.
* `scripts/slurm_design.sh`: Submits system design jobs via `sbatch scripts/slurm_design.sh`.
* `scripts/slurm_target.sh`: Submits target execution jobs via `sbatch scripts/slurm_target.sh`.

---

## Acknowledgments & Citation

This work builds upon the foundational Automated Design of Agentic Systems (ADAS) concept introduced by Hu et al.:
> Hu, S., Lu, C., & Clune, J. (2025). *Automated Design of Agentic Systems*. Published as a conference paper at ICLR 2025. arXiv:2408.08435v2.
