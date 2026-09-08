# Data Analyst Agent Specification

This example task specification defines an automated data analyst system designed to inspect CSV and JSON datasets, dynamically write and execute analysis scripts using `pandas` and `matplotlib`, self-heal upon execution traceback errors, save data/plot artifacts, and provide analytical summaries.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/data_analyst/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/data_analyst/task.json
   ```

3. **Run Design Optimization (Sandboxed Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/data_analyst/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name data_analyst_v0 --task-spec example_specs/data_analyst/task.json --state '{"analysis_task": "Plot distribution of total_amount"}'
   ```
