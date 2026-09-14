# Data Analyst Agent Specification

This example task specification defines an automated data analyst system designed to inspect CSV and JSON datasets, dynamically write and execute analysis scripts using `pandas` and `matplotlib`, self-heal upon execution traceback errors, save data/plot artifacts, and provide analytical summaries.

## File Fixtures

The TaskSpec generates the following input fixtures for individual test cases and mounts the selected files under `ADAS_INPUT_DIR` (the logical `input/` directory). The agent must discover the available files at runtime rather than rely on fixed filenames.

* `sales.csv` — clean sales data for baseline aggregation and visualization.
* `customers.json` — customer records for multi-file join analysis.
* `messy_sales.csv` — sales data with missing, duplicate, and inconsistently typed values.
* `schema_variant.json` — valid JSON with an alternative schema.
* `malformed_records.csv` — sparse records and data-level anomalies for resilience testing.

Generated TXT, CSV, JSON, and plot artifacts must be written beneath `ADAS_OUTPUT_DIR` (the logical `output/` directory).

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/data_analyst_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/data_analyst_agent/task.json
   ```

3. **Run Design Optimization (Sandboxed Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/data_analyst_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name data_analyst_v0 --task-spec example_specs/data_analyst_agent/task.json --state '{"analysis_task": "Plot distribution of total_amount"}'
   ```
