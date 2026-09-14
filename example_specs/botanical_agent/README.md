# Botanical Agent Specification

This example task specification defines a single-turn botanical expert and companion-planting agent. The target system inspects an embedded SQLite database, performs read-only queries, and returns grounded horticultural guidance with scientific names, care requirements, and compatible or antagonistic plant relationships.

## Database Fixture

The setup stage creates `data/botanical_garden.db` beneath `ADAS_INPUT_DIR`. It contains plant records, companion relationships, and botanical care or ecological facts. The agent must discover the schema at runtime and use read-only SQL queries rather than assuming a fixed table layout.

The target state accepts `user_request` and returns `sql_queries`, `retrieved_records`, and `final_answer`.

## Running the Lifecycle for this Task

1. **Setup Fixtures & Preflight (Sandboxed):**

   ```bash
   python create_setup.py --task-spec example_specs/botanical_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**

   ```bash
   python create_validation.py --task-spec example_specs/botanical_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**

   ```bash
   python invoke_design.py --task-spec example_specs/botanical_agent/task.json
   ```

4. **Execute Target System in Sandbox:**

   ```bash
   python invoke_target.py --system-name botanical_agent --task-spec example_specs/botanical_agent/task.json --state '{"user_request": "Which companion plants are beneficial for tomatoes?"}'
   ```
