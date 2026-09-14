# IMDb Movie Agent Specification

This example task specification exercises ADAS's external database resource support (`resource_manifest.available_resources`). The target system is a single-turn movie expert and recommendation agent connected to the real-world, public IMDb relational database (`imdb_ijs` hosted at `relational.fel.cvut.cz`).

## Database Context & Resources

Rather than bundling or seeding a local 470+ MB database, the agent accesses a live, read-only MariaDB instance containing 7 tables and over 5.6 million records:
* `movies` (388k rows): `id`, `name`, `year`, `rank`
* `actors` (817k rows): `id`, `first_name`, `last_name`, `gender`
* `directors` (86k rows): `id`, `first_name`, `last_name`
* `roles` (3.4M rows): `actor_id`, `movie_id`, `role`
* `movies_directors` (371k rows): `director_id`, `movie_id`
* `movies_genres` (395k rows): `movie_id`, `genre`
* `directors_genres` (156k rows): `director_id`, `genre`, `prob`

The agent is provided with tools to inspect schema (`describe_table`) and execute read-only SQL queries (`execute_sql_query`) using credentials configured via the `IMDB_DATABASE_URI` environment variable. Because the database contains millions of records, the agent must generate efficient SQL queries utilizing indexed joins and appropriate `LIMIT` clauses.

## Running the Lifecycle for this Task

Set the required database connection variable in the project `.env` file:

```dotenv
IMDB_DATABASE_URI=mysql+pymysql://guest:ctu-relational@relational.fel.cvut.cz:3306/imdb_ijs
```

1. **Setup Fixtures & Preflight (Sandboxed):**
   ```bash
   python create_setup.py --task-spec example_specs/movie_agent/task.json
   ```

2. **Generate Frozen Programmatic Validation Logic:**
   ```bash
   python create_validation.py --task-spec example_specs/movie_agent/task.json
   ```

3. **Run Design Optimization (Meta-Agent):**
   ```bash
   python invoke_design.py --task-spec example_specs/movie_agent/task.json
   ```

4. **Execute Target System in Sandbox:**
   ```bash
   python invoke_target.py --system-name imdb_movie_expert_agent --task-spec example_specs/movie_agent/task.json --state '{"user_request": "Could you recommend 3 mind-bending thriller movies like Memento?"}'
   ```
