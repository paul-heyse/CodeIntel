Here’s a repo‑aware implementation plan for `test_profile.*` and `behavioral_coverage.*` that plugs cleanly into your existing CodeIntel pipeline (DuckDB schemas, AST/SCIP ingestion, test/coverage analytics, subsystems, docs views, Prefect/MCP). I’ll lean on the architecture + metadata docs you shared.

---

## 0. Where these datasets sit in your pipeline

You already have, per `<repo, commit>`:

* **Tests + coverage edges**: `analytics.test_catalog`, `analytics.test_coverage_edges`, and `analytics.coverage_functions`. 
* **GOIDs + crosswalk**: `core.goids`, `core.goid_crosswalk`. 
* **Subsystems**: `analytics.subsystems`, `analytics.subsystem_modules`.
* **AST index**: `core.ast_nodes` and helpers like `AstSpanIndex`, `parse_python_module`.
* **Test graph metrics**: bipartite test⇄function graph and `analytics.test_graph_metrics_tests` / `analytics.test_graph_metrics_functions`. 

We’re going to add:

* `analytics.test_profile`
* `analytics.behavioral_coverage`

…and a new analytics module that:

* walks tests, coverage and AST to compute per‑test profile fields, and
* assigns behavior tags via heuristics on names/AST/markers (with a hook for future LLM classification).

These will then be surfaced via new `docs.*` views and exported JSONL/Parquet.

---

## 1. New schemas

### 1.1 `analytics.test_profile`

**Goal:** one row per *test* (pytest nodeid). This is the test‑side analogue of `function_profile.*`: identity, execution info, structural metrics, what it covers, and how “risky/flaky” it looks. 

Add to `config/schemas/tables.py`: 

```python
"analytics.test_profile": TableSchema(
    schema="analytics",
    name="test_profile",
    columns=[
        # Identity
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("test_id", "VARCHAR", nullable=False),         # pytest nodeid
        Column("test_goid_h128", "DECIMAL(38,0)"),            # optional GOID for test function
        Column("urn", "VARCHAR"),                             # GOID URN if mapped
        Column("rel_path", "VARCHAR", nullable=False),
        Column("module", "VARCHAR"),                          # dotted module path
        Column("qualname", "VARCHAR"),                        # test function/method qualname
        Column("language", "VARCHAR"),                        # 'python'

        Column("kind", "VARCHAR"),                            # 'function', 'method', 'param_case', etc.

        # Execution & status
        Column("status", "VARCHAR"),                          # passed/failed/error/xfail/skip/...
        Column("duration_ms", "DOUBLE"),
        Column("markers", "JSON"),                            # list[str]
        Column("flaky", "BOOLEAN"),                           # from markers
        Column("last_run_at", "TIMESTAMP"),                   # optional if you track

        # Coverage footprint
        Column("functions_covered", "JSON"),                  # list[{function_goid_h128, urn, module, qualname, coverage_ratio, coverage_share}]
        Column("functions_covered_count", "INTEGER"),
        Column("primary_function_goids", "JSON"),             # list[goid_h128] where coverage_share >= threshold
        Column("subsystems_covered", "JSON"),                 # list[{subsystem_id, name}]
        Column("subsystems_covered_count", "INTEGER"),
        Column("primary_subsystem_id", "VARCHAR"),

        # Structural/AST-based metrics
        Column("assert_count", "INTEGER"),
        Column("raise_count", "INTEGER"),
        Column("uses_parametrize", "BOOLEAN"),
        Column("uses_fixtures", "BOOLEAN"),                   # heuristically from args/markers

        # I/O & external effects
        Column("io_bound", "BOOLEAN"),                        # any of the below
        Column("uses_network", "BOOLEAN"),
        Column("uses_db", "BOOLEAN"),
        Column("uses_filesystem", "BOOLEAN"),
        Column("uses_subprocess", "BOOLEAN"),

        # Quality & flakiness heuristics
        Column("flakiness_score", "DOUBLE"),                  # 0.0–1.0 heuristic
        Column("importance_score", "DOUBLE"),                 # optional: e.g. breadth of coverage
        Column("notes", "VARCHAR"),

        # Test-graph metrics (optional but very useful)
        Column("tg_degree", "INTEGER"),
        Column("tg_weighted_degree", "DOUBLE"),
        Column("tg_proj_degree", "INTEGER"),
        Column("tg_proj_weight", "DOUBLE"),
        Column("tg_proj_clustering", "DOUBLE"),
        Column("tg_proj_betweenness", "DOUBLE"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "test_id"),
    description="Per-test profile combining execution status, coverage footprint, AST metrics, IO usage, and flakiness/importance heuristics.",
)
```

Minimal compliance with your spec is everything up through `flakiness_score`; the `tg_*` fields are a natural join to `analytics.test_graph_metrics_tests` so agents can see which tests are “central”.

### 1.2 `analytics.behavioral_coverage`

**Goal:** map each test to **behavior tags** like `happy_path`, `error_paths`, `edge_cases`, `concurrency`, etc., with provenance (heuristic vs LLM).

The simplest shape is “one row per test with a JSON list of tags”:

```python
"analytics.behavioral_coverage": TableSchema(
    schema="analytics",
    name="behavioral_coverage",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("test_id", "VARCHAR", nullable=False),

        Column("test_goid_h128", "DECIMAL(38,0)"),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("qualname", "VARCHAR"),

        # Behavior tags
        Column("behavior_tags", "JSON", nullable=False),   # list[str]
        Column("tag_source", "VARCHAR", nullable=False),   # 'heuristic', 'llm', 'mixed'
        Column("heuristic_version", "VARCHAR"),            # e.g. 'v1'
        Column("llm_model", "VARCHAR"),                    # if/when used
        Column("llm_run_id", "VARCHAR"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "test_id"),
    description="Behavioral coverage tags per test, with heuristic/LLM provenance.",
)
```

If you later want more granularity, you can add a normalized `analytics.behavioral_coverage_tags` (one row per `(test_id, tag)`), but this wide version is great for LLM consumption and for `docs.*` views.

---

## 2. Data sources & field derivation

### 2.1 Base identity & execution fields

Source: `analytics.test_catalog` + `core.goids`.

* For each row in `analytics.test_catalog` for `<repo, commit>`:

  * `test_id`, `test_goid_h128`, `rel_path`, `qualname`, `status`, `duration_ms`, `markers`, `flaky`, `kind`.

* Left‑join to `core.goids` on `test_goid_h128`:

  * Grab `urn`, `language`, `module`.

These are your base columns in `test_profile`.

### 2.2 `functions_covered` and `subsystems_covered`

Source: `analytics.test_coverage_edges`, `analytics.coverage_functions`, `analytics.subsystem_modules`, `analytics.subsystems`.

1. **Functions covered**

   For each `test_id`:

   ```sql
   WITH per_edge AS (
       SELECT
           e.test_id,
           e.function_goid_h128,
           SUM(e.covered_lines)        AS covered_lines,
           SUM(e.executable_lines)     AS executable_lines
       FROM analytics.test_coverage_edges e
       WHERE e.repo = ? AND e.commit = ?
       GROUP BY e.test_id, e.function_goid_h128
   ),
   per_test_totals AS (
       SELECT
           test_id,
           SUM(covered_lines) AS total_covered_lines
       FROM per_edge
       GROUP BY test_id
   )
   SELECT
       p.test_id,
       p.function_goid_h128,
       p.covered_lines * 1.0 / NULLIF(p.executable_lines, 0) AS coverage_ratio,
       p.covered_lines * 1.0 / NULLIF(t.total_covered_lines, 0) AS coverage_share
   FROM per_edge p
   JOIN per_test_totals t USING (test_id);
   ```

   Join that to `core.goids` to get `urn`, `rel_path`, `module`, `qualname`. Then pack into JSON for each test:

   ```sql
   SELECT
     test_id,
     LIST({
       'function_goid_h128': function_goid_h128,
       'urn': urn,
       'module': module,
       'qualname': qualname,
       'coverage_ratio': coverage_ratio,
       'coverage_share': coverage_share
     }) AS functions_covered,
     COUNT(*) AS functions_covered_count,
     LIST_IF(function_goid_h128, coverage_share >= 0.4) AS primary_function_goids
   FROM ...
   GROUP BY test_id;
   ```

   (Use appropriate DuckDB JSON/LIST construction functions in your actual implementation.)

2. **Subsystems covered**

   * Map each `function_goid_h128` → `module` (from `core.goids` or `analytics.function_profile`).
   * Map each `module` → `subsystem_id` using `analytics.subsystem_modules`. 
   * Aggregate per `(test_id, subsystem_id)` the total `covered_lines` (or `coverage_share`).

   Then, per test:

   ```sql
   SELECT
     test_id,
     LIST({
       'subsystem_id': subsystem_id,
       'name': s.name,
       'coverage_share': coverage_share
     }) AS subsystems_covered,
     COUNT(*) AS subsystems_covered_count,
     FIRST(subsystem_id ORDER BY coverage_share DESC) AS primary_subsystem_id
   FROM (
       SELECT
         e.test_id,
         sm.subsystem_id,
         SUM(e.covered_lines) * 1.0 / NULLIF(t.total_covered_lines, 0) AS coverage_share
       FROM analytics.test_coverage_edges e
       JOIN core.goids g ON g.goid_h128 = e.function_goid_h128
       JOIN analytics.subsystem_modules sm ON sm.module = g.module
       JOIN per_test_totals t ON t.test_id = e.test_id
       JOIN analytics.subsystems s ON s.subsystem_id = sm.subsystem_id
       WHERE e.repo = ? AND e.commit = ?
       GROUP BY e.test_id, sm.subsystem_id
   ) x
   GROUP BY test_id;
   ```

### 2.3 `assert_count` / `raise_count`

Source: AST via `core.ast_nodes` or fresh AST parse, and GOID spans.

You already have:

* `core.ast_nodes` with `node_type`, `path`, `lineno`, `end_lineno`, `parent_qualname`. 
* `core.goids`/`core.goid_crosswalk` with ranges for test functions.

You can calculate purely in SQL:

```sql
WITH test_bounds AS (
    SELECT
        t.test_id,
        t.test_goid_h128,
        g.rel_path,
        g.qualname,
        g.start_line,
        g.end_line
    FROM analytics.test_catalog t
    JOIN core.goids g
      ON g.goid_h128 = t.test_goid_h128
    WHERE t.repo = ? AND t.commit = ?
),
asserts AS (
    SELECT
        b.test_id,
        COUNT(*) AS assert_count
    FROM test_bounds b
    JOIN core.ast_nodes n
      ON n.path = b.rel_path
     AND n.lineno >= b.start_line
     AND n.lineno <= COALESCE(b.end_line, n.lineno)
     AND n.node_type = 'Assert'
    GROUP BY b.test_id
),
raises AS (
    SELECT
        b.test_id,
        COUNT(*) AS raise_count
    FROM test_bounds b
    JOIN core.ast_nodes n
      ON n.path = b.rel_path
     AND n.lineno >= b.start_line
     AND n.lineno <= COALESCE(b.end_line, n.lineno)
     AND n.node_type = 'Raise'
    GROUP BY b.test_id
)
SELECT
    test_id,
    COALESCE(assert_count, 0) AS assert_count,
    COALESCE(raise_count, 0)  AS raise_count
FROM test_bounds
LEFT JOIN asserts USING (test_id)
LEFT JOIN raises USING (test_id);
```

No new parsing needed; you’re reusing `ast_nodes.*`.

If you prefer Python, you can instead use `parse_python_module` + `AstSpanIndex` from `ingestion/ast_utils.py` and walk `ast.Assert`/`ast.Raise` inside each test’s span.

### 2.4 `uses_parametrize`, `uses_fixtures`

Source: `analytics.test_catalog.markers`, `qualname`, plus optional AST.

Heuristics:

* `uses_parametrize`: `test_catalog.kind == 'parametrized_case'` OR `markers` contains `parametrize`/`pytest.mark.parametrize` pattern. 
* `uses_fixtures`: either

  * test function has parameters whose names don’t look like plain data (lazy heuristic), or
  * `markers` contains fixture markers, or
  * AST finds `request.getfixturevalue` calls.

Start with markers + non‑empty arg list; add AST later if needed.

### 2.5 IO & effects flags: `io_bound`, `uses_network`, `uses_db`, `uses_filesystem`, `uses_subprocess`

Source: per‑file AST with import resolution.

You don’t currently have a “per‑file imports” table, but you do have AST + CST and a robust import resolver used by the call‑graph builder.

Implementation pattern (Python, in analytics module):

1. For each *test file* (`rel_path` from `test_catalog`), parse it once via `parse_python_module` / LibCST.

2. Build a simple import map:

   * Walk `ast.Import` & `ast.ImportFrom`:

     * Map local names → canonical library names:
       `import requests as r` → `r -> requests`
       `from sqlalchemy import create_engine` → `create_engine -> sqlalchemy`
       `from .db import Session` → treat as internal (no external flag here).

3. For each test function in that file (bounded by GOID range):

   * Walk its AST nodes; for each `ast.Call`:

     * Identify the base name:

       * `requests.get` → `requests`
       * `client.get` where `client` was assigned from `requests.Session()` → treat as `requests` where possible (you can reuse some import/callsite logic from your callgraph builder if you want, but it’s fine to keep a simpler heuristic here).

4. Compare base library / function names against a static spec, e.g.:

   ```python
   IO_SPEC = {
       "network": {
           "libs": ["requests", "httpx", "urllib3", "aiohttp", "socket", "boto3", "paramiko"],
           "funcs": ["get", "post", "put", "delete", "request", "send"]
       },
       "db": {
           "libs": ["sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"],
           "funcs": ["execute", "session", "commit", "query"]
       },
       "filesystem": {
           "libs": ["pathlib", "os", "shutil"],
           "funcs": ["open", "unlink", "remove", "rmtree", "rename"]
       },
       "subprocess": {
           "libs": ["subprocess"],
           "funcs": ["run", "Popen", "call", "check_call"]
       },
   }
   ```

5. Set flags if any call in that test matches the corresponding lib/func patterns.

6. `io_bound = uses_network OR uses_db OR uses_filesystem OR uses_subprocess`.

This mirrors how you already use heuristics + config for external dependencies and config graphs.

### 2.6 Flakiness & importance scores

Source: `status`, `markers`, IO flags, `duration_ms`, and test‑graph metrics.

Define a simple heuristic in Python:

```python
def compute_flakiness_score(
    *,
    status: str,
    markers: list[str],
    duration_ms: float | None,
    uses_network: bool,
    uses_db: bool,
    uses_filesystem: bool,
    uses_subprocess: bool,
) -> float:
    score = 0.0

    if "flaky" in markers:
        score += 0.6
    if status in ("xfail", "xpass"):
        score += 0.2
    if uses_network:
        score += 0.15
    if uses_db or uses_subprocess:
        score += 0.1
    if uses_filesystem:
        score += 0.05
    if duration_ms and duration_ms > 2000:   # 2s threshold from config
        score += 0.1

    return min(score, 1.0)
```

You can evolve this as you start capturing multiple historical runs (e.g., bump score if status has changed across runs).

For `importance_score`, a good starting point is:

* rescale `tg_weighted_degree` or `functions_covered_count` into [0,1] (e.g. min‑max or log scaling), and/or
* weight by subsystem risk of covered subsystems.

This lets agents prioritize “high‑breadth, high‑impact” tests.

---

## 3. Behavioral coverage heuristics (`behavior_tags`)

We want tags like: `happy_path`, `error_paths`, `edge_cases`, `concurrency`, `db_interaction`, `network_interaction`, `config_behavior`, etc.

### 3.1 Heuristic sources

1. **Names & nodeid**

   * `test_name` and `test_id` string:

     * Contains `happy`, `ok`, `success` → `happy_path`.
     * Contains `error`, `fail`, `invalid`, `exception` → `error_paths`.
     * Contains `edge`, `boundary`, `corner` → `edge_cases`.
     * Contains `concurrent`, `parallel`, `thread`, `async`, `race` → `concurrency`.
     * Contains `db`, `sql`, `transaction` → `db_interaction`.
     * Contains `http`, `request`, `api`, `endpoint` → `network_interaction`.

2. **Markers**

   * `xfail` + reason mentioning bugs → add `known_bug`.
   * Custom markers like `integration`, `e2e`, `slow`, `network`, `db` → map to `integration_scenario`, `io_heavy`, `network_interaction`, `db_interaction`.

3. **IO flags from test_profile**

   * `uses_network` → `network_interaction`.
   * `uses_db` → `db_interaction`.
   * `uses_filesystem` → `filesystem_interaction`.
   * `uses_subprocess` → `process_interaction`.

4. **AST pattern hints (optional but cheap)**

   Using the same AST we already parsed for asserts:

   * `pytest.raises` or `with pytest.raises` → `error_paths`.
   * Multiple assertions involving boundaries (e.g., `>=`, `<=`, “max”, “min” in assert messages) → `edge_cases`.
   * `async def` or use of `asyncio`, `trio`, `anyio`, or `threading` → `concurrency`.

### 3.2 Implementation sketch

In `analytics/behavioral_coverage.py`:

```python
BEHAVIOR_TAGS = {
    "happy_path",
    "error_paths",
    "edge_cases",
    "concurrency",
    "network_interaction",
    "db_interaction",
    "filesystem_interaction",
    "process_interaction",
    "integration_scenario",
    "io_heavy",
    "known_bug",
}

def infer_behavior_tags(test_row: TestProfileRow, ast_info: AstTestInfo) -> list[str]:
    tags: set[str] = set()

    name = test_row.qualname or test_row.test_id
    lower_name = name.lower()
    markers = [m.lower() for m in (test_row.markers or [])]

    # Names
    if any(k in lower_name for k in ["happy", "ok", "success"]):
        tags.add("happy_path")
    if any(k in lower_name for k in ["error", "fail", "invalid", "exception"]):
        tags.add("error_paths")
    if any(k in lower_name for k in ["edge", "boundary", "corner"]):
        tags.add("edge_cases")
    if any(k in lower_name for k in ["concurrent", "parallel", "thread", "async", "race"]):
        tags.add("concurrency")

    # Markers
    if "xfail" in markers:
        tags.add("known_bug")
    if "integration" in markers or "e2e" in markers:
        tags.add("integration_scenario")
    if "slow" in markers:
        tags.add("io_heavy")
    if "network" in markers:
        tags.add("network_interaction")
    if "db" in markers or "database" in markers:
        tags.add("db_interaction")

    # IO flags
    if test_row.uses_network:
        tags.add("network_interaction")
    if test_row.uses_db:
        tags.add("db_interaction")
    if test_row.uses_filesystem:
        tags.add("filesystem_interaction")
    if test_row.uses_subprocess:
        tags.add("process_interaction")
    if test_row.io_bound:
        tags.add("io_heavy")

    # AST hints
    if ast_info.uses_pytest_raises:
        tags.add("error_paths")
    if ast_info.uses_concurrency_lib:
        tags.add("concurrency")
    if ast_info.has_boundary_asserts:
        tags.add("edge_cases")

    return sorted(tags)
```

Write a small `AstTestInfo` struct when you walk AST for asserts/IO to capture these booleans.

### 3.3 LLM classification hook (Phase 2)

Once heuristics are in place, you can add a **secondary pass** that:

1. Materializes a candidate view:

   ```sql
   CREATE OR REPLACE VIEW analytics.v_behavioral_classification_input AS
   SELECT
       t.repo,
       t.commit,
       t.test_id,
       t.rel_path,
       t.qualname,
       source_code,  -- from joining to core.ast_nodes / file contents
       p.functions_covered,
       p.subsystems_covered,
       p.assert_count,
       p.markers
   FROM analytics.test_profile p
   JOIN analytics.test_catalog t USING (repo, commit, test_id);
   ```

2. Feeds each row to an offline LLM script (not in CodeIntel core) that returns a list of tags.

3. Writes those into `analytics.behavioral_coverage` with `tag_source = 'llm'` or `'mixed'`.

For now, you can keep `tag_source = 'heuristic'` everywhere and leave the LLM path as a documented extension.

---

## 4. Analytics implementation module

Create a new module, e.g. `analytics/test_profiles.py`, that handles both `test_profile` and the heuristic part of `behavioral_coverage`.

### 4.1 Config models

In `config/models.py` add:

```python
class TestProfileConfig(BaseModel):
    repo: str
    commit: str
    slow_test_threshold_ms: float = 2000.0
    io_spec: dict[str, Any] | None = None  # fallback to defaults if None


class BehavioralCoverageConfig(BaseModel):
    repo: str
    commit: str
    heuristic_version: str = "v1"
```

You can load `io_spec` from YAML or use a built‑in default.

### 4.2 `build_test_profile(con, cfg)`

High‑level flow:

```python
def build_test_profile(con: duckdb.DuckDBPyConnection, cfg: TestProfileConfig) -> None:
    ensure_schema(con, "analytics.test_profile")

    con.execute(
        "DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    # 1. Load tests
    tests = con.execute("""
        SELECT
            t.test_id,
            t.test_goid_h128,
            t.repo,
            t.commit,
            t.rel_path,
            t.qualname,
            t.kind,
            t.status,
            t.duration_ms,
            t.markers,
            t.flaky,
            g.urn,
            g.language,
            g.qualname AS goid_qualname,
            g.module
        FROM analytics.test_catalog t
        LEFT JOIN core.goids g
          ON g.goid_h128 = t.test_goid_h128
        WHERE t.repo = ? AND t.commit = ?
    """, [cfg.repo, cfg.commit]).fetchall()

    # 2. Precompute per-test coverage + subsystems via SQL CTEs (as above)
    functions_cov = load_functions_covered(con, cfg)
    subsystems_cov = load_subsystems_covered(con, cfg)

    # 3. Precompute assert/raise counts and AST hints
    ast_index = build_test_ast_index(con, cfg)  # or compute lazily per file

    # 4. Load test graph metrics (optional)
    test_graph_metrics = load_test_graph_metrics(con, cfg)

    # 5. Loop over tests, compute IO flags, flakiness_score, importance_score
    rows: list[TestProfileRow] = []
    for t in tests:
        ast_info = ast_index.get((t.rel_path, t.qualname))
        io_flags = infer_io_flags(ast_info, cfg.io_spec or DEFAULT_IO_SPEC)
        flakiness_score = compute_flakiness_score(
            status=t.status,
            markers=t.markers or [],
            duration_ms=t.duration_ms,
            **io_flags,
        )
        importance_score = compute_importance_score(
            functions_cov.get(t.test_id),
            test_graph_metrics.get(t.test_id),
        )

        row = TestProfileRow(
            repo=cfg.repo,
            commit=cfg.commit,
            test_id=t.test_id,
            test_goid_h128=t.test_goid_h128,
            urn=t.urn,
            rel_path=t.rel_path,
            module=t.module,
            qualname=t.qualname or t.goid_qualname,
            language=t.language or "python",
            kind=t.kind,
            status=t.status,
            duration_ms=t.duration_ms,
            markers=t.markers,
            flaky=t.flaky,
            last_run_at=datetime.utcnow(),
            functions_covered=functions_cov.get(t.test_id, {}).get("functions_covered", []),
            functions_covered_count=functions_cov.get(t.test_id, {}).get("count", 0),
            primary_function_goids=functions_cov.get(t.test_id, {}).get("primary_goids", []),
            subsystems_covered=subsystems_cov.get(t.test_id, {}).get("subsystems", []),
            subsystems_covered_count=subsystems_cov.get(t.test_id, {}).get("count", 0),
            primary_subsystem_id=subsystems_cov.get(t.test_id, {}).get("primary_subsystem_id"),
            assert_count=ast_info.assert_count,
            raise_count=ast_info.raise_count,
            uses_parametrize=ast_info.uses_parametrize,
            uses_fixtures=ast_info.uses_fixtures,
            io_bound=io_flags["io_bound"],
            uses_network=io_flags["uses_network"],
            uses_db=io_flags["uses_db"],
            uses_filesystem=io_flags["uses_filesystem"],
            uses_subprocess=io_flags["uses_subprocess"],
            flakiness_score=flakiness_score,
            importance_score=importance_score,
            tg_degree=test_graph_metrics.get(t.test_id, {}).get("degree"),
            tg_weighted_degree=test_graph_metrics.get(t.test_id, {}).get("weighted_degree"),
            tg_proj_degree=test_graph_metrics.get(t.test_id, {}).get("proj_degree"),
            tg_proj_weight=test_graph_metrics.get(t.test_id, {}).get("proj_weight"),
            tg_proj_clustering=test_graph_metrics.get(t.test_id, {}).get("proj_clustering"),
            tg_proj_betweenness=test_graph_metrics.get(t.test_id, {}).get("proj_betweenness"),
            created_at=datetime.utcnow(),
        )
        rows.append(row)

    insert_test_profile_rows(con, rows)
```

The helper functions:

* `load_functions_covered` / `load_subsystems_covered` – run the SQL CTEs we sketched and return dicts keyed by `test_id`.
* `build_test_ast_index` – pre‑parse each test file once, build `AstSpanIndex` per file, and compute `AstTestInfo` for each test (assert/raise counts, pytest.raises, concurrency libs, etc.).
* `load_test_graph_metrics` – read `analytics.test_graph_metrics_tests` keyed by `test_id`.
* `insert_test_profile_rows` – use your existing row→tuple helpers from `models/rows.py`. 

### 4.3 `build_behavioral_coverage(con, cfg)`

In the same module or a sibling:

```python
def build_behavioral_coverage(con: duckdb.DuckDBPyConnection, cfg: BehavioralCoverageConfig) -> None:
    ensure_schema(con, "analytics.behavioral_coverage")

    con.execute(
        "DELETE FROM analytics.behavioral_coverage WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    rows: list[BehavioralCoverageRow] = []

    # Join test_profile with minimal AST hints (if needed)
    tests = con.execute("""
        SELECT
            p.repo, p.commit, p.test_id,
            p.test_goid_h128,
            p.rel_path,
            p.qualname,
            p.markers,
            p.uses_network,
            p.uses_db,
            p.uses_filesystem,
            p.uses_subprocess,
            a.assert_count,
            a.raise_count
        FROM analytics.test_profile p
        LEFT JOIN analytics.test_profile a
          ON a.repo = p.repo AND a.commit = p.commit AND a.test_id = p.test_id
        WHERE p.repo = ? AND p.commit = ?
    """, [cfg.repo, cfg.commit]).fetchall()

    ast_index = build_test_ast_index_for_behavior(con, cfg)   # can reuse from profile build

    for t in tests:
        ast_info = ast_index.get((t.rel_path, t.qualname))
        tags = infer_behavior_tags(t, ast_info)
        row = BehavioralCoverageRow(
            repo=cfg.repo,
            commit=cfg.commit,
            test_id=t.test_id,
            test_goid_h128=t.test_goid_h128,
            rel_path=t.rel_path,
            qualname=t.qualname,
            behavior_tags=tags,
            tag_source="heuristic",
            heuristic_version=cfg.heuristic_version,
            created_at=datetime.utcnow(),
        )
        rows.append(row)

    insert_behavioral_rows(con, rows)
```

Later, you can layer an LLM classification step that updates `behavior_tags` and `tag_source` to `"mixed"` or `"llm"`.

---

## 5. Orchestration integration

You already drive everything via `orchestration/steps.py` and a Prefect flow. 

### 5.1 New steps

In `orchestration/steps.py`:

```python
@dataclass
class TestProfileStep:
    name: str = "test_profile"
    # Needs tests, coverage edges, subsystem mapping, and test graph metrics
    deps: Sequence[str] = (
        "tests",                    # builds analytics.test_catalog
        "coverage_analytics",       # builds coverage_functions + test_coverage_edges
        "subsystems",               # builds analytics.subsystem_modules
        "test_graph_metrics",       # if you have this as a step
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = TestProfileConfig(repo=ctx.repo, commit=ctx.commit)
        build_test_profile(con, cfg)


@dataclass
class BehavioralCoverageStep:
    name: str = "behavioral_coverage"
    deps: Sequence[str] = ("test_profile",)    # must run after TestProfileStep

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = BehavioralCoverageConfig(repo=ctx.repo, commit=ctx.commit)
        build_behavioral_coverage(con, cfg)
```

Insert these steps into your step graph after the existing coverage+tests+subsystems steps and before docs export / risk (or alongside, since they’re independent of risk).

### 5.2 Prefect flow tasks

In `orchestration/prefect_flow.py`, add tasks that simply call these `run` methods or wrap `build_test_profile` / `build_behavioral_coverage` directly, mirroring the pattern used for other analytics steps.

---

## 6. Docs export & views

### 6.1 Export to JSONL/Parquet

In `docs_export/export_jsonl.py` / `export_parquet.py`, extend the dataset map:

```python
DATASETS.update({
    "analytics.test_profile": "test_profile",
    "analytics.behavioral_coverage": "behavioral_coverage",
})
```

This will produce:

* `test_profile.jsonl` / `.parquet`
* `behavioral_coverage.jsonl` / `.parquet`

in `Document Output/`, alongside your existing metadata outputs. 

### 6.2 `docs.v_test_architecture` (LLM‑friendly view)

In `storage/views.py`, define a denormalized test view, analogous to `docs.v_function_architecture` / `docs.v_module_architecture`:

```sql
CREATE OR REPLACE VIEW docs.v_test_architecture AS
SELECT
    p.repo,
    p.commit,
    p.test_id,
    p.test_goid_h128,
    p.urn,
    p.rel_path,
    p.module,
    p.qualname,
    p.language,
    p.kind,

    -- Execution
    p.status,
    p.duration_ms,
    p.markers,
    p.flaky,
    p.flakiness_score,
    p.importance_score,

    -- Coverage footprint
    p.functions_covered,
    p.functions_covered_count,
    p.primary_function_goids,
    p.subsystems_covered,
    p.subsystems_covered_count,
    p.primary_subsystem_id,

    -- Structural
    p.assert_count,
    p.raise_count,
    p.uses_parametrize,
    p.uses_fixtures,

    -- IO / effects
    p.io_bound,
    p.uses_network,
    p.uses_db,
    p.uses_filesystem,
    p.uses_subprocess,

    -- Test graph metrics
    p.tg_degree,
    p.tg_weighted_degree,
    p.tg_proj_degree,
    p.tg_proj_weight,
    p.tg_proj_clustering,
    p.tg_proj_betweenness,

    -- Behavioral tags
    b.behavior_tags,
    b.tag_source,
    b.heuristic_version,
    b.llm_model,
    b.llm_run_id,

    p.created_at
FROM analytics.test_profile p
LEFT JOIN analytics.behavioral_coverage b
  ON b.repo = p.repo
 AND b.commit = p.commit
 AND b.test_id = p.test_id;
```

Agents can now ask “show me error‑path tests covering subsystem X that look flaky and IO‑bound” with a single query over this view.

---

## 7. Testing strategy

### 7.1 Unit tests (Python)

Create `tests/analytics/test_test_profiles.py` with:

* **AST + IO heuristics tests**:

  * Construct minimal test files with various imports (`requests`, `sqlalchemy`, `subprocess`, `open`) and assert that `infer_io_flags` sets the correct booleans.

* **Assert/raise counting**:

  * Use in‑memory `core.ast_nodes` fixtures or parse sample files, and test that `assert_count`/`raise_count` match expected numbers.

* **Behavior tag inference**:

  * Feed in synthetic `TestProfileRow` + `AstTestInfo` and assert that:

    * `test_error_on_invalid_input` → includes `error_paths`, `edge_cases` when asserts use boundary conditions.
    * `test_happy_path_works` → `happy_path`.

### 7.2 Integration tests (DuckDB)

* Use a tiny synthetic repo with:

  * A couple of unit tests and integration tests.
  * Tests that clearly use network, DB, filesystem, subprocess.

* Run the full ingest + graphs + analytics pipeline (or just the relevant steps) against it.

* Assert:

  ```sql
  SELECT COUNT(*) FROM analytics.test_profile WHERE repo = 'synthetic';
  ```

  is equal to number of tests, and that expected columns (e.g., `uses_network`, `behavior_tags`) are set for specific `test_id`s.

* Verify JSONL export:

  * `test_profile.jsonl` and `behavioral_coverage.jsonl` exist under `Document Output/`.
  * `jq '.[0] | {test_id, behavior_tags}'` returns the expected structure.

---

## 8. How this helps your agent

With these two datasets wired in:

* For a **planned code change**, the agent can:

  * Find functions in the impact slice → see which tests cover them via existing `test_coverage_edges.*`. 
  * Then query `docs.v_test_architecture` to prioritize tests that:

    * are high‑importance (`importance_score`),
    * cover relevant subsystems, and
    * exercise relevant behaviors (e.g., `error_paths`).

* When **adding new tests**, the agent can:

  * Look at `behavioral_coverage` for existing tests on a function/subsystem and see which tags are missing (e.g., no `edge_cases` tags) and suggest new tests to fill those gaps.

* For **flakiness remediation**, the agent can:

  * Filter to tests with high `flakiness_score` and `io_bound = true` and propose isolation strategies (mock IO, split tests, etc.), using the IO flags and behavior tags to explain *why* they might be flaky.

All of this builds on the ingestion, graphs, analytics, and docs‑view machinery you already have—just adding the “per‑test profile” and “what behaviors does this test actually exercise?” perspectives that were missing.
