Here’s a concrete, repo‑aware implementation plan for **entrypoints** and **external dependencies**, wired into your existing CodeIntel pipeline (DuckDB schemas, ingestion, graphs, analytics, docs_export, Prefect steps, MCP). I’ll lean on the architecture you documented (GOIDs, call graph, coverage, config_values, subsystems, NetworkX graph layer) so we reuse as much as possible.

I’ll structure it as:

1. Data model & schemas
2. How to detect and populate `entrypoints.*`
3. How to detect and populate `external_dependencies.*`
4. Orchestration, exports, docs views
5. Testing strategy

---

## 1. Data model & schemas

We’ll add **four** tables under `analytics.*`:

* `analytics.entrypoints` – one row per external entrypoint.
* `analytics.entrypoint_tests` – bipartite edges entrypoint ↔ tests.
* `analytics.external_dependencies` – one row per external dep (e.g. “boto3”, “redis”).
* `analytics.external_dependency_calls` – edges dependency ↔ function GOID.

All will be driven off existing `core.*`, `graph.*`, `analytics.*`, and `subsystems.*` tables.

### 1.1 `analytics.entrypoints`

Add to `config/schemas/tables.py`: 

```python
"analytics.entrypoints": TableSchema(
    schema="analytics",
    name="entrypoints",
    columns=[
        # identity
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("entrypoint_id", "VARCHAR", nullable=False),  # stable hash/URN
        Column("kind", "VARCHAR", nullable=False),           # 'http', 'cli', 'cron', 'event', 'rpc', 'websocket', etc.

        # handler
        Column("handler_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("handler_urn", "VARCHAR", nullable=False),
        Column("handler_rel_path", "VARCHAR", nullable=False),
        Column("handler_module", "VARCHAR", nullable=False),
        Column("handler_qualname", "VARCHAR", nullable=False),

        # HTTP-specific
        Column("http_method", "VARCHAR"),                    # GET/POST/...
        Column("route_path", "VARCHAR"),
        Column("status_codes", "JSON"),                      # [200, 201, 400]
        Column("auth_required", "BOOLEAN"),

        # CLI-specific
        Column("command_name", "VARCHAR"),
        Column("arguments_schema", "JSON"),                  # structured arg spec

        # Job-specific
        Column("schedule", "VARCHAR"),                       # cron string, interval, etc.
        Column("trigger", "VARCHAR"),                        # e.g. 'cron', 'interval', 'event'

        # Architecture context
        Column("subsystem_id", "VARCHAR"),                   # from analytics.subsystems
        Column("subsystem_name", "VARCHAR"),
        Column("tags", "JSON"),                              # from modules/tags_index
        Column("owners", "JSON"),

        # Tests/coverage summary (from test_coverage_edges + coverage_functions)
        Column("tests_touching", "INTEGER"),
        Column("failing_tests", "INTEGER"),
        Column("slow_tests", "INTEGER"),
        Column("flaky_tests", "INTEGER"),
        Column("entrypoint_coverage_ratio", "DOUBLE"),       # coverage on handler function
        Column("last_test_status", "VARCHAR"),               # 'all_passing', 'some_failing', 'untested', etc.

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "entrypoint_id"),
    description="External entrypoints (HTTP routes, CLI commands, jobs) mapped to handler GOIDs and tests.",
)
```

### 1.2 `analytics.entrypoint_tests`

This gives you explicit edges from entrypoints to tests (useful for impact analysis, networkx bipartite graphs, etc.).

```python
"analytics.entrypoint_tests": TableSchema(
    schema="analytics",
    name="entrypoint_tests",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("entrypoint_id", "VARCHAR", nullable=False),
        Column("test_id", "VARCHAR", nullable=False),
        Column("test_goid_h128", "DECIMAL(38,0)"),
        Column("coverage_ratio", "DOUBLE"),   # how much of the handler this test covers
        Column("status", "VARCHAR"),          # from analytics.test_catalog
        Column("duration_ms", "DOUBLE"),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "entrypoint_id", "test_id"),
    description="Edges mapping entrypoints to tests that exercise their handlers.",
)
```

### 1.3 `analytics.external_dependencies`

High‑level view per dependency (library/service):

```python
"analytics.external_dependencies": TableSchema(
    schema="analytics",
    name="external_dependencies",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("dep_id", "VARCHAR", nullable=False),

        Column("library", "VARCHAR", nullable=False),         # e.g. 'boto3'
        Column("service_name", "VARCHAR"),                    # e.g. 'aws_s3'; may be same as library
        Column("category", "VARCHAR"),                        # 'db', 'cache', 'queue', 'http_api', etc.

        # Usage aggregates
        Column("function_count", "INTEGER", nullable=False),
        Column("callsite_count", "INTEGER", nullable=False),
        Column("modules_json", "JSON", nullable=False),       # list of modules using this dep
        Column("usage_modes", "JSON", nullable=False),        # ['read', 'write', ...]
        Column("config_keys", "JSON"),                        # from analytics.config_values
        Column("risk_level", "VARCHAR"),                      # optional aggregate based on risk of usage sites

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "dep_id"),
    description="External libraries / services used by the repo and their aggregate usage characteristics.",
)
```

### 1.4 `analytics.external_dependency_calls`

Edges from functions to dependencies (like a bipartite graph compressed to rows).

```python
"analytics.external_dependency_calls": TableSchema(
    schema="analytics",
    name="external_dependency_calls",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("dep_id", "VARCHAR", nullable=False),
        Column("library", "VARCHAR", nullable=False),
        Column("service_name", "VARCHAR"),

        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("function_urn", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=False),
        Column("qualname", "VARCHAR", nullable=False),

        Column("callsite_count", "INTEGER", nullable=False),
        Column("modes", "JSON", nullable=False),            # ['read', 'write', 'admin']
        Column("evidence_json", "JSON"),                    # snippet of AST or decorator info

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "dep_id", "function_goid_h128"),
    description="Function-level callsites into external dependencies with modes and evidence.",
)
```

---

## 2. Detecting & populating `entrypoints.*`

### 2.1 Overall approach

We’ll implement `analytics/entrypoints.py` that:

1. Uses **LibCST/AST** to find framework‑specific entrypoint patterns.

2. Maps discovered handler spans to **GOIDs** via `core.goids` / `core.goid_crosswalk`. 

3. Enriches each entrypoint with:

   * Subsystem ID from `analytics.subsystem_modules`.
   * Tests/coverage from `analytics.test_coverage_edges`, `analytics.test_catalog`, `analytics.coverage_functions`. 
   * Tags/owners from `core.modules` and `analytics.tags_index`.

4. Writes into `analytics.entrypoints` + `analytics.entrypoint_tests`.

We’ll start with framework‑aware detection for:

* **HTTP APIs** – FastAPI/Starlette, Flask (optionally Django).
* **CLI** – click, typer, argparse‑style main functions.
* **Jobs** – simple patterns for `APScheduler`, `schedule`, or custom “cron” tags.

We can fold in more later via a plug‑in registry.

### 2.2 Config model

In `config/models.py`, add:

```python
@dataclass(frozen=True)
class EntryPointsConfig:
    repo: str
    commit: str

    # detection toggles
    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "EntryPointsConfig":
        return cls(repo=repo, commit=commit)
```

Align this with other analytics config types (CoverageAnalyticsStepConfig, GraphMetricsStepConfig, etc.). 

### 2.3 AST/CST detection helpers

Leverage your **CST utilities** and module scanner so we don’t reinvent parsing.

Create `analytics/entrypoint_detectors.py` with:

* Base interface:

  ```python
  @dataclass
  class EntryPointCandidate:
      kind: str              # 'http' | 'cli' | 'cron' | ...
      rel_path: str
      module: str
      qualname: str          # function qualname
      lineno: int
      end_lineno: int | None

      # HTTP-specific
      http_method: str | None = None
      route_path: str | None = None
      status_codes: list[int] | None = None
      auth_required: bool | None = None

      # CLI-specific
      command_name: str | None = None
      arguments_schema: dict[str, Any] | None = None

      # Job-specific
      schedule: str | None = None
      trigger: str | None = None

      evidence: dict[str, Any] = field(default_factory=dict)
  ```

* Per‑framework detectors that accept a CST module + module metadata and yield `EntryPointCandidate` instances.

Examples (at planning level):

#### 2.3.1 FastAPI / Starlette HTTP routes

Using LibCST (you already use it in `ingestion/cst_extract.py` and `ingestion/cst_utils.py`):

* Heuristics:

  * Identify FastAPI app variables: assignment `app = FastAPI(...)` or `app = fastapi.FastAPI(...)`.
  * For any function with decorator `@app.get(...)`, `@app.post(...)`, etc.:

    * `kind = "http"`.
    * `http_method` from decorator function name (`get`, `post`, `put`, `delete`, `patch`, `options`).
    * `route_path` from first positional argument or `path=` kwarg.
    * `status_codes` from `status_code=` kwarg or default `[200]`.
    * `auth_required` if decorator stack includes `@login_required`, `@router.get(..., dependencies=[Depends(auth)])`, or tags (configurable).
    * `qualname` from existing AST/CST qname logic.

#### 2.3.2 Flask routes

* Look for decorators `@app.route("/path", methods=["GET","POST"])` (or `blueprint.route`):

  * `http_method` from `methods` list (if missing, default `["GET"]`).
  * Same as above for `route_path`.
  * `auth_required` if decorator list includes `@login_required` or similar.

#### 2.3.3 Click CLI commands

* Recognize `@click.command()` or `@click.group()` decorated functions:

  * `kind = "cli"`.
  * `command_name` from decorator `name` kwarg or function name.
  * `arguments_schema` from:

    * Param decorators: `@click.option("--foo", type=int, default=...)` etc.
    * Extract each option’s flags, type, required/optional into a JSON schema.

#### 2.3.4 Typer CLI commands

* Recognize `app = typer.Typer()`, `@app.command()` decorated functions:

  * `command_name` from decorator or function name.
  * `arguments_schema` from function parameters + type hints (you already know how to inspect signatures for `function_types.*`). 

#### 2.3.5 Jobs / cron

Start simple:

* Recognize `@scheduler.scheduled_job("cron", ...)` or `@app.on_event("startup")` patterns (APScheduler, FastAPI events).
* Recognize `schedule.every(...).do(func)` style calls.
* Set `kind = "cron"` or `kind = "event_handler"`, `schedule` from decorator arguments, `trigger` from `cron`/`interval`.

This is enough to bootstrap; more frameworks can be added as new detectors.

### 2.4 Mapping entrypoint candidates → GOIDs

Using the spans from `EntryPointCandidate` and `core.goids` / `core.goid_crosswalk`: 

1. For each candidate:

   * `rel_path` + (`lineno`, `end_lineno`, `module`, `qualname`).

2. Query `core.goids` directly:

   ```sql
   SELECT goid_h128, urn, rel_path, qualname
   FROM core.goids
   WHERE rel_path = ?
     AND kind IN ('function', 'method')
     AND start_line <= ?
     AND (end_line IS NULL OR end_line >= ?)
   ```

   If multiple GOIDs match, prefer one with matching `qualname`.

3. Enrich with module + tags/owners by joining `core.modules` on `rel_path` → `module`. 

4. Compute `entrypoint_id` as a stable hash:

   ```python
   raw = f"{repo}:{commit}:{kind}:{handler_urn}:{http_method or ''}:{route_path or ''}:{command_name or ''}:{schedule or ''}"
   entrypoint_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
   ```

### 2.5 Attaching subsystem_id

Leverage your **subsystems** work: modules → subsystems mapping.

* Join `handler_module` to `analytics.subsystem_modules`:

  ```sql
  SELECT subsystem_id
  FROM analytics.subsystem_modules
  WHERE repo = ? AND commit = ? AND module = ?
  ```

* Then join to `analytics.subsystems` to get `subsystem_name` and risk stats:

  ```sql
  SELECT name
  FROM analytics.subsystems
  WHERE repo = ? AND commit = ? AND subsystem_id = ?
  ```

If no subsystem info (e.g. early run), leave fields NULL.

### 2.6 Attaching tests & coverage

You already have:

* `analytics.coverage_functions` (per function GOID coverage). 
* `analytics.test_coverage_edges` (test ↔ function edges). 
* `analytics.test_catalog` (test metadata). 

Use them to fill `entrypoints.tests_*` and `entrypoint_tests.*`.

**Aggregated tests per entrypoint:**

For each handler GOID:

```sql
WITH edges AS (
    SELECT e.test_id, e.coverage_ratio
    FROM analytics.test_coverage_edges e
    WHERE e.function_goid_h128 = :handler_goid
),
joined AS (
    SELECT
        edges.test_id,
        edges.coverage_ratio,
        tc.status,
        tc.duration_ms,
        tc.flaky
    FROM edges
    LEFT JOIN analytics.test_catalog tc
       ON edges.test_id = tc.test_id
)
SELECT
    COUNT(DISTINCT test_id) AS tests_touching,
    COUNT(DISTINCT CASE WHEN status IN ('failed','error') THEN test_id END) AS failing_tests,
    COUNT(DISTINCT CASE WHEN duration_ms > 1000 THEN test_id END) AS slow_tests,
    COUNT(DISTINCT CASE WHEN flaky THEN test_id END) AS flaky_tests,
    MAX(status) AS last_test_status
FROM joined;
```

Use the resulting fields to populate the `analytics.entrypoints` row.

**Per‑entrypoint edges:**

Insert into `analytics.entrypoint_tests` by reusing `joined` CTE:

```sql
INSERT INTO analytics.entrypoint_tests (
    repo, commit, entrypoint_id, test_id, test_goid_h128,
    coverage_ratio, status, duration_ms, created_at
)
SELECT
    :repo, :commit, :entrypoint_id, j.test_id, tc.test_goid_h128,
    j.coverage_ratio, j.status, j.duration_ms, :now
FROM joined j
LEFT JOIN analytics.test_catalog tc
  ON j.test_id = tc.test_id;
```

**Coverage for handler:**

Just look up `analytics.coverage_functions`:

```sql
SELECT coverage_ratio
FROM analytics.coverage_functions
WHERE function_goid_h128 = :handler_goid;
```

Store as `entrypoint_coverage_ratio`.

### 2.7 Implementation: `analytics/entrypoints.py`

Skeleton:

```python
# analytics/entrypoints.py

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

import duckdb

from codeintel.config.models import EntryPointsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.analytics.entrypoint_detectors import (
    detect_http_entrypoints,
    detect_cli_entrypoints,
    detect_job_entrypoints,
    EntryPointCandidate,
)

log = logging.getLogger(__name__)


def build_entrypoints(con: duckdb.DuckDBPyConnection, cfg: EntryPointsConfig) -> None:
    ensure_schema(con, "analytics.entrypoints")
    ensure_schema(con, "analytics.entrypoint_tests")

    con.execute(
        "DELETE FROM analytics.entrypoints WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.entrypoint_tests WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    candidates = []
    if cfg.detect_fastapi or cfg.detect_flask:
        candidates.extend(detect_http_entrypoints(con, cfg))
    if cfg.detect_click or cfg.detect_typer:
        candidates.extend(detect_cli_entrypoints(con, cfg))
    if cfg.detect_cron:
        candidates.extend(detect_job_entrypoints(con, cfg))

    for cand in candidates:
        _insert_entrypoint(con, cfg, cand, now)

    n = con.execute(
        "SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()[0]
    log.info("entrypoints populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
```

`_insert_entrypoint` handles:

* Mapping `cand` → GOID (query `core.goids`).
* Computing `entrypoint_id`.
* Looking up module → subsystem via `analytics.subsystem_modules`.
* Aggregating tests/coverage via the SQL above.
* Inserting into `analytics.entrypoints` + `analytics.entrypoint_tests`.

---

## 3. Detecting & populating `external_dependencies.*`

### 3.1 Overall approach

We’ll implement `analytics/dependencies.py` that:

1. Uses **imports + callgraph + CST** to find library usage:

   * `import boto3`, `from redis import Redis`, `import stripe`…
   * Functions that call methods on imported symbols (e.g. `boto3.client("s3")`, `redis.Redis().set(...)`).

2. Classifies each call into **modes** (`read`, `write`, `admin`) using a configurable pattern file (e.g. `config/dependency_patterns.yml`).

3. Aggregates to:

   * Per‑function rows in `analytics.external_dependency_calls`.
   * Per‑dependency summary in `analytics.external_dependencies`.

4. Relates dependencies to **config keys** via `analytics.config_values`. 

5. Optionally uses your **call graph** and NetworkX overlays to propagate usage to higher‑level architecture (e.g., subsystem risk) but the core tables are static SQL + AST.

### 3.2 Config model

In `config/models.py`:

```python
@dataclass(frozen=True)
class ExternalDependenciesConfig:
    repo: str
    commit: str
    dependency_patterns_path: Path | None = None   # YAML with library/mode rules

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, dependency_patterns_path: Path | None = None) -> "ExternalDependenciesConfig":
        return cls(repo=repo, commit=commit, dependency_patterns_path=dependency_patterns_path)
```

### 3.3 Pattern file for classification

Add `config/dependency_patterns.yml` in your repo (consumed by the analytics module), e.g.:

```yaml
libs:
  boto3:
    service_name: "aws"
    category: "cloud"
    patterns:
      - match: "client('s3')"          # substring in call source or AST path
        mode: ["read", "write"]
      - match: "resource('dynamodb')"
        mode: ["read", "write"]
  redis:
    service_name: "redis"
    category: "cache"
    patterns:
      - method: "get"
        mode: ["read"]
      - method: "set"
        mode: ["write"]
      - method: "delete"
        mode: ["write"]
  stripe:
    service_name: "stripe"
    category: "payment"
    patterns:
      - method_prefix: "Charge."
        mode: ["write"]
```

This keeps the classification logic data‑driven.

### 3.4 Discovering dependency usage sites

We can implement a two‑phase analysis:

1. **Import scanning** – map imported names → library.
2. **Call scanning** – for each function GOID, find calls that target imported names.

We can reuse your existing **CST/AST utilities** and GOID mapping approach (similar to how the call graph builder resolves callsites).

Rough algorithm:

#### 3.4.1 Build module‑level import map

Using LibCST or AST (`ingestion/cst_extract.py` or `ingestion/py_ast_extract.py`):

* For each `rel_path`:

  * For each `import boto3`:

    * Map `alias or name` → `boto3`.

  * For each `from redis import Redis as R`:

    * Map `R` → `redis` (base library).

Store in an in‑memory dict keyed by `(rel_path, local_name)`.

Optionally persist this to DuckDB as `analytics.import_aliases` table if you want reuse in other analytics.

#### 3.4.2 Scan function bodies for calls

For each function GOID:

* Use `core.goids` / `goid_crosswalk` to get `(rel_path, start_line, end_line)`; then use CST/AST to find call expressions in that span.

* For each call expression `target(...`:

  * Resolve the **base symbol** name: e.g., `boto3.client`, `redis.Redis`, `stripe.Charge.create`, `r.set`.

  * Use the import map to resolve base to a **library** name.

  * If library is one of interest (or just any third‑party lib):

    * Record a callsite:

      ```python
      DependencyCall(
          library="boto3",
          rel_path=rel_path,
          qualname=function_qualname,
          lineno=call_lineno,
          target_repr="boto3.client('s3')",
          method_name="client",
          attr_chain=["client"],
          args=...,
      )
      ```

  * Aggregate per function GOID:

    * `callsite_count` = number of such calls.
    * `modes` computed using patterns.

#### 3.4.3 Mode classification

Given a `DependencyCall` and loaded patterns for `library`:

* For each pattern in `patterns`:

  * If `method` defined and `call.method_name == pattern.method`: include `pattern.mode`.
  * If `method_prefix` defined and `target_repr` starts with it: include `pattern.mode`.
  * If `match` string present and `match in target_repr`: include `pattern.mode`.

If no pattern matches:

* Default `modes = ["unknown"]`.

At the function‑level, union modes across callsites, e.g. `["read", "write"]`.

### 3.5 Populating `external_dependency_calls`

Implementation in `analytics/dependencies.py`:

```python
def build_external_dependency_calls(con: duckdb.DuckDBPyConnection, cfg: ExternalDependenciesConfig) -> None:
    ensure_schema(con, "analytics.external_dependency_calls")

    con.execute(
        "DELETE FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    # 1) load module -> path -> import aliases
    alias_map = _build_import_alias_map(con, cfg)

    # 2) iterate functions (from analytics.function_metrics or core.goids)
    funcs = con.execute(
        """
        SELECT function_goid_h128, urn, rel_path, qualname
        FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()

    rows = []
    for goid, urn, rel_path, qualname in funcs:
        calls = find_dependency_calls_for_function(rel_path, qualname, alias_map, cfg)
        grouped = _group_calls_by_library(calls)

        for library, call_group in grouped.items():
            dep_id = _dep_id(cfg.repo, cfg.commit, library)
            module = _module_for_path(con, rel_path)
            modes = sorted({m for c in call_group for m in c.modes})
            evidence = {"examples": [c.target_repr for c in call_group[:3]]}

            rows.append((
                cfg.repo, cfg.commit, dep_id, library, _service_name(library),
                goid, urn, rel_path, module, qualname,
                len(call_group), json.dumps(modes), json.dumps(evidence), now,
            ))

    con.executemany(
        """
        INSERT INTO analytics.external_dependency_calls (
          repo, commit, dep_id, library, service_name,
          function_goid_h128, function_urn, rel_path, module, qualname,
          callsite_count, modes, evidence_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
```

Helper `_module_for_path` just joins `core.modules` on `rel_path`. 

### 3.6 Populating `external_dependencies`

Aggregate calls to dependency‑level rows:

```python
def build_external_dependencies(con: duckdb.DuckDBPyConnection, cfg: ExternalDependenciesConfig) -> None:
    ensure_schema(con, "analytics.external_dependencies")
    con.execute(
        "DELETE FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    # 1) per-dep aggregation from external_dependency_calls
    dep_rows = con.execute(
        """
        WITH per_dep AS (
          SELECT
            dep_id,
            library,
            service_name,
            COUNT(DISTINCT function_goid_h128) AS function_count,
            SUM(callsite_count)               AS callsite_count,
            LIST(DISTINCT module)             AS modules_json,
            LIST_DISTINCT(m) AS all_modes
          FROM analytics.external_dependency_calls
          WHERE repo = ? AND commit = ?
          GROUP BY dep_id, library, service_name
        ),
        modes AS (
          SELECT
            dep_id,
            ARRAY_DISTINCT(
              FLATTEN(
                LIST_TRANSFORM(all_modes, m -> m)
              )
            ) AS usage_modes
          FROM per_dep
        ),
        cfg_keys AS (
          SELECT
            dep_id,
            LIST(DISTINCT cv.key) AS config_keys
          FROM analytics.external_dependency_calls edc
          JOIN analytics.config_values cv
            ON cv.reference_modules @> [edc.module]
          WHERE edc.repo = ? AND edc.commit = ?
          GROUP BY dep_id
        )
        SELECT
          p.dep_id,
          p.library,
          p.service_name,
          p.function_count,
          p.callsite_count,
          p.modules_json,
          m.usage_modes,
          ck.config_keys
        FROM per_dep p
        LEFT JOIN modes m USING (dep_id)
        LEFT JOIN cfg_keys ck USING (dep_id)
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit],
    ).fetchall()

    rows = []
    for dep_id, library, service_name, fn_count, callsite_count, modules_json, usage_modes, config_keys in dep_rows:
        # simple risk heuristic: many write/admin modes => higher risk
        risk_level = _compute_dep_risk_level(usage_modes, fn_count, callsite_count)
        rows.append((
            cfg.repo, cfg.commit, dep_id, library, service_name,
            _category_for_library(library),
            fn_count, callsite_count,
            modules_json,
            usage_modes,
            config_keys,
            risk_level,
            now,
        ))

    con.executemany(
        """
        INSERT INTO analytics.external_dependencies (
          repo, commit, dep_id, library, service_name, category,
          function_count, callsite_count, modules_json, usage_modes, config_keys,
          risk_level, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
```

Here we:

* Join to `analytics.config_values` (per‑key view of config usage) to get `config_keys` for modules that use the dependency. 

* Derive `risk_level` heuristically (e.g., `high` if `admin` mode used or many write calls, `medium` if mostly read).

---

## 4. Orchestration, exports, docs views

### 4.1 Orchestration steps

In `orchestration/steps.py`, add:

```python
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.dependencies import (
    build_external_dependency_calls,
    build_external_dependencies,
)
from codeintel.config.models import EntryPointsConfig, ExternalDependenciesConfig
```

#### 4.1.1 EntryPointsStep

```python
@dataclass
class EntryPointsStep:
    """Detect HTTP/CLI/job entrypoints and map them to GOIDs + tests."""

    name: str = "entrypoints"
    deps: Sequence[str] = (
        "goids",          # needs core.goids/crosswalk
        "cst",            # or ast, for decorators/imports
        "import_graph",   # optional, for module names
        "subsystems",     # to attach subsystem_id
        "coverage_analytics",
        "test_coverage_edges",
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = EntryPointsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        build_entrypoints(con, cfg)
```

#### 4.1.2 ExternalDependenciesStep

```python
@dataclass
class ExternalDependenciesStep:
    """Identify external libraries/services and map functions to them."""

    name: str = "external_dependencies"
    deps: Sequence[str] = (
        "goids",
        "cst",
        "import_graph",
        "coverage_analytics",  # optional, if risk uses coverage
        "subsystems",          # optional, if we later attach subsystem risk
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        dep_cfg = ExternalDependenciesConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            dependency_patterns_path=ctx.paths.dependency_patterns_path,
        )
        build_external_dependency_calls(con, dep_cfg)
        build_external_dependencies(con, dep_cfg)
```

Wire them into your Prefect flow after analytics and subsystems steps, before docs export.

### 4.2 Exports

In `docs_export/export_jsonl.py` and `export_parquet.py`, add mapping entries:

```python
JSONL_DATASETS.update({
    "analytics.entrypoints": "entrypoints.jsonl",
    "analytics.entrypoint_tests": "entrypoint_tests.jsonl",
    "analytics.external_dependencies": "external_dependencies.jsonl",
    "analytics.external_dependency_calls": "external_dependency_calls.jsonl",
})

PARQUET_DATASETS.update({
    "analytics.entrypoints": "entrypoints.parquet",
    "analytics.entrypoint_tests": "entrypoint_tests.parquet",
    "analytics.external_dependencies": "external_dependencies.parquet",
    "analytics.external_dependency_calls": "external_dependency_calls.parquet",
})
```

So `generate_documents.sh` will emit these alongside your existing JSONL/Parquet datasets. 

### 4.3 Docs views (LLM‑friendly)

In `storage/views.py`, create concise views:

```sql
CREATE OR REPLACE VIEW docs.v_entrypoints AS
SELECT
  e.repo,
  e.commit,
  e.entrypoint_id,
  e.kind,
  e.handler_goid_h128,
  e.handler_urn,
  e.handler_rel_path,
  e.handler_module,
  e.handler_qualname,
  e.http_method,
  e.route_path,
  e.status_codes,
  e.auth_required,
  e.command_name,
  e.arguments_schema,
  e.schedule,
  e.trigger,
  e.subsystem_id,
  e.subsystem_name,
  e.tags,
  e.owners,
  e.tests_touching,
  e.failing_tests,
  e.slow_tests,
  e.flaky_tests,
  e.entrypoint_coverage_ratio,
  e.last_test_status,
  e.created_at
FROM analytics.entrypoints e;
```

And for dependencies:

```sql
CREATE OR REPLACE VIEW docs.v_external_dependencies AS
SELECT
  d.repo,
  d.commit,
  d.dep_id,
  d.library,
  d.service_name,
  d.category,
  d.function_count,
  d.callsite_count,
  d.modules_json,
  d.usage_modes,
  d.config_keys,
  d.risk_level,
  d.created_at
FROM analytics.external_dependencies d;
```

These become the main API for your FastAPI server and MCP tools when answering queries like “list API entrypoints in subsystem X” or “where do we call Redis and in what modes?”

You can also add:

* `docs.v_external_dependency_calls` – just `SELECT * FROM analytics.external_dependency_calls` for detailed usage.

### 4.4 Server / MCP hooks

In `server/query_templates.py` (or equivalent), add:

```python
SQL_GET_ENTRYPOINTS_FOR_SUBSYSTEM = """
SELECT *
FROM docs.v_entrypoints
WHERE repo = :repo
  AND commit = :commit
  AND subsystem_id = :subsystem_id
"""

SQL_GET_EXTERNAL_DEPENDENCIES = """
SELECT *
FROM docs.v_external_dependencies
WHERE repo = :repo
  AND commit = :commit
ORDER BY risk_level DESC, callsite_count DESC
"""

SQL_GET_DEP_CALLS_FOR_DEP = """
SELECT *
FROM analytics.external_dependency_calls
WHERE repo = :repo
  AND commit = :commit
  AND dep_id = :dep_id
"""
```

MCP tools can wrap these into:

* `list_entrypoints(subsystem_id?)`
* `get_entrypoint(entrypoint_id)`
* `list_external_dependencies(category?, risk_level?)`
* `get_external_dependency(dep_id)`

---

## 5. Testing strategy

To keep this robust and consistent with the rest of CodeIntel, I’d add:

### 5.1 Unit tests (small synthetic repo)

Create a tiny fixture repo that includes:

* `api.py` with FastAPI routes and Flask routes.
* `cli.py` with click and typer commands.
* `jobs.py` with APScheduler or `schedule.every()` usage.
* `deps.py` with `boto3`, `redis`, `stripe` imports and calls.
* A few pytest tests that hit some of the handlers.

Run the pipeline end‑to‑end against that fixture and assert:

* `analytics.entrypoints` has one row per decorated handler with correct `kind`, `http_method`, `route_path`, `command_name`.

* `entrypoints.handler_goid_h128` matches the GOID for the underlying function (check via `core.goids`). 

* `tests_touching` for each entrypoint matches counts derived from `analytics.test_coverage_edges`.

* `analytics.external_dependency_calls` contains:

  * A row for functions that use boto3 with `library='boto3'` and expected `modes`.
  * A row for Redis functions with `modes` containing `"read"`/`"write"` as appropriate.

* `analytics.external_dependencies` aggregates correctly:

  * `function_count` and `callsite_count` are sums across calls.
  * `config_keys` reflect config_values referencing modules that use the dependency (easy to test with a config file containing API keys). 

### 5.2 Integration tests on a real repo snapshot

Use your own CodeIntel repo (or another non‑trivial service) as test input:

* Run full pipeline including new steps.

* Smoke test:

  * `SELECT COUNT(*) FROM analytics.entrypoints` > 0.
  * At least some entrypoints have `subsystem_id`.
  * External deps include things like `fastapi`, `requests`, `pytest`, etc.

* Verify no regressions in existing analytics and docs exports.

---

## 6. How this fits your existing architecture

This design:

* Reuses your **GOID registry**, **call graph**, **coverage/test graphs**, **config_values**, **subsystems**, and **networkx layer** as the substrate.
* Follows the same pattern as your current analytics:

  * schema in `config/schemas/tables.py`,
  * ingesters/analytics in `analytics/*.py`,
  * orchestration via `orchestration/steps.py` + Prefect flow,
  * export via `docs_export/*`,
  * consumption via `docs.*` views and the server/MCP APIs.

Once this is in, an AI agent can:

* Ask “show me all HTTP/CLI entrypoints in subsystem X with low coverage and high risk” by joining `docs.v_entrypoints` with existing function/module/subsystem profile views.
* Ask “show me all functions that talk to Redis, grouped by subsystem and mode” via `docs.v_external_dependencies` + `external_dependency_calls`.

If you’d like, the next step can be to **tighten the initial detector set** (e.g., focus first on FastAPI + click + boto3/redis/stripe) and then iterate on the pattern file as you see real repos.
