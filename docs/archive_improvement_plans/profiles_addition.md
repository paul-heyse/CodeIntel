Here’s a concrete, code-level plan for adding the three entity‑centric profiles into the existing CodeIntel pipeline, wired all the way through DuckDB schemas → analytics → orchestration → exports → docs/MCP.

I’ll keep everything aligned with how things already work in `analytics`, `config/schemas`, `orchestration/steps`, and `docs_export`.

---

## 0. Where this fits in the current pipeline

Right now you already have:

* Per‑function metrics & types: `analytics.function_metrics`, `analytics.function_types`
* Coverage: `analytics.coverage_functions`, `analytics.test_coverage_edges`
* Test catalog: `analytics.test_catalog`
* File-level analytics: `core.ast_metrics`, `analytics.hotspots`, `analytics.typedness`, `analytics.static_diagnostics`
* Module registry + tags/owners: `core.modules` (with `tags`, `owners`)
* Risk summary per function: `analytics.goid_risk_factors` (loc, complexity, typedness, hotspots, static errors, coverage, tests, risk_score, risk_level, tags, owners)
* Graphs: call graph, import graph, CFG/DFG, symbol uses.
* Docs view: `docs.v_function_summary` = join of `goid_risk_factors` + metrics + docstrings + tags/owners.

We’ll add **three new analytics tables** plus optional docs views:

* `analytics.function_profile`
* `analytics.file_profile`
* `analytics.module_profile`
* `docs.v_function_profile`, `docs.v_file_profile`, `docs.v_module_profile` as convenience views over those tables.

We’ll also add a new orchestration step `ProfilesStep` that runs after `RiskFactorsStep` and before `ExportDocsStep`.

---

## 1. Schema design (DuckDB / config/schemas/tables.py)

### 1.1 `analytics.function_profile`

**Goal:** One row per function GOID, *denormalized*, built atop `analytics.goid_risk_factors` with extra detail: docstrings, types, tests, call graph degrees, and module metadata.

Add a new `TableSchema` entry in `config/schemas/tables.py` alongside `analytics.goid_risk_factors`:

```python
# config/schemas/tables.py

"analytics.function_profile": TableSchema(
    schema="analytics",
    name="function_profile",
    columns=[
        # Identity & location
        Column("function_goid_h128", "DECIMAL(38,0)"),
        Column("urn", "VARCHAR"),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("rel_path", "VARCHAR"),
        Column("module", "VARCHAR"),           # derived from rel_path/modules
        Column("language", "VARCHAR"),
        Column("kind", "VARCHAR"),
        Column("qualname", "VARCHAR"),
        Column("start_line", "INTEGER"),
        Column("end_line", "INTEGER"),

        # Structure & complexity (function_metrics)
        Column("loc", "INTEGER"),
        Column("logical_loc", "INTEGER"),
        Column("cyclomatic_complexity", "INTEGER"),
        Column("complexity_bucket", "VARCHAR"),
        Column("param_count", "INTEGER"),
        Column("positional_params", "INTEGER"),
        Column("keyword_params", "INTEGER"),
        Column("vararg", "BOOLEAN"),
        Column("kwarg", "BOOLEAN"),
        Column("max_nesting_depth", "INTEGER"),
        Column("stmt_count", "INTEGER"),
        Column("decorator_count", "INTEGER"),
        Column("has_docstring", "BOOLEAN"),

        # Typedness (function_types)
        Column("total_params", "INTEGER"),
        Column("annotated_params", "INTEGER"),
        Column("return_type", "VARCHAR"),
        Column("param_types", "JSON"),
        Column("fully_typed", "BOOLEAN"),
        Column("partial_typed", "BOOLEAN"),
        Column("untyped", "BOOLEAN"),
        Column("typedness_bucket", "VARCHAR"),
        Column("typedness_source", "VARCHAR"),

        # Static diagnostics & file typedness (from risk_factors join)
        Column("file_typed_ratio", "DOUBLE"),
        Column("static_error_count", "INTEGER"),
        Column("has_static_errors", "BOOLEAN"),

        # Coverage (coverage_functions / risk_factors)
        Column("executable_lines", "INTEGER"),
        Column("covered_lines", "INTEGER"),
        Column("coverage_ratio", "DOUBLE"),
        Column("tested", "BOOLEAN"),
        Column("untested_reason", "VARCHAR"),

        # Test summary (test_coverage_edges + test_catalog)
        Column("tests_touching", "INTEGER"),
        Column("failing_tests", "INTEGER"),
        Column("slow_tests", "INTEGER"),
        Column("flaky_tests", "INTEGER"),
        Column("last_test_status", "VARCHAR"),
        Column("dominant_test_status", "VARCHAR"),   # optional mode over statuses
        Column("slow_test_threshold_ms", "DOUBLE"),  # config echo

        # Call graph metrics (graph.call_graph_edges/nodes)
        Column("call_fan_in", "INTEGER"),
        Column("call_fan_out", "INTEGER"),
        Column("call_edge_in_count", "INTEGER"),
        Column("call_edge_out_count", "INTEGER"),
        Column("call_is_leaf", "BOOLEAN"),
        Column("call_is_entrypoint", "BOOLEAN"),
        Column("call_is_public", "BOOLEAN"),  # from call_graph_nodes.is_public

        # Risk
        Column("risk_score", "DOUBLE"),
        Column("risk_level", "VARCHAR"),
        Column("risk_component_coverage", "DOUBLE"),
        Column("risk_component_complexity", "DOUBLE"),
        Column("risk_component_static", "DOUBLE"),
        Column("risk_component_hotspot", "DOUBLE"),

        # Ownership & tags (from modules/tags_index via risk_factors)
        Column("tags", "JSON"),
        Column("owners", "JSON"),

        # Docstrings summary (core.docstrings / v_function_summary-like)
        Column("doc_short", "VARCHAR"),
        Column("doc_long", "VARCHAR"),
        Column("doc_params", "JSON"),
        Column("doc_returns", "JSON"),

        Column("created_at", "TIMESTAMP"),
    ],
    indexes=(
        Index("idx_function_profile_goid", ("function_goid_h128",)),
        Index("idx_function_profile_repo_commit", ("repo", "commit")),
    ),
    description="Denormalized per-function profile with metrics, tests, graph, docs, and risk.",
)
```

Notes:

* Most of these fields are directly sourced from existing tables (`function_metrics`, `function_types`, `coverage_functions`, `goid_risk_factors`, `test_catalog`, `test_coverage_edges`, `call_graph_*`, `docstrings`, `modules`).
* `risk_component_*` are derived from the same formula used in `RiskFactorsStep`, but exposed as separate columns.

### 1.2 `analytics.file_profile`

**Goal:** One row per file (`rel_path`) aggregating function‑level and file‑level signals.

New `TableSchema`:

```python
"analytics.file_profile": TableSchema(
    schema="analytics",
    name="file_profile",
    columns=[
        # Identity
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("rel_path", "VARCHAR"),
        Column("module", "VARCHAR"),            # join core.modules
        Column("language", "VARCHAR"),

        # AST metrics & hotspots
        Column("node_count", "INTEGER"),
        Column("function_count", "INTEGER"),
        Column("class_count", "INTEGER"),
        Column("avg_depth", "DOUBLE"),
        Column("max_depth", "INTEGER"),
        Column("ast_complexity", "DOUBLE"),
        Column("hotspot_score", "DOUBLE"),
        Column("commit_count", "INTEGER"),
        Column("author_count", "INTEGER"),
        Column("lines_added", "INTEGER"),
        Column("lines_deleted", "INTEGER"),

        # Typedness & static diagnostics
        Column("annotation_ratio", "DOUBLE"),
        Column("untyped_defs", "INTEGER"),
        Column("overlay_needed", "BOOLEAN"),
        Column("type_error_count", "INTEGER"),
        Column("static_error_count", "INTEGER"),
        Column("has_static_errors", "BOOLEAN"),

        # Aggregated function metrics (from function_profile / function_metrics)
        Column("total_functions", "INTEGER"),
        Column("public_functions", "INTEGER"),
        Column("avg_loc", "DOUBLE"),
        Column("max_loc", "INTEGER"),
        Column("avg_cyclomatic_complexity", "DOUBLE"),
        Column("max_cyclomatic_complexity", "INTEGER"),
        Column("high_risk_function_count", "INTEGER"),
        Column("medium_risk_function_count", "INTEGER"),
        Column("max_risk_score", "DOUBLE"),

        # Coverage & tests aggregated from function_profile / coverage_functions
        Column("file_coverage_ratio", "DOUBLE"),            # sum covered / sum exec
        Column("tested_function_count", "INTEGER"),
        Column("untested_function_count", "INTEGER"),
        Column("tests_touching", "INTEGER"),                # distinct tests

        # Ownership & tags (from core.modules / tags_index)
        Column("tags", "JSON"),
        Column("owners", "JSON"),

        Column("created_at", "TIMESTAMP"),
    ],
    indexes=(
        Index("idx_file_profile_repo_commit_relpath", ("repo", "commit", "rel_path")),
    ),
    description="Per-file aggregation of structure, risk, coverage, and ownership.",
)
```

### 1.3 `analytics.module_profile`

**Goal:** One row per module (logical package), aggregating file and function stats plus import graph metrics.

```python
"analytics.module_profile": TableSchema(
    schema="analytics",
    name="module_profile",
    columns=[
        # Identity
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
        Column("module", "VARCHAR"),
        Column("path", "VARCHAR"),            # from core.modules
        Column("language", "VARCHAR"),

        # Size & structure
        Column("file_count", "INTEGER"),
        Column("total_loc", "INTEGER"),
        Column("total_logical_loc", "INTEGER"),
        Column("function_count", "INTEGER"),
        Column("class_count", "INTEGER"),
        Column("avg_file_complexity", "DOUBLE"),
        Column("max_file_complexity", "DOUBLE"),

        # Risk aggregation from function_profile
        Column("high_risk_function_count", "INTEGER"),
        Column("medium_risk_function_count", "INTEGER"),
        Column("low_risk_function_count", "INTEGER"),
        Column("max_risk_score", "DOUBLE"),
        Column("avg_risk_score", "DOUBLE"),

        # Coverage aggregation
        Column("module_coverage_ratio", "DOUBLE"),
        Column("tested_function_count", "INTEGER"),
        Column("untested_function_count", "INTEGER"),

        # Import graph metrics
        Column("import_fan_in", "INTEGER"),       # dst_fan_in from import_graph_edges
        Column("import_fan_out", "INTEGER"),      # src_fan_out
        Column("cycle_group", "INTEGER"),         # from import_graph_edges
        Column("in_cycle", "BOOLEAN"),

        # Ownership & tags
        Column("tags", "JSON"),
        Column("owners", "JSON"),

        Column("created_at", "TIMESTAMP"),
    ],
    indexes=(
        Index("idx_module_profile_repo_commit_module", ("repo", "commit", "module")),
    ),
    description="Per-module summary of size, risk, coverage, imports, and ownership.",
)
```

---

## 2. Analytics implementation (`analytics/profiles.py`)

Create a new module `analytics/profiles.py` that’s similar in style to `coverage_analytics.py` and `functions.py`: it will:

* Call `ensure_schema` for each new table.
* Delete rows for the current `(repo, commit)` before recomputing.
* Insert via a big `INSERT INTO ... SELECT ...` SQL statement over existing tables.
* Log row counts.

We’ll also add a small config class in `config/models.py` (next section).

### 2.1 Config model

In `config/models.py`, add:

```python
# config/models.py

class ProfilesAnalyticsConfig(BaseModel):
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, repo: str, commit: str) -> "ProfilesAnalyticsConfig":
        return cls(repo=repo, commit=commit)
```

(You can later extend with thresholds, e.g. `slow_test_threshold_ms`.)

### 2.2 Building function_profile

In `analytics/profiles.py`:

```python
# analytics/profiles.py

from __future__ import annotations

import logging
from datetime import UTC, datetime

import duckdb

from codeintel.config.models import ProfilesAnalyticsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)


def build_function_profile(con: StorageGateway, cfg: ProfilesAnalyticsConfig) -> None:
    """Populate analytics.function_profile for a given repo/commit."""
    ensure_schema(con, "analytics.function_profile")

    # Clear existing rows for this repo/commit
    con.execute(
        "DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    # Materialize call graph degrees in CTEs for readability
    con.execute(
        """
        INSERT INTO analytics.function_profile (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            module,
            language,
            kind,
            qualname,
            start_line,
            end_line,
            loc,
            logical_loc,
            cyclomatic_complexity,
            complexity_bucket,
            param_count,
            positional_params,
            keyword_params,
            vararg,
            kwarg,
            max_nesting_depth,
            stmt_count,
            decorator_count,
            has_docstring,
            total_params,
            annotated_params,
            return_type,
            param_types,
            fully_typed,
            partial_typed,
            untyped,
            typedness_bucket,
            typedness_source,
            file_typed_ratio,
            static_error_count,
            has_static_errors,
            executable_lines,
            covered_lines,
            coverage_ratio,
            tested,
            untested_reason,
            tests_touching,
            failing_tests,
            slow_tests,
            flaky_tests,
            last_test_status,
            dominant_test_status,
            slow_test_threshold_ms,
            call_fan_in,
            call_fan_out,
            call_edge_in_count,
            call_edge_out_count,
            call_is_leaf,
            call_is_entrypoint,
            call_is_public,
            risk_score,
            risk_level,
            risk_component_coverage,
            risk_component_complexity,
            risk_component_static,
            risk_component_hotspot,
            tags,
            owners,
            doc_short,
            doc_long,
            doc_params,
            doc_returns,
            created_at
        )
        WITH rf AS (
            SELECT *
            FROM analytics.goid_risk_factors
            WHERE repo = ? AND commit = ?
        ),
        fm AS (
            SELECT *
            FROM analytics.function_metrics
        ),
        ft AS (
            SELECT *
            FROM analytics.function_types
        ),
        doc AS (
            SELECT
                repo,
                commit,
                rel_path,
                qualname,
                kind,
                short_desc    AS doc_short,
                long_desc     AS doc_long,
                params        AS doc_params,
                returns       AS doc_returns
            FROM core.docstrings
        ),
        t_stats AS (
            -- aggregate test_coverage_edges + test_catalog per function
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id)                                     AS tests_touching,
                COUNT(DISTINCT CASE WHEN tc.status IN ('failed','error') THEN e.test_id END)
                    AS failing_tests,
                COUNT(DISTINCT CASE WHEN tc.duration_ms > 1000 THEN e.test_id END)
                    AS slow_tests,  -- threshold may move into config
                COUNT(DISTINCT CASE WHEN tc.flaky THEN e.test_id END)
                    AS flaky_tests,
                -- e.last_status is per test→function; choose "worst" or mode
                MAX(e.last_status)                                            AS last_test_status,
                MODE() WITHIN GROUP (ORDER BY tc.status)                      AS dominant_test_status
            FROM analytics.test_coverage_edges AS e
            LEFT JOIN analytics.test_catalog AS tc
                ON e.test_id = tc.test_id
            GROUP BY e.function_goid_h128
        ),
        cg_out AS (
            SELECT
                caller_goid_h128 AS function_goid_h128,
                COUNT(*)                         AS call_edge_out_count,
                COUNT(DISTINCT callee_goid_h128) AS call_fan_out
            FROM graph.call_graph_edges
            GROUP BY caller_goid_h128
        ),
        cg_in AS (
            SELECT
                callee_goid_h128 AS function_goid_h128,
                COUNT(*)                         AS call_edge_in_count,
                COUNT(DISTINCT caller_goid_h128) AS call_fan_in
            FROM graph.call_graph_edges
            WHERE callee_goid_h128 IS NOT NULL
            GROUP BY callee_goid_h128
        ),
        cg_nodes AS (
            SELECT
                goid_h128 AS function_goid_h128,
                is_public
            FROM graph.call_graph_nodes
        ),
        cg_degrees AS (
            SELECT
                COALESCE(co.function_goid_h128, ci.function_goid_h128, cn.function_goid_h128) AS function_goid_h128,
                COALESCE(ci.call_fan_in, 0)       AS call_fan_in,
                COALESCE(co.call_fan_out, 0)      AS call_fan_out,
                COALESCE(ci.call_edge_in_count,0) AS call_edge_in_count,
                COALESCE(co.call_edge_out_count,0)AS call_edge_out_count,
                CASE WHEN co.call_fan_out IS NULL OR co.call_fan_out = 0 THEN TRUE ELSE FALSE END AS call_is_leaf,
                -- Entrypoint = never called but calls others (heuristic)
                CASE WHEN ci.call_fan_in = 0 AND co.call_fan_out > 0 THEN TRUE ELSE FALSE END AS call_is_entrypoint,
                cn.is_public                      AS call_is_public
            FROM cg_out AS co
            FULL OUTER JOIN cg_in AS ci USING (function_goid_h128)
            FULL OUTER JOIN cg_nodes AS cn USING (function_goid_h128)
        )
        SELECT
            rf.function_goid_h128,
            rf.urn,
            rf.repo,
            rf.commit,
            rf.rel_path,
            m.module,
            rf.language,
            rf.kind,
            rf.qualname,
            fm.start_line,
            fm.end_line,
            rf.loc,
            rf.logical_loc,
            rf.cyclomatic_complexity,
            rf.complexity_bucket,
            fm.param_count,
            fm.positional_params,
            fm.keyword_params,
            fm.vararg,
            fm.kwarg,
            fm.max_nesting_depth,
            fm.stmt_count,
            fm.decorator_count,
            fm.has_docstring,
            ft.total_params,
            ft.annotated_params,
            ft.return_type,
            ft.param_types,
            ft.fully_typed,
            ft.partial_typed,
            ft.untyped,
            rf.typedness_bucket,
            rf.typedness_source,
            rf.file_typed_ratio,
            rf.static_error_count,
            rf.has_static_errors,
            rf.executable_lines,
            rf.covered_lines,
            rf.coverage_ratio,
            rf.tested,
            cf.untested_reason,
            COALESCE(t_stats.tests_touching, 0),
            COALESCE(t_stats.failing_tests, 0),
            COALESCE(t_stats.slow_tests, 0),
            COALESCE(t_stats.flaky_tests, 0),
            rf.last_test_status,
            t_stats.dominant_test_status,
            1000.0, -- slow_test_threshold_ms (ms)
            COALESCE(cg.call_fan_in, 0),
            COALESCE(cg.call_fan_out, 0),
            COALESCE(cg.call_edge_in_count, 0),
            COALESCE(cg.call_edge_out_count, 0),
            cg.call_is_leaf,
            cg.call_is_entrypoint,
            cg.call_is_public,
            rf.risk_score,
            rf.risk_level,
            -- risk components derived from the same formula used in RiskFactorsStep
            COALESCE(1.0 - rf.coverage_ratio, 1.0) * 0.4 AS risk_component_coverage,
            CASE rf.complexity_bucket
                WHEN 'high' THEN 0.4
                WHEN 'medium' THEN 0.2
                ELSE 0.0
            END AS risk_component_complexity,
            CASE WHEN rf.has_static_errors THEN 0.2 ELSE 0.0 END AS risk_component_static,
            CASE WHEN rf.hotspot_score > 0 THEN 0.1 ELSE 0.0 END AS risk_component_hotspot,
            rf.tags,
            rf.owners,
            doc.doc_short,
            doc.doc_long,
            doc.doc_params,
            doc.doc_returns,
            ?
        FROM rf
        LEFT JOIN analytics.function_metrics AS fm
            ON rf.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.function_types AS ft
            ON rf.function_goid_h128 = ft.function_goid_h128
        LEFT JOIN analytics.coverage_functions AS cf
            ON rf.function_goid_h128 = cf.function_goid_h128
        LEFT JOIN t_stats
            ON rf.function_goid_h128 = t_stats.function_goid_h128
        LEFT JOIN cg_degrees AS cg
            ON rf.function_goid_h128 = cg.function_goid_h128
        LEFT JOIN core.modules AS m
            ON m.path = rf.rel_path
        LEFT JOIN doc
            ON doc.repo = rf.repo
           AND doc.commit = rf.commit
           AND doc.rel_path = rf.rel_path
           AND doc.qualname = rf.qualname
           AND doc.kind = rf.kind
        """,
        [cfg.repo, cfg.commit, now],
    )

    n = con.execute(
        "SELECT COUNT(*) FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    log.info("function_profile populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
```

Notes:

* This SQL closely mirrors the existing `RiskFactorsStep` join but adds CTEs for call graph and tests (you already have the underlying tables, so this is just wiring).
* You can factor out the risk-component weights into constants or config to avoid duplication with `RiskFactorsStep`.

### 2.3 Building file_profile

Add:

```python
def build_file_profile(con: StorageGateway, cfg: ProfilesAnalyticsConfig) -> None:
    ensure_schema(con, "analytics.file_profile")
    con.execute(
        "DELETE FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.file_profile (
            repo,
            commit,
            rel_path,
            module,
            language,
            node_count,
            function_count,
            class_count,
            avg_depth,
            max_depth,
            ast_complexity,
            hotspot_score,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            annotation_ratio,
            untyped_defs,
            overlay_needed,
            type_error_count,
            static_error_count,
            has_static_errors,
            total_functions,
            public_functions,
            avg_loc,
            max_loc,
            avg_cyclomatic_complexity,
            max_cyclomatic_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            max_risk_score,
            file_coverage_ratio,
            tested_function_count,
            untested_function_count,
            tests_touching,
            tags,
            owners,
            created_at
        )
        WITH fm AS (
            SELECT
                repo,
                commit,
                rel_path,
                COUNT(*) AS total_functions,
                COUNT(*) FILTER (WHERE call_is_public) AS public_functions,
                AVG(loc) AS avg_loc,
                MAX(loc) AS max_loc,
                AVG(cyclomatic_complexity) AS avg_cyclomatic_complexity,
                MAX(cyclomatic_complexity) AS max_cyclomatic_complexity,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_function_count,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_risk_function_count,
                MAX(risk_score) AS max_risk_score,
                SUM(covered_lines) AS sum_covered_lines,
                SUM(executable_lines) AS sum_exec_lines,
                SUM(CASE WHEN tested THEN 1 ELSE 0 END) AS tested_function_count,
                SUM(CASE WHEN NOT tested THEN 1 ELSE 0 END) AS untested_function_count,
                SUM(tests_touching) AS tests_touching
            FROM analytics.function_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, rel_path
        ),
        ast AS (
            SELECT * FROM core.ast_metrics
        ),
        hs AS (
            SELECT * FROM analytics.hotspots
        ),
        ty AS (
            SELECT * FROM analytics.typedness
        ),
        sd AS (
            SELECT
                path AS rel_path,
                type_error_count AS static_error_count,
                type_error_count > 0 AS has_static_errors
            FROM analytics.static_diagnostics
        ),
        mod AS (
            SELECT repo, commit, path, module, language, tags, owners
            FROM core.modules
        )
        SELECT
            fm.repo,
            fm.commit,
            fm.rel_path,
            mod.module,
            mod.language,
            ast.node_count,
            ast.function_count,
            ast.class_count,
            ast.avg_depth,
            ast.max_depth,
            ast.complexity AS ast_complexity,
            hs.score       AS hotspot_score,
            hs.commit_count,
            hs.author_count,
            hs.lines_added,
            hs.lines_deleted,
            ty.annotation_ratio,
            ty.untyped_defs,
            ty.overlay_needed,
            ty.type_error_count,
            sd.static_error_count,
            sd.has_static_errors,
            fm.total_functions,
            fm.public_functions,
            fm.avg_loc,
            fm.max_loc,
            fm.avg_cyclomatic_complexity,
            fm.max_cyclomatic_complexity,
            fm.high_risk_function_count,
            fm.medium_risk_function_count,
            fm.max_risk_score,
            CASE
                WHEN fm.sum_exec_lines > 0 THEN fm.sum_covered_lines * 1.0 / fm.sum_exec_lines
                ELSE NULL
            END AS file_coverage_ratio,
            fm.tested_function_count,
            fm.untested_function_count,
            fm.tests_touching,
            mod.tags,
            mod.owners,
            ?
        FROM fm
        LEFT JOIN ast
            ON fm.rel_path = ast.rel_path
        LEFT JOIN hs
            ON fm.rel_path = hs.rel_path
        LEFT JOIN ty
            ON fm.rel_path = ty.path
        LEFT JOIN sd
            ON fm.rel_path = sd.rel_path
        LEFT JOIN mod
            ON fm.repo = mod.repo
           AND fm.commit = mod.commit
           AND fm.rel_path = mod.path
        """,
        [cfg.repo, cfg.commit, now],
    )

    n = con.execute(
        "SELECT COUNT(*) FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    log.info("file_profile populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
```

This uses `function_profile` as the main aggregation source so you get all risk/coverage/test semantics “for free”.

### 2.4 Building module_profile

```python
def build_module_profile(con: StorageGateway, cfg: ProfilesAnalyticsConfig) -> None:
    ensure_schema(con, "analytics.module_profile")
    con.execute(
        "DELETE FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.module_profile (
            repo,
            commit,
            module,
            path,
            language,
            file_count,
            total_loc,
            total_logical_loc,
            function_count,
            class_count,
            avg_file_complexity,
            max_file_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            low_risk_function_count,
            max_risk_score,
            avg_risk_score,
            module_coverage_ratio,
            tested_function_count,
            untested_function_count,
            import_fan_in,
            import_fan_out,
            cycle_group,
            in_cycle,
            tags,
            owners,
            created_at
        )
        WITH fp AS (
            SELECT
                repo,
                commit,
                module,
                SUM(total_functions) AS function_count,
                SUM(public_functions) AS public_function_count,
                SUM(high_risk_function_count) AS high_risk_function_count,
                SUM(medium_risk_function_count) AS medium_risk_function_count,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_function_count,
                MAX(max_risk_score) AS max_risk_score,
                AVG(max_risk_score) AS avg_risk_score,
                SUM(total_functions) AS total_functions,
                SUM(tested_function_count) AS tested_function_count,
                SUM(untested_function_count) AS untested_function_count,
                SUM(file_coverage_ratio * total_functions) / NULLIF(SUM(total_functions), 0)
                    AS module_coverage_ratio
            FROM analytics.file_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        files AS (
            SELECT
                repo,
                commit,
                module,
                COUNT(*) AS file_count,
                SUM(avg_loc * total_functions) AS total_loc,
                SUM(avg_cyclomatic_complexity * total_functions) AS total_complexity,
                MAX(ast_complexity) AS max_file_complexity,
                SUM(total_functions) AS total_functions
            FROM analytics.file_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        mod AS (
            SELECT repo, commit, module, path, language, tags, owners
            FROM core.modules
        ),
        imports AS (
            SELECT
                src_module AS module,
                MAX(src_fan_out) AS import_fan_out,
                MAX(dst_fan_in) FILTER (WHERE dst_module = src_module) AS import_fan_in,
                MAX(cycle_group) AS cycle_group,
                MAX(CASE WHEN cycle_group IS NOT NULL THEN 1 ELSE 0 END) AS in_cycle_flag
            FROM graph.import_graph_edges
            GROUP BY src_module
        )
        SELECT
            fp.repo,
            fp.commit,
            fp.module,
            mod.path,
            mod.language,
            files.file_count,
            files.total_loc,
            NULL,  -- total_logical_loc: optional, or compute from function_metrics
            fp.function_count,
            NULL,  -- class_count: derive from ast_metrics if desired
            CASE
                WHEN files.file_count > 0 THEN files.total_complexity / files.file_count
                ELSE NULL
            END AS avg_file_complexity,
            files.max_file_complexity,
            fp.high_risk_function_count,
            fp.medium_risk_function_count,
            fp.low_risk_function_count,
            fp.max_risk_score,
            fp.avg_risk_score,
            fp.module_coverage_ratio,
            fp.tested_function_count,
            fp.untested_function_count,
            COALESCE(imports.import_fan_in, 0),
            COALESCE(imports.import_fan_out, 0),
            imports.cycle_group,
            imports.in_cycle_flag > 0 AS in_cycle,
            mod.tags,
            mod.owners,
            ?
        FROM fp
        LEFT JOIN files
            ON fp.repo = files.repo
           AND fp.commit = files.commit
           AND fp.module = files.module
        LEFT JOIN mod
            ON fp.repo = mod.repo
           AND fp.commit = mod.commit
           AND fp.module = mod.module
        LEFT JOIN imports
            ON fp.module = imports.module
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit, now],
    )

    n = con.execute(
        "SELECT COUNT(*) FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    log.info("module_profile populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
```

(You can refine the logical_loc and class_count wiring in a second pass using `core.ast_metrics` and `analytics.function_metrics`.)

---

## 3. Orchestration integration (`orchestration/steps.py`)

You already have `RiskFactorsStep` that builds `analytics.goid_risk_factors`.

Add a new step class:

```python
# orchestration/steps.py

from codeintel.analytics.profiles import (
    build_function_profile,
    build_file_profile,
    build_module_profile,
)
from codeintel.config.models import ProfilesAnalyticsConfig


@dataclass
class ProfilesStep:
    """Build function/file/module profiles on top of risk factors and graphs."""

    name: str = "profiles"
    deps: Sequence[str] = (
        "risk_factors",   # needs analytics.goid_risk_factors
        "callgraph",      # for call graph metrics
        "import_graph",   # for module import metrics
    )

    def run(self, ctx: PipelineContext, con: StorageGateway) -> None:
        _log_step(self.name)
        cfg = ProfilesAnalyticsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
        )
        build_function_profile(con, cfg)
        build_file_profile(con, cfg)
        build_module_profile(con, cfg)
```

Then include `ProfilesStep` in the pipeline step graph before `ExportDocsStep`. For example, near where `RiskFactorsStep` and `ExportDocsStep` are registered:

```python
PIPELINE_STEPS: dict[str, type[PipelineStep]] = {
    # ... existing steps ...
    "risk_factors": RiskFactorsStep,
    "profiles": ProfilesStep,
    "export_docs": ExportDocsStep,
}
```

This keeps the dependency chain:

`function_metrics` → `coverage_functions` → `test_coverage_edges` → `goid_risk_factors` → `profiles` → `docs_export`.

---

## 4. Docs views (`storage/views.py`)

You already have `docs.v_function_summary` and `docs.v_call_graph_enriched`.

Add simple pass‑through views so the server & MCP can treat profiles as first‑class docs surfaces:

```python
# storage/views.py

def create_all_views(con: StorageGateway) -> None:
    # existing v_function_summary, v_call_graph_enriched...

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_profile AS
        SELECT *
        FROM analytics.function_profile
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_file_profile AS
        SELECT *
        FROM analytics.file_profile
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_profile AS
        SELECT *
        FROM analytics.module_profile
        """
    )
```

Later you can make these views more curated (e.g., hide low‑level flags, add derived labels), but a `SELECT *` is a good first pass.

---

## 5. Export wiring (`docs_export/export_jsonl.py` & `export_parquet.py`)

Your JSONL/Parquet exporters currently know about `core.*`, `analytics.*`, and `graph.*` tables and map them to filenames like `ast_nodes.jsonl`, `goid_risk_factors.jsonl`, etc.

Update the mapping so the new tables are included:

```python
# docs_export/export_jsonl.py

TABLES_TO_EXPORT: dict[str, str] = {
    # existing entries ...
    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
    "analytics.function_profile": "function_profile.jsonl",
    "analytics.file_profile": "file_profile.jsonl",
    "analytics.module_profile": "module_profile.jsonl",
}
```

And mirror the same in `export_parquet.py`:

```python
TABLES_TO_EXPORT_PARQUET: dict[str, str] = {
    # ...
    "analytics.function_profile": "function_profile.parquet",
    "analytics.file_profile": "file_profile.parquet",
    "analytics.module_profile": "module_profile.parquet",
}
```

No changes to CLI are required; `pipeline run` already ends in `ExportDocsStep`, which will now pick up the additional tables.

---

## 6. Server & MCP: exposing profiles

Your FastAPI server and MCP tools query `docs.*` views via SQL templates.

### 6.1 Server query templates

In `server/query_templates.py`, add a couple of templates:

```python
SQL_GET_FUNCTION_PROFILE = """
SELECT *
FROM docs.v_function_profile
WHERE repo = :repo
  AND commit = :commit
  AND function_goid_h128 = :function_goid_h128
"""

SQL_GET_FILE_PROFILE = """
SELECT *
FROM docs.v_file_profile
WHERE repo = :repo
  AND commit = :commit
  AND rel_path = :rel_path
"""

SQL_GET_MODULE_PROFILE = """
SELECT *
FROM docs.v_module_profile
WHERE repo = :repo
  AND commit = :commit
  AND module = :module
"""
```

Wire these into new FastAPI endpoints (e.g., `/profiles/function`, `/profiles/file`, `/profiles/module`).

### 6.2 MCP tools

In `mcp/tools.py`, add tools along the lines of:

* `get_function_profile(function_goid_h128: str, repo: str, commit: str)`
* `get_file_profile(rel_path: str, repo: str, commit: str)`
* `get_module_profile(module: str, repo: str, commit: str)`

Each tool just runs the corresponding SQL template against the backend and returns a JSON object that matches your `function_profile.*`, etc.

This gives an AI agent a *single hop* way to answer “tell me everything about this function/file/module” with one query rather than manual joins.

---

## 7. README_METADATA updates

Add new sections to `README_METADATA.md` alongside `goid_risk_factors.*` so downstream consumers know these exist.

### 7.1 Function Profile (`function_profile.*`)

Document:

* Purpose: denormalized per-function view combining `goid_risk_factors`, `function_metrics`, `function_types`, `coverage_functions`, tests, call graph, docstrings, tags, and owners.
* Origin: `ProfilesStep` in `orchestration/steps.py` using `analytics/profiles.build_function_profile`.
* Key columns (high‑level groups; you don’t need to list every field, but at least identity, coverage, tests, graph metrics, risk).

### 7.2 File Profile (`file_profile.*`)

* Purpose: per‑file aggregated metrics (AST metrics, hotspots, typedness, static diagnostics) plus aggregated function risk and coverage.
* Origin: `ProfilesStep` (`build_file_profile`).
* Columns as in the schema above.

### 7.3 Module Profile (`module_profile.*`)

* Purpose: per‑module architecture summary (size, risk, coverage, import graph topology, ownership).
* Origin: `ProfilesStep` (`build_module_profile`).
* Columns as in the schema above.

---

## 8. Testing & validation

To make this robust and friendly for an AI agent, I’d include:

1. **Unit tests** for `analytics/profiles.py`:

   * Use a tiny synthetic DuckDB DB (a few functions, tests, and call graph edges).
   * Insert rows into `function_metrics`, `function_types`, `coverage_functions`, `goid_risk_factors`, `test_catalog`, `test_coverage_edges`, `graph.call_graph_*`, `core.docstrings`, `core.modules`, `analytics.hotspots`, `analytics.typedness`, `analytics.static_diagnostics`.
   * Run `build_function_profile`, `build_file_profile`, `build_module_profile`.
   * Assert that:

     * Function rows have correct degrees (`call_fan_in`, `call_fan_out`) for simple call graphs.
     * Coverage and test aggregates match hand‑computed expectations.
     * File and module aggregates (function counts, coverage) roll up correctly.

2. **Integration tests** wired through `PipelineContext`:

   * Use the existing small sample repo(s) you’re already running the pipeline on.
   * Call `RepoScanStep` → … → `RiskFactorsStep` → `ProfilesStep`.
   * Assert that:

     * `analytics.function_profile` row count == `analytics.goid_risk_factors` row count for that repo/commit.
     * Every `function_profile.rel_path` is present in `file_profile`.
     * Every `file_profile.module` is present in `module_profile`.

3. **Docs export smoke test**:

   * After running `ExportDocsStep`, verify that `function_profile.jsonl`, `file_profile.jsonl`, and `module_profile.jsonl` exist under `Document Output/` and are non‑empty.

---

## 9. How this helps the AI agent

Once this is in place, an agent can:

* Hit **`docs.v_function_profile`** once to get *everything* about a function: signature, complexity, types, coverage & tests, call graph degrees, docstring text, risk breakdown, tags, owners.
* Hit **`docs.v_file_profile`** to choose the right file(s) to read or refactor by balancing hotspot score, risk, and test coverage.
* Hit **`docs.v_module_profile`** to understand architecture boundaries, import layering, ownership, and “hot spots” at the module level.

And because these are exported as JSONL, you can also feed them directly into LLM context windows without reassembling joins in‑prompt.

If you’d like, next step I can help trim or expand the column sets to match exactly how you want the agent to “think” about the codebase (e.g., more emphasis on tests vs. graph topology).
