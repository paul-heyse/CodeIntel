# Analytics refactor detailed implementation plan #

Here’s how I’d turn that “break up the big aggregators” idea into an actual, implementable refactor, with concrete module structure and code snippets keyed to your repo.

I’ll focus on:

1. Function profiles (deep, with snippets)
2. File & module profiles (same pattern, but lighter)
3. Test profiles (how to split that huge file into recipe modules)
4. Value objects / row models
5. Graph metrics alignment (FunctionGraphFeatures)
6. Testing and rollout

All paths & types are aligned with your current code:

* `analytics/profiles.py` (4 functions, big SQL)
* `analytics/tests/profiles.py`
* `analytics/graphs/graph_metrics.py`, `analytics/graph_rows/graph_metrics.py`
* `analytics/context.py` (`AnalyticsContext`)
* `storage/gateway.py`
* `config/steps_analytics.py` (`ProfilesAnalyticsStepConfig`)
* `pipeline/orchestration/steps_analytics.py` (`ProfilesStep`)

---

## 1. Introduce a `analytics/profiles/` subpackage

### 1.1 New layout

Create a new subpackage:

```text
analytics/
  profiles/
    __init__.py
    types.py
    functions.py
    files.py
    modules.py
    graph_features.py
```

And slim down `analytics/profiles.py` to become a *compatibility + orchestration shim* that delegates to the new modules (we’ll patch this at the end).

### 1.2 `analytics/profiles/types.py`

Define the core “recipe ingredients” as dataclasses + typed maps. These are *pure Python*, DuckDB-agnostic, and easy for tests/agents to reason about.

```python
# analytics/profiles/types.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from codeintel.storage.gateway import DuckDBConnection


@dataclass(frozen=True)
class FunctionProfileInputs:
    """
    Shared inputs used by the function_profile recipe.

    This is a handle object, not a giant in-memory materialization. The actual
    data still lives in DuckDB; helper functions run queries against `con`.
    """

    con: DuckDBConnection
    repo: str
    commit: str
    created_at: datetime


@dataclass(frozen=True)
class FunctionRiskView:
    """Summarized risk factors per function."""

    function_goid_h128: int
    risk_score: float
    risk_factors_json: str  # or dict[str, Any] if you prefer
    risk_bucket: str | None


@dataclass(frozen=True)
class CoverageSummary:
    """Aggregated coverage for a single function."""

    function_goid_h128: int
    lines_covered: int
    lines_total: int
    coverage_ratio: float | None
    covered_by_tests: int
    flaky_tests: int
    slow_tests: int


@dataclass(frozen=True)
class TestSummary:
    """Test-centric signals used in function_profile."""

    function_goid_h128: int
    test_count: int
    dominant_status: str | None
    behavior_tags_json: str | None  # aligned with tests.analytics tables


@dataclass(frozen=True)
class FunctionGraphFeatures:
    """
    Graph-driven metrics for a function.

    This is the narrowed subset of the CentralityBundle / NeighborStats that
    function_profile actually uses.
    """

    function_goid_h128: int
    call_fan_in: int
    call_fan_out: int
    pagerank: float | None
    betweenness: float | None
    in_degree_centrality: float | None
    out_degree_centrality: float | None
    component_id: int
    in_cycle: bool


@dataclass(frozen=True)
class FunctionDocView:
    """Docstring-derived information per function."""

    function_goid_h128: int
    short_desc: str | None
    long_desc: str | None
    params_json: str | None
    returns_json: str | None


@dataclass(frozen=True)
class FunctionHistoryView:
    """History metrics per function."""

    function_goid_h128: int
    created_at: datetime | None
    last_modified_at: datetime | None
    churn_score: float | None
    commit_count: int
    author_count: int
```

You’ll likely want more fields, but this captures the pattern: the *recipe* manipulates these simple value types, not raw `duckdb.DuckDBPyRelation`.

---

## 2. Refactor **function profiles** into a recipe

We’ll move almost all of the 460-line `build_function_profile` body into `analytics/profiles/functions.py`, as smaller helpers.

### 2.1 `compute_function_profile_inputs`

This is the new “entry” object for the recipe. It wraps the connection + repo/commit + timestamp.

```python
# analytics/profiles/functions.py
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.analytics.context import AnalyticsContext
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import function_profile_row_to_tuple, FunctionProfileRowModel

from .types import (
    CoverageSummary,
    FunctionDocView,
    FunctionGraphFeatures,
    FunctionHistoryView,
    FunctionProfileInputs,
    FunctionRiskView,
    TestSummary,
)


def compute_function_profile_inputs(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
) -> FunctionProfileInputs:
    """
    Create a FunctionProfileInputs handle for the current snapshot.

    This is intentionally tiny; all heavy lifting happens in downstream helpers.
    """
    con = gateway.con
    now = datetime.now(tz=UTC)

    return FunctionProfileInputs(
        con=con,
        repo=cfg.repo,
        commit=cfg.commit,
        created_at=now,
    )
```

### 2.2 Risk view: `join_function_risk`

Move the `rf AS (...)` CTE logic from `build_function_profile` into a focused helper that returns a mapping `goid -> FunctionRiskView`.

```python
def join_function_risk(inputs: FunctionProfileInputs) -> Mapping[int, FunctionRiskView]:
    """
    Summarize risk factors into a per-function view.

    This corresponds to the `rf` CTE in the original SQL.
    """
    con = inputs.con
    rows = con.execute(
        """
        SELECT
            function_goid_h128,
            risk_score,
            risk_factors_json,
            risk_bucket
        FROM analytics.goid_risk_factors
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, FunctionRiskView] = {}
    for (
        function_goid_h128,
        risk_score,
        risk_factors_json,
        risk_bucket,
    ) in rows:
        result[int(function_goid_h128)] = FunctionRiskView(
            function_goid_h128=int(function_goid_h128),
            risk_score=float(risk_score) if risk_score is not None else 0.0,
            risk_factors_json=str(risk_factors_json) if risk_factors_json is not None else "[]",
            risk_bucket=str(risk_bucket) if risk_bucket is not None else None,
        )
    return result
```

You can keep this exactly aligned with the original CTE’s column set; I’m only showing a minimal version.

### 2.3 Coverage: `join_function_coverage`

This pulls the logic from the `func_cov AS (...)` CTE in your current SQL.

```python
def join_function_coverage(inputs: FunctionProfileInputs) -> Mapping[int, CoverageSummary]:
    """
    Aggregate line + test coverage signals for each function.

    Derived from analytics.coverage_functions and analytics.test_coverage_edges.
    """
    con = inputs.con
    rows = con.execute(
        """
        WITH fc AS (
            SELECT
                function_goid_h128,
                SUM(lines_covered) AS lines_covered,
                SUM(lines_total)   AS lines_total
            FROM analytics.coverage_functions
            WHERE repo = ? AND commit = ?
            GROUP BY function_goid_h128
        ),
        tests AS (
            SELECT
                function_goid_h128,
                COUNT(DISTINCT test_id) AS test_count,
                COUNT(DISTINCT CASE WHEN tc.duration_ms > ? THEN test_id END) AS slow_tests,
                COUNT(DISTINCT CASE WHEN tc.flaky THEN test_id END) AS flaky_tests
            FROM analytics.test_coverage_edges AS e
            LEFT JOIN analytics.test_catalog AS tc
              ON e.test_id = tc.test_id
             AND e.repo = tc.repo
             AND e.commit = tc.commit
            WHERE e.repo = ? AND e.commit = ?
            GROUP BY function_goid_h128
        )
        SELECT
            fc.function_goid_h128,
            fc.lines_covered,
            fc.lines_total,
            tests.test_count,
            tests.flaky_tests,
            tests.slow_tests
        FROM fc
        LEFT JOIN tests
          ON tests.function_goid_h128 = fc.function_goid_h128
        """,
        [
            inputs.repo,
            inputs.commit,
            # slow threshold; you already have this as a constant in profiles.py,
            # you can thread it via cfg if you prefer.
            2_000,  # ms
            inputs.repo,
            inputs.commit,
        ],
    ).fetchall()

    result: dict[int, CoverageSummary] = {}
    for (
        function_goid_h128,
        lines_covered,
        lines_total,
        test_count,
        flaky_tests,
        slow_tests,
    ) in rows:
        lines_cov = int(lines_covered or 0)
        lines_tot = int(lines_total or 0)
        coverage_ratio = (lines_cov / lines_tot) if lines_tot > 0 else None
        result[int(function_goid_h128)] = CoverageSummary(
            function_goid_h128=int(function_goid_h128),
            lines_covered=lines_cov,
            lines_total=lines_tot,
            coverage_ratio=coverage_ratio,
            covered_by_tests=int(test_count or 0),
            flaky_tests=int(flaky_tests or 0),
            slow_tests=int(slow_tests or 0),
        )
    return result
```

You don’t have to be this literal; the key point is: **this entire concern is its own function**.

### 2.4 Tests + behavior: `join_function_tests`

Similarly, factor out the `test_stats` / `behavior` part of your SQL.

```python
def join_function_tests(inputs: FunctionProfileInputs) -> Mapping[int, TestSummary]:
    """
    Combine test coverage and behavioral tags per function.

    Derived from analytics.test_coverage_edges + analytics.test_profile.
    """
    con = inputs.con
    rows = con.execute(
        """
        WITH behavior AS (
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id) AS test_count,
                MODE() WITHIN GROUP (ORDER BY tp.status) AS dominant_status,
                MODE() WITHIN GROUP (ORDER BY tp.behavior_tags_json) AS behavior_tags_json
            FROM analytics.test_coverage_edges AS e
            LEFT JOIN analytics.test_profile AS tp
              ON tp.test_id = e.test_id
             AND tp.repo = e.repo
             AND tp.commit = e.commit
            WHERE e.repo = ? AND e.commit = ?
            GROUP BY e.function_goid_h128
        )
        SELECT
            function_goid_h128,
            test_count,
            dominant_status,
            behavior_tags_json
        FROM behavior
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, TestSummary] = {}
    for function_goid_h128, test_count, dominant_status, behavior_tags_json in rows:
        result[int(function_goid_h128)] = TestSummary(
            function_goid_h128=int(function_goid_h128),
            test_count=int(test_count or 0),
            dominant_status=str(dominant_status) if dominant_status is not None else None,
            behavior_tags_json=str(behavior_tags_json) if behavior_tags_json is not None else None,
        )
    return result
```

Again: you already have some of this logic in `analytics/tests/profiles.py`; this helper can re-use those modules once you split them (see section 3).

### 2.5 Graph features: `join_function_graph_metrics`

Here we leverage the `FunctionGraphMetricInputs` + `centrality_directed` / `neighbor_stats` that you already have under `analytics.graph_service` / `analytics.graph_rows.graph_metrics`.

In `analytics/profiles/graph_features.py`:

```python
# analytics/profiles/graph_features.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    neighbor_stats,
    component_metadata,
)
from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import GraphContextSpec, resolve_graph_context
from codeintel.config import GraphMetricsStepConfig
from codeintel.storage.gateway import StorageGateway

from .types import FunctionGraphFeatures


def summarize_graph_for_function_profile(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> Mapping[int, FunctionGraphFeatures]:
    """
    Build FunctionGraphFeatures keyed by function_goid_h128.

    This should mirror the call-graph CTEs inside build_function_profile,
    but via the shared graph_service APIs.
    """
    runtime_opts = runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    runtime = resolve_graph_runtime(runtime_opts)

    ctx_spec = GraphContextSpec.from_config(cfg)
    ctx: GraphContext = resolve_graph_context(gateway, ctx_spec, runtime=runtime)

    graph = runtime.ensure_call_graph()
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality = centrality_directed(graph, ctx)
    components = component_metadata(graph)

    features: dict[int, FunctionGraphFeatures] = {}

    # You already have a mapping from node -> GOID in call_graph_nodes / docs.v_call_graph_enriched.
    rows = gateway.con.execute(
        """
        SELECT
            goid_h128,
            node_id
        FROM docs.v_call_graph_enriched
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()

    for goid_h128, node_id in rows:
        node = str(node_id)

        # centrality is a Mapping[str, Mapping[Any, float]]
        pagerank = centrality.pagerank.get(node)
        betweenness = centrality.betweenness.get(node)
        indeg = centrality.in_degree_centrality.get(node)
        outdeg = centrality.out_degree_centrality.get(node)

        comp_meta = components.component_meta.get(node, {})
        component_id = int(comp_meta.get("component_id", -1))
        in_cycle = bool(comp_meta.get("in_cycle", False))

        stats_for_node = stats.node_stats.get(node)
        call_fan_in = stats_for_node.in_degree if stats_for_node is not None else 0
        call_fan_out = stats_for_node.out_degree if stats_for_node is not None else 0

        features[int(goid_h128)] = FunctionGraphFeatures(
            function_goid_h128=int(goid_h128),
            call_fan_in=int(call_fan_in),
            call_fan_out=int(call_fan_out),
            pagerank=pagerank,
            betweenness=betweenness,
            in_degree_centrality=indeg,
            out_degree_centrality=outdeg,
            component_id=component_id,
            in_cycle=in_cycle,
        )

    return features
```

This gives you a **single contract** for graph usage in profiles.

Then `join_function_graph_metrics` in `analytics/profiles/functions.py` can simply adapt this into a `Mapping[int, FunctionGraphFeatures]` via `summarize_graph_for_function_profile`.

### 2.6 Docstrings & history

Same pattern: pull each concern into its own helper, using the existing CTEs/relations.

```python
def join_function_docs(inputs: FunctionProfileInputs) -> Mapping[int, FunctionDocView]:
    con = inputs.con
    rows = con.execute(
        """
        SELECT
            function_goid_h128,
            short_desc,
            long_desc,
            params,
            returns
        FROM docs.v_function_docs
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, FunctionDocView] = {}
    for function_goid_h128, short_desc, long_desc, params, returns in rows:
        result[int(function_goid_h128)] = FunctionDocView(
            function_goid_h128=int(function_goid_h128),
            short_desc=short_desc,
            long_desc=long_desc,
            params_json=params,
            returns_json=returns,
        )
    return result


def join_function_history(inputs: FunctionProfileInputs) -> Mapping[int, FunctionHistoryView]:
    con = inputs.con
    rows = con.execute(
        """
        SELECT
            function_goid_h128,
            created_at,
            last_modified_at,
            churn_score,
            commit_count,
            author_count
        FROM analytics.function_history
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, FunctionHistoryView] = {}
    for (
        function_goid_h128,
        created_at,
        last_modified_at,
        churn_score,
        commit_count,
        author_count,
    ) in rows:
        result[int(function_goid_h128)] = FunctionHistoryView(
            function_goid_h128=int(function_goid_h128),
            created_at=created_at,
            last_modified_at=last_modified_at,
            churn_score=float(churn_score) if churn_score is not None else None,
            commit_count=int(commit_count or 0),
            author_count=int(author_count or 0),
        )
    return result
```

### 2.7 Building final rows: `build_function_profile_rows`

Now we’re just merging dictionaries by `function_goid_h128` and assembling row models.

First, add row models to `storage/rows.py` (we’ll cover this in section 4), e.g.:

```python
# storage/rows.py
class FunctionProfileRowModel(TypedDict):
    repo: str
    commit: str
    function_goid_h128: int
    urn: str
    rel_path: str
    module: str
    language: str
    kind: str
    qualname: str
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    # ... plus all analytics columns: risk, coverage, tests, graph, docs, history, etc.


def function_profile_row_to_tuple(row: FunctionProfileRowModel) -> tuple[object, ...]:
    """Serialize a FunctionProfileRowModel into INSERT column order."""
    return (
        row["repo"],
        row["commit"],
        row["function_goid_h128"],
        row["urn"],
        # etc...
    )
```

Then in `analytics/profiles/functions.py`:

```python
@dataclass(frozen=True)
class FunctionBaseInfo:
    """Static/non-analytics fields loaded from docs.v_function_summary or similar."""

    function_goid_h128: int
    urn: str
    rel_path: str
    module: str
    language: str
    kind: str
    qualname: str
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    param_count: int
    total_params: int
    annotated_params: int
    return_type: str | None


def load_function_base_info(inputs: FunctionProfileInputs) -> Mapping[int, FunctionBaseInfo]:
    con = inputs.con
    rows = con.execute(
        """
        SELECT
            function_goid_h128,
            urn,
            rel_path,
            module,
            language,
            kind,
            qualname,
            loc,
            logical_loc,
            cyclomatic_complexity,
            param_count,
            total_params,
            annotated_params,
            return_type
        FROM docs.v_function_summary
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, FunctionBaseInfo] = {}
    for row in rows:
        (
            function_goid_h128,
            urn,
            rel_path,
            module,
            language,
            kind,
            qualname,
            loc,
            logical_loc,
            cyclomatic_complexity,
            param_count,
            total_params,
            annotated_params,
            return_type,
        ) = row
        result[int(function_goid_h128)] = FunctionBaseInfo(
            function_goid_h128=int(function_goid_h128),
            urn=str(urn),
            rel_path=str(rel_path),
            module=str(module),
            language=str(language),
            kind=str(kind),
            qualname=str(qualname),
            loc=int(loc or 0),
            logical_loc=int(logical_loc or 0),
            cyclomatic_complexity=int(cyclomatic_complexity or 0),
            param_count=int(param_count or 0),
            total_params=int(total_params or 0),
            annotated_params=int(annotated_params or 0),
            return_type=str(return_type) if return_type is not None else None,
        )
    return result
```

Finally, merge everything:

```python
def build_function_profile_rows(
    inputs: FunctionProfileInputs,
    *,
    risk_by_func: Mapping[int, FunctionRiskView],
    cov_by_func: Mapping[int, CoverageSummary],
    tests_by_func: Mapping[int, TestSummary],
    graph_by_func: Mapping[int, FunctionGraphFeatures],
    docs_by_func: Mapping[int, FunctionDocView],
    history_by_func: Mapping[int, FunctionHistoryView],
) -> Iterable[FunctionProfileRowModel]:
    """
    Assemble the final FunctionProfile rows by merging all feature views.

    Missing views for a function are tolerated and filled with defaults.
    """
    base_info = load_function_base_info(inputs)

    for goid, base in base_info.items():
        risk = risk_by_func.get(goid)
        cov = cov_by_func.get(goid)
        tests = tests_by_func.get(goid)
        graph = graph_by_func.get(goid)
        docs = docs_by_func.get(goid)
        hist = history_by_func.get(goid)

        row: FunctionProfileRowModel = {
            "repo": inputs.repo,
            "commit": inputs.commit,
            "function_goid_h128": goid,
            "urn": base.urn,
            "rel_path": base.rel_path,
            "module": base.module,
            "language": base.language,
            "kind": base.kind,
            "qualname": base.qualname,
            "loc": base.loc,
            "logical_loc": base.logical_loc,
            "cyclomatic_complexity": base.cyclomatic_complexity,
            "param_count": base.param_count,
            "total_params": base.total_params,
            "annotated_params": base.annotated_params,
            "return_type": base.return_type,
            # risk
            "risk_score": risk.risk_score if risk else 0.0,
            "risk_factors_json": risk.risk_factors_json if risk else "[]",
            "risk_bucket": risk.risk_bucket if risk else None,
            # coverage
            "lines_covered": cov.lines_covered if cov else 0,
            "lines_total": cov.lines_total if cov else 0,
            "coverage_ratio": cov.coverage_ratio if cov else None,
            "covered_by_tests": cov.covered_by_tests if cov else 0,
            "flaky_tests": cov.flaky_tests if cov else 0,
            "slow_tests": cov.slow_tests if cov else 0,
            # tests / behavior
            "test_count": tests.test_count if tests else 0,
            "dominant_test_status": tests.dominant_status if tests else None,
            "behavior_tags_json": tests.behavior_tags_json if tests else None,
            # graph
            "call_fan_in": graph.call_fan_in if graph else 0,
            "call_fan_out": graph.call_fan_out if graph else 0,
            "pagerank": graph.pagerank if graph else None,
            "betweenness": graph.betweenness if graph else None,
            "in_degree_centrality": graph.in_degree_centrality if graph else None,
            "out_degree_centrality": graph.out_degree_centrality if graph else None,
            "component_id": graph.component_id if graph else -1,
            "in_cycle": graph.in_cycle if graph else False,
            # history
            "created_at": hist.created_at if hist else None,
            "last_modified_at": hist.last_modified_at if hist else None,
            "churn_score": hist.churn_score if hist else None,
            "commit_count": hist.commit_count if hist else 0,
            "author_count": hist.author_count if hist else 0,
            # created_at / updated_at columns for profile itself:
            "profile_created_at": inputs.created_at,
        }

        yield row
```

### 2.8 Writing rows: `write_function_profile_rows`

Leverage `macro_insert_rows` via the gateway, keeping the insert path consistent with the rest of your ingestion.

```python
def write_function_profile_rows(
    gateway: StorageGateway,
    rows: Iterable[FunctionProfileRowModel],
) -> int:
    """
    Insert function_profile rows into analytics.function_profile.

    Returns the number of inserted rows.
    """
    con = gateway.con
    tuples = [function_profile_row_to_tuple(row) for row in rows]

    con.execute(
        """
        DELETE FROM analytics.function_profile
        WHERE repo = ? AND commit = ?
        """,
        [rows[0]["repo"], rows[0]["commit"]] if tuples else [],
    )
    if not tuples:
        return 0

    con.executemany(
        """
        INSERT INTO analytics.function_profile (
            -- column list aligned with function_profile_row_to_tuple
            repo,
            commit,
            function_goid_h128,
            urn,
            rel_path,
            module,
            language,
            kind,
            qualname,
            loc,
            logical_loc,
            cyclomatic_complexity,
            param_count,
            total_params,
            annotated_params,
            return_type,
            risk_score,
            risk_factors_json,
            risk_bucket,
            lines_covered,
            lines_total,
            coverage_ratio,
            covered_by_tests,
            flaky_tests,
            slow_tests,
            test_count,
            dominant_test_status,
            behavior_tags_json,
            call_fan_in,
            call_fan_out,
            pagerank,
            betweenness,
            in_degree_centrality,
            out_degree_centrality,
            component_id,
            in_cycle,
            created_at,
            last_modified_at,
            churn_score,
            commit_count,
            author_count,
            profile_created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tuples,
    )

    return len(tuples)
```

*(You already have similar patterns elsewhere; you can swap this for `macro_insert_rows` if you prefer.)*

### 2.9 New orchestration: replace `build_function_profile`

Finally, shrink `analytics/profiles.py` to the orchestrator you actually want:

```python
# analytics/profiles.py (rewritten, but keep public signature)
from __future__ import annotations

import logging

from codeintel.analytics.context import AnalyticsContext
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway

from codeintel.analytics.profiles.functions import (
    compute_function_profile_inputs,
    join_function_risk,
    join_function_coverage,
    join_function_tests,
    join_function_docs,
    join_function_history,
    build_function_profile_rows,
    write_function_profile_rows,
)
from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile

log = logging.getLogger(__name__)


def build_function_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    catalog_provider=None,  # kept for now, can be used by load_function_base_info
    context: AnalyticsContext | None = None,
) -> None:
    """
    Orchestrate function_profile build using small, testable helpers.
    """
    log.info("Building analytics.function_profile for %s@%s", cfg.repo, cfg.commit)
    inputs = compute_function_profile_inputs(gateway, cfg)

    risk_by_func = join_function_risk(inputs)
    cov_by_func = join_function_coverage(inputs)
    tests_by_func = join_function_tests(inputs)
    docs_by_func = join_function_docs(inputs)
    history_by_func = join_function_history(inputs)

    from codeintel.config import GraphMetricsStepConfig

    graph_cfg = GraphMetricsStepConfig(snapshot=cfg.snapshot)
    graph_by_func = summarize_graph_for_function_profile(
        gateway,
        graph_cfg,
        runtime=context.graph_runtime if context is not None else None,
    )

    rows = list(
        build_function_profile_rows(
            inputs,
            risk_by_func=risk_by_func,
            cov_by_func=cov_by_func,
            tests_by_func=tests_by_func,
            graph_by_func=graph_by_func,
            docs_by_func=docs_by_func,
            history_by_func=history_by_func,
        )
    )

    count = write_function_profile_rows(gateway, rows)
    log.info("function_profile populated: %s rows for %s@%s", count, cfg.repo, cfg.commit)
```

From the pipeline’s perspective (`ProfilesStep.run`), nothing changes: it still calls `build_function_profile(gateway, cfg, ...)`, but the inner structure is now **recipe-like**.

---

## 3. Apply the same pattern to file & module profiles

You don’t need as much complexity here; conceptually it’s identical:

* `analytics/profiles/files.py`:

  * `compute_file_profile_inputs` (same shape, or reuse `FunctionProfileInputs` renamed to `ProfileInputs`)
  * `join_file_risk`
  * `join_file_coverage`
  * `join_file_history`
  * `build_file_profile_rows`
  * `write_file_profile_rows`

* `analytics/profiles/modules.py`:

  * Similar helpers; much of this is currently in the module branch of your existing SQL.

Then in `analytics/profiles.py`:

```python
from codeintel.analytics.profiles.files import (
    build_file_profile_rows,
    compute_file_profile_inputs,
    write_file_profile_rows,
    # etc...
)
from codeintel.analytics.profiles.modules import (
    build_module_profile_rows,
    compute_module_profile_inputs,
    write_module_profile_rows,
)

def build_file_profile(...): ...
def build_module_profile(...): ...
```

Each becomes a ~30–40 line orchestrator that:

1. Builds inputs
2. Calls a small set of `compute_*/join_*` helpers
3. Builds rows
4. Writes rows

---

## 4. Split **test profiles** into recipe modules

Your current `analytics/tests/profiles.py` is already helper-heavy but monolithic. The goal is to mirror what we just did for functions.

### 4.1 New subpackage layout

Create:

```text
analytics/tests_profiles/
  __init__.py
  coverage_inputs.py
  behavioral_tags.py
  importance.py
  rows.py
```

Example:

* `coverage_inputs.py`:

  * `load_test_coverage_edges`
  * `aggregate_test_coverage_by_function`
  * `aggregate_test_coverage_by_subsystem`

* `behavioral_tags.py`:

  * `infer_behavior_tags` (moving your AST + name + markers heuristics here)
  * `_tags_from_name`, `_tags_from_markers`, `_tags_from_io_flags`, `_tags_from_ast_info` live here

* `importance.py`:

  * `compute_flakiness_score`
  * `compute_importance_score`
  * `compute_test_weight`

* `rows.py`:

  * `build_test_profile_rows`
  * `build_behavioral_coverage_rows`

Then, shrink `analytics/tests/profiles.py` to:

```python
# analytics/tests/profiles.py (orchestrator only)
from __future__ import annotations

import logging

from codeintel.config import TestProfileStepConfig
from codeintel.pipeline.orchestration.core import PipelineContext
from codeintel.storage.gateway import StorageGateway

from codeintel.analytics.tests_profiles.coverage_inputs import (
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
)
from codeintel.analytics.tests_profiles.behavioral_tags import infer_behavior_tags
from codeintel.analytics.tests_profiles.importance import compute_importance_score
from codeintel.analytics.tests_profiles.rows import (
    build_test_profile_rows,
    build_behavioral_coverage_rows,
    write_test_profile_rows,
    write_behavioral_coverage_rows,
)

log = logging.getLogger(__name__)


def build_test_profile(gateway: StorageGateway, cfg: TestProfileStepConfig) -> None:
    """
    Orchestrate analytics.test_profile using coverage + behavior + importance.
    """
    log.info("Building analytics.test_profile for %s@%s", cfg.repo, cfg.commit)

    cov_by_test = aggregate_test_coverage_by_function(gateway.con, cfg)
    behavior_by_test = infer_behavior_tags(gateway.con, cfg)
    importance_by_test = compute_importance_score(cov_by_test, behavior_by_test, cfg)

    rows = build_test_profile_rows(cfg, cov_by_test, behavior_by_test, importance_by_test)
    count = write_test_profile_rows(gateway, rows)
    log.info("test_profile populated: %s rows", count)


def build_behavioral_coverage(gateway: StorageGateway, cfg: TestProfileStepConfig) -> None:
    """
    Orchestrate analytics.behavioral_coverage from test profile + coverage.
    """
    log.info("Building analytics.behavioral_coverage for %s@%s", cfg.repo, cfg.commit)

    cov_by_function = aggregate_test_coverage_by_subsystem(gateway.con, cfg)
    rows = build_behavioral_coverage_rows(cfg, cov_by_function)
    count = write_behavioral_coverage_rows(gateway, rows)
    log.info("behavioral_coverage populated: %s rows", count)
```

The **semantics stay identical**, but the logic is now grouped by concern and testable independently.

---

## 5. Value objects & row models in `storage/rows.py`

This satisfies your item (c): push row models down.

Add new TypedDicts + serializers for:

* `FunctionProfileRowModel` / `function_profile_row_to_tuple`
* `FileProfileRowModel` / `file_profile_row_to_tuple`
* `ModuleProfileRowModel` / `module_profile_row_to_tuple`
* `TestProfileRowModel` / `test_profile_row_to_tuple`
* `BehavioralCoverageRowModel` / `behavioral_coverage_row_to_tuple`

Example for functions (pattern is same for others):

```python
# storage/rows.py
class FunctionProfileRowModel(TypedDict):
    repo: str
    commit: str
    function_goid_h128: int
    urn: str
    rel_path: str
    module: str
    language: str
    kind: str
    qualname: str
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    param_count: int
    total_params: int
    annotated_params: int
    return_type: object
    risk_score: float
    risk_factors_json: object
    risk_bucket: object
    lines_covered: int
    lines_total: int
    coverage_ratio: float | None
    covered_by_tests: int
    flaky_tests: int
    slow_tests: int
    test_count: int
    dominant_test_status: object
    behavior_tags_json: object
    call_fan_in: int
    call_fan_out: int
    pagerank: float | None
    betweenness: float | None
    in_degree_centrality: float | None
    out_degree_centrality: float | None
    component_id: int
    in_cycle: bool
    created_at: object
    last_modified_at: object
    churn_score: float | None
    commit_count: int
    author_count: int
    profile_created_at: object


def function_profile_row_to_tuple(row: FunctionProfileRowModel) -> tuple[object, ...]:
    return (
        row["repo"],
        row["commit"],
        row["function_goid_h128"],
        row["urn"],
        row["rel_path"],
        row["module"],
        row["language"],
        row["kind"],
        row["qualname"],
        row["loc"],
        row["logical_loc"],
        row["cyclomatic_complexity"],
        row["param_count"],
        row["total_params"],
        row["annotated_params"],
        row["return_type"],
        row["risk_score"],
        row["risk_factors_json"],
        row["risk_bucket"],
        row["lines_covered"],
        row["lines_total"],
        row["coverage_ratio"],
        row["covered_by_tests"],
        row["flaky_tests"],
        row["slow_tests"],
        row["test_count"],
        row["dominant_test_status"],
        row["behavior_tags_json"],
        row["call_fan_in"],
        row["call_fan_out"],
        row["pagerank"],
        row["betweenness"],
        row["in_degree_centrality"],
        row["out_degree_centrality"],
        row["component_id"],
        row["in_cycle"],
        row["created_at"],
        row["last_modified_at"],
        row["churn_score"],
        row["commit_count"],
        row["author_count"],
        row["profile_created_at"],
    )
```

These row models give you **compile-time field checking** and a shared contract between analytics + storage.

---

## 6. Wiring & rollout

### 6.1 Pipeline step remains the same

In `pipeline/orchestration/steps_analytics.py`, `ProfilesStep.run` is already calling:

```python
build_function_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
build_file_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
build_module_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
```

Since we kept those names/signatures, you don’t need to change the pipeline at all; the step automatically uses the refactored implementation.

### 6.2 Testing strategy

To avoid subtle regressions:

1. **Snapshot-based equivalence test**

   * Before refactor, run pipeline on a small but representative repo and snapshot `analytics.function_profile`, `analytics.file_profile`, `analytics.module_profile` to Parquet.
   * After refactor, run again and snapshot to a second set of Parquet files.
   * Write a test that loads both and asserts:

     * Same row count
     * Same set of primary keys (`repo, commit, function_goid_h128`)
     * For numeric columns: either exact equality or within a tiny epsilon for floats
     * For JSON columns: parse + compare normalized objects

2. **Unit tests per helper**

   * For each `join_*` helper, spin up an **in-memory DuckDB** with:

     * `CREATE TABLE` statements matching schema for only the relevant tables
     * A tiny number of rows (2–3 functions, some with partial data)
   * Assert the mapping outputs exactly as expected.

3. **Graph features sanity tests**

   * Inject a tiny call graph into DuckDB / GraphRuntime:

     * 3–4 functions with known in/out degrees and centrality.
   * Assert that `summarize_graph_for_function_profile` returns non-empty, consistent `FunctionGraphFeatures`.

4. **Performance sanity**

   * Compare time of old vs new implementation on a mid-size repo snapshot; they should be in the same ballpark. If you see a regression, you can:

     * Move some merging work back into DuckDB via CTEs in `build_function_profile_rows`, or
     * Use temporary tables/views for intermediate results.

---

If you’d like, I can next:

* Turn this into a **step-by-step patch plan** (ordered diffs for each file), *or*

# Analytics refactor step by step patch plan #

Got you 👍 — let’s turn that design into something your implementation agent can actually chew on.

I’ll assume paths like:

* `src/codeintel/analytics/...`
* `src/codeintel/storage/...`
* `src/codeintel/config/...`

If your layout is slightly different, just adjust the prefixes.

I’ll give you:

1. **Step ordering** (what to do first, second, …)
2. **Concrete diffs** (new files + edits) that define the *target* structure
3. Where appropriate, I’ll point out “move this logic from X → Y” instead of inlining 400-line SQL blocks.

You can hand this whole thing straight to an LLM implementation agent and say “apply these patches, preserving semantics.”

---

## Step 0 – High-level sequencing

Apply these in roughly this order:

1. Create the **profiles subpackage** and core **types**.
2. Add **function-profile recipe helpers** (skeletons, but with real types).
3. Add **row models** for profiles in `storage/rows.py`.
4. Convert `analytics/profiles.py` into a **thin orchestrator** that calls the recipe helpers.
5. Split **test profiles** into their own mini-package and make `analytics/tests/profiles.py` an orchestrator.

You can treat each numbered step below as a “commit”.

---

## Step 1 – New `analytics.profiles` subpackage

### 1.1 Create `src/codeintel/analytics/profiles/__init__.py`

```diff
diff --git a/src/codeintel/analytics/profiles/__init__.py b/src/codeintel/analytics/profiles/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
--- /dev/null
+++ b/src/codeintel/analytics/profiles/__init__.py
@@ -0,0 +1,20 @@
+"""Composable profile “recipes” for functions, files, and modules.
+
+This subpackage factors the large aggregators in :mod:`codeintel.analytics.profiles`
+into small, testable helpers:
+
+* Common value objects live in :mod:`codeintel.analytics.profiles.types`
+* Function profile assembly lives in :mod:`codeintel.analytics.profiles.functions`
+* Graph-centric helpers live in :mod:`codeintel.analytics.profiles.graph_features`
+* File / module profile recipes live in :mod:`codeintel.analytics.profiles.files`
+  and :mod:`codeintel.analytics.profiles.modules`.
+
+The goal is to make the analytics surface easier for both humans and LLM agents
+to reason about by breaking monolithic functions into explicit recipes.
+"""
+
+from . import types as types  # re-export for convenience
+
+__all__ = [
+    "types",
+]
```

### 1.2 Create `src/codeintel/analytics/profiles/types.py`

This defines the **value objects** used by profile recipes.

```diff
diff --git a/src/codeintel/analytics/profiles/types.py b/src/codeintel/analytics/profiles/types.py
new file mode 100644
index 0000000000000000000000000000000000000000..bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
--- /dev/null
+++ b/src/codeintel/analytics/profiles/types.py
@@ -0,0 +1,181 @@
+"""Shared value objects for analytics profile recipes.
+
+These dataclasses are intentionally small, immutable “snapshots” of the
+ingredients that go into function/file/module profiles.  They keep the recipe
+code simple and make it much easier to unit test in isolation.
+"""
+
+from __future__ import annotations
+
+from collections.abc import Mapping
+from dataclasses import dataclass
+from datetime import datetime
+from typing import Any
+
+from codeintel.storage.gateway import DuckDBConnection
+
+
+@dataclass(frozen=True)
+class FunctionProfileInputs:
+    """Handle object for function profile computations.
+
+    This does *not* eagerly materialize large relations; it just carries the
+    connection and snapshot identity.  Helpers use ``con`` to run focused
+    queries.
+    """
+
+    con: DuckDBConnection
+    repo: str
+    commit: str
+    created_at: datetime
+
+
+@dataclass(frozen=True)
+class FunctionRiskView:
+    """Summarized risk factors per function."""
+
+    function_goid_h128: int
+    risk_score: float
+    risk_factors_json: str
+    risk_level: str | None
+    risk_component_coverage: float | None
+    risk_component_complexity: float | None
+    risk_component_static: float | None
+    risk_component_hotspot: float | None
+
+
+@dataclass(frozen=True)
+class CoverageSummary:
+    """Aggregated coverage metrics for a single function."""
+
+    function_goid_h128: int
+    executable_lines: int
+    covered_lines: int
+    coverage_ratio: float | None
+    tested: bool
+    untested_reason: str | None
+    tests_touching: int
+    failing_tests: int
+    slow_tests: int
+    flaky_tests: int
+    last_test_status: str | None
+    dominant_test_status: str | None
+    slow_test_threshold_ms: float | None
+
+
+@dataclass(frozen=True)
+class TestSummary:
+    """Test-centric signals used by function_profile."""
+
+    function_goid_h128: int
+    test_count: int
+    dominant_status: str | None
+    behavior_tags_json: str | None
+
+
+@dataclass(frozen=True)
+class FunctionGraphFeatures:
+    """Call-graph-driven metrics used in :mod:`analytics.function_profile`."""
+
+    function_goid_h128: int
+    call_fan_in: int
+    call_fan_out: int
+    call_edge_in_count: int
+    call_edge_out_count: int
+    call_is_leaf: bool
+    call_is_entrypoint: bool
+    call_is_public: bool
+
+
+@dataclass(frozen=True)
+class FunctionDocView:
+    """Docstring-derived information per function."""
+
+    function_goid_h128: int
+    doc_short: str | None
+    doc_long: str | None
+    doc_params: str | None
+    doc_returns: str | None
+
+
+@dataclass(frozen=True)
+class FunctionHistoryView:
+    """History and churn metrics per function."""
+
+    function_goid_h128: int
+    created_in_commit: str | None
+    created_at_history: datetime | None
+    last_modified_commit: str | None
+    last_modified_at: datetime | None
+    age_days: int | None
+    commit_count: int
+    author_count: int
+    lines_added: int
+    lines_deleted: int
+    churn_score: float | None
+    stability_bucket: str | None
+
+
+@dataclass(frozen=True)
+class FunctionBaseInfo:
+    """Static/non-analytics fields from docs and symbol tables."""
+
+    function_goid_h128: int
+    urn: str
+    repo: str
+    commit: str
+    rel_path: str
+    module: str
+    language: str
+    kind: str
+    qualname: str
+    start_line: int | None
+    end_line: int | None
+    loc: int
+    logical_loc: int
+    cyclomatic_complexity: int
+    complexity_bucket: str | None
+    param_count: int
+    positional_params: int
+    keyword_params: int
+    vararg: bool
+    kwarg: bool
+    max_nesting_depth: int | None
+    stmt_count: int | None
+    decorator_count: int | None
+    has_docstring: bool
+    total_params: int
+    annotated_params: int
+    return_type: str | None
+    param_types: str | None
+    fully_typed: bool
+    partial_typed: bool
+    untyped: bool
+    typedness_bucket: str | None
+    typedness_source: str | None
+    file_typed_ratio: float | None
+    static_error_count: int
+    has_static_errors: bool
+
+
+# NOTE:
+# -----
+# This module deliberately does *not* define row shapes for the DuckDB tables;
+# those live in :mod:`codeintel.storage.rows` so that both analytics and
+# storage share a single contract.  These value objects are “pre-row” domain
+# types used to assemble those rows.
```

We’re not yet wiring any logic — just giving your agents a clear type vocabulary.

---

## Step 2 – Add function-profile recipe helpers (skeletons)

We’ll create `functions.py` with clear signatures and docstrings. The actual SQL joins will be *moved* from `analytics/profiles.py` into these helpers by your implementation agent.

### 2.1 Create `src/codeintel/analytics/profiles/functions.py`

```diff
diff --git a/src/codeintel/analytics/profiles/functions.py b/src/codeintel/analytics/profiles/functions.py
new file mode 100644
index 0000000000000000000000000000000000000000..cccccccccccccccccccccccccccccccccccccccc
--- /dev/null
+++ b/src/codeintel/analytics/profiles/functions.py
@@ -0,0 +1,268 @@
+"""Function profile recipe helpers.
+
+This module is the main “home” for :mod:`analytics.function_profile` logic.
+Instead of one 400-line function, we expose a sequence of small, focused
+helpers that each concern themselves with one slice of the profile:
+
+* static base info                → :class:`FunctionBaseInfo`
+* risk signals                    → :class:`FunctionRiskView`
+* coverage & tests                → :class:`CoverageSummary`, :class:`TestSummary`
+* graph-derived features          → :class:`FunctionGraphFeatures`
+* docstring surfaces              → :class:`FunctionDocView`
+* history / churn                 → :class:`FunctionHistoryView`
+
+The target end-state is that :func:`build_function_profile_recipe` becomes
+a ~40-line orchestrator that calls these helpers and then writes
+`analytics.function_profile` rows via :mod:`codeintel.storage.rows`.
+"""
+
+from __future__ import annotations
+
+from collections.abc import Iterable, Mapping
+from datetime import UTC, datetime
+from typing import cast
+
+from codeintel.analytics.context import AnalyticsContext
+from codeintel.config import GraphMetricsStepConfig, ProfilesAnalyticsStepConfig
+from codeintel.storage.gateway import StorageGateway
+from codeintel.storage.rows import FunctionProfileRowModel, function_profile_row_to_tuple
+
+from .types import (
+    CoverageSummary,
+    FunctionBaseInfo,
+    FunctionDocView,
+    FunctionGraphFeatures,
+    FunctionHistoryView,
+    FunctionProfileInputs,
+    FunctionRiskView,
+    TestSummary,
+)
+
+
+# -----------------------------------------------------------------------------
+# Inputs
+# -----------------------------------------------------------------------------
+
+
+def compute_function_profile_inputs(
+    gateway: StorageGateway,
+    cfg: ProfilesAnalyticsStepConfig,
+) -> FunctionProfileInputs:
+    """Construct the handle object used by the recipe helpers.
+
+    This intentionally does *not* compute anything expensive.  It simply
+    normalizes the connection and snapshot identity so helpers can query
+    against the right repo/commit.
+    """
+    con = gateway.con
+    now = datetime.now(tz=UTC)
+
+    return FunctionProfileInputs(
+        con=con,
+        repo=cfg.repo,
+        commit=cfg.commit,
+        created_at=now,
+    )
+
+
+# -----------------------------------------------------------------------------
+# Per-concern helpers
+# -----------------------------------------------------------------------------
+
+
+def load_function_base_info(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, FunctionBaseInfo]:
+    """Load static per-function info used as the base of the profile.
+
+    **Implementation note for refactor:**
+
+    The body of this helper should be derived from the “base” CTE in the
+    existing :func:`codeintel.analytics.profiles.build_function_profile`
+    query — the part that pulls from `docs.v_function_summary` /
+    `core.functions` / typing / static-analysis tables.
+
+    For the initial migration, have your implementation agent:
+
+    * copy the SELECT for static columns into a standalone query here
+    * iterate over ``fetchall()`` and construct :class:`FunctionBaseInfo`
+      instances keyed by ``function_goid_h128``.
+    """
+    con = inputs.con
+
+    # TODO: move static “base” CTE / SELECT from build_function_profile here.
+    #
+    # Pseudocode:
+    # rows = con.execute("SELECT ... FROM docs.v_function_summary JOIN ... WHERE repo = ? AND commit = ?", [inputs.repo, inputs.commit]).fetchall()
+    # result: dict[int, FunctionBaseInfo] = {}
+    # for row in rows:
+    #     result[row.function_goid_h128] = FunctionBaseInfo(...)
+    # return result
+    raise NotImplementedError("load_function_base_info must be implemented from existing SQL CTEs.")
+
+
+def join_function_risk(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, FunctionRiskView]:
+    """Summarize risk factors per function.
+
+    Implementation should be derived from the `rf AS (...)` CTE in the
+    existing function_profile SQL, typically backed by
+    `analytics.goid_risk_factors`.
+    """
+    con = inputs.con
+
+    # TODO: copy risk CTE/SELECT from build_function_profile and hydrate
+    # FunctionRiskView instances keyed by function_goid_h128.
+    raise NotImplementedError("join_function_risk must be implemented from existing SQL CTEs.")
+
+
+def join_function_coverage(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, CoverageSummary]:
+    """Aggregate coverage/test signals per function.
+
+    Implementation should be derived from the coverage-related CTEs in the
+    current SQL (e.g. coverage_functions + test_coverage_edges joins).
+    """
+    con = inputs.con
+
+    # TODO: copy coverage CTE/SELECT from build_function_profile and hydrate
+    # CoverageSummary instances keyed by function_goid_h128.
+    raise NotImplementedError("join_function_coverage must be implemented from existing SQL CTEs.")
+
+
+def join_function_tests(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, TestSummary]:
+    """Combine behavioral/test status signals per function.
+
+    This should mirror the parts of the existing SQL that pull in
+    `analytics.test_profile` (or equivalent) and derive dominant status,
+    behavioral tags, etc.
+    """
+    con = inputs.con
+
+    # TODO: move test-behavior CTE/SELECT from build_function_profile here.
+    raise NotImplementedError("join_function_tests must be implemented from existing SQL CTEs.")
+
+
+def join_function_graph_metrics(
+    inputs: FunctionProfileInputs,
+    cfg: GraphMetricsStepConfig,
+    *,
+    context: AnalyticsContext | None = None,
+) -> Mapping[int, FunctionGraphFeatures]:
+    """Summarize call-graph metrics per function.
+
+    For the first pass, you probably want to *reuse* the existing `cg_in`,
+    `cg_out`, and `cg_degrees` CTEs that aggregate `graph.call_graph_edges`
+    and `graph.call_graph_nodes` rather than recomputing metrics from a
+    live graph runtime.
+
+    That logic should be moved here so it can be tested in isolation.
+    """
+    con = inputs.con
+
+    # TODO: move cg_in/cg_out/cg_degrees CTEs from build_function_profile here
+    # and hydrate FunctionGraphFeatures keyed by function_goid_h128.
+    raise NotImplementedError("join_function_graph_metrics must be implemented from existing SQL CTEs.")
+
+
+def join_function_docs(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, FunctionDocView]:
+    """Attach docstring surfaces to each function.
+
+    Implementation should be based on the join against `docs.v_function_docs`
+    in the current SQL, mapping to :class:`FunctionDocView`.
+    """
+    con = inputs.con
+
+    # TODO: move docstring SELECT from build_function_profile here.
+    raise NotImplementedError("join_function_docs must be implemented from existing SQL CTEs.")
+
+
+def join_function_history(
+    inputs: FunctionProfileInputs,
+) -> Mapping[int, FunctionHistoryView]:
+    """Attach function_history metrics to each function."""
+    con = inputs.con
+
+    # TODO: move function_history SELECT from build_function_profile here.
+    raise NotImplementedError("join_function_history must be implemented from existing SQL CTEs.")
+
+
+# -----------------------------------------------------------------------------
+# Row assembly
+# -----------------------------------------------------------------------------
+
+
+def build_function_profile_rows(
+    inputs: FunctionProfileInputs,
+    *,
+    base_by_func: Mapping[int, FunctionBaseInfo],
+    risk_by_func: Mapping[int, FunctionRiskView],
+    cov_by_func: Mapping[int, CoverageSummary],
+    tests_by_func: Mapping[int, TestSummary],
+    graph_by_func: Mapping[int, FunctionGraphFeatures],
+    docs_by_func: Mapping[int, FunctionDocView],
+    history_by_func: Mapping[int, FunctionHistoryView],
+) -> Iterable[FunctionProfileRowModel]:
+    """Assemble :class:`FunctionProfileRowModel` values from per-concern views.
+
+    This is where the denormalized profile row is actually constructed.  The
+    goal is to keep this as “dumb” as possible: simple dictionary lookups,
+    defaulting, and field-by-field construction corresponding to the
+    `analytics.function_profile` schema.
+    """
+    for goid, base in base_by_func.items():
+        risk = risk_by_func.get(goid)
+        cov = cov_by_func.get(goid)
+        tests = tests_by_func.get(goid)
+        graph = graph_by_func.get(goid)
+        docs = docs_by_func.get(goid)
+        hist = history_by_func.get(goid)
+
+        # NOTE: this mapping should be kept 1:1 with the column list in
+        # config.schemas.tables for analytics.function_profile.
+        row: FunctionProfileRowModel = {
+            # identity
+            "function_goid_h128": goid,
+            "urn": base.urn,
+            "repo": base.repo,
+            "commit": base.commit,
+            "rel_path": base.rel_path,
+            "module": base.module,
+            "language": base.language,
+            "kind": base.kind,
+            "qualname": base.qualname,
+            "start_line": base.start_line,
+            "end_line": base.end_line,
+            "loc": base.loc,
+            "logical_loc": base.logical_loc,
+            "cyclomatic_complexity": base.cyclomatic_complexity,
+            "complexity_bucket": base.complexity_bucket,
+            "param_count": base.param_count,
+            "positional_params": base.positional_params,
+            "keyword_params": base.keyword_params,
+            "vararg": base.vararg,
+            "kwarg": base.kwarg,
+            "max_nesting_depth": base.max_nesting_depth,
+            "stmt_count": base.stmt_count,
+            "decorator_count": base.decorator_count,
+            "has_docstring": base.has_docstring,
+            "total_params": base.total_params,
+            "annotated_params": base.annotated_params,
+            "return_type": base.return_type,
+            "param_types": base.param_types,
+            "fully_typed": base.fully_typed,
+            "partial_typed": base.partial_typed,
+            "untyped": base.untyped,
+            "typedness_bucket": base.typedness_bucket,
+            "typedness_source": base.typedness_source,
+            "file_typed_ratio": base.file_typed_ratio,
+            "static_error_count": base.static_error_count,
+            "has_static_errors": base.has_static_errors,
+            # The rest of the fields (risk, coverage, graph, docs, history)
+            # should be added here as you implement the join_* helpers.
+        }
+
+        yield row
+
+
+def write_function_profile_rows(
+    gateway: StorageGateway,
+    rows: Iterable[FunctionProfileRowModel],
+) -> int:
+    """Insert rows into ``analytics.function_profile`` and return the count."""
+    rows = list(rows)
+    if not rows:
+        return 0
+
+    repo = rows[0]["repo"]
+    commit = rows[0]["commit"]
+
+    con = gateway.con
+    con.execute(
+        "DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?",
+        [repo, commit],
+    )
+
+    tuples = [function_profile_row_to_tuple(row) for row in rows]
+    con.executemany(
+        """
+        INSERT INTO analytics.function_profile (
+            function_goid_h128,
+            urn,
+            repo,
+            commit,
+            rel_path,
+            module,
+            language,
+            kind,
+            qualname,
+            start_line,
+            end_line,
+            loc,
+            logical_loc,
+            cyclomatic_complexity,
+            complexity_bucket,
+            param_count,
+            positional_params,
+            keyword_params,
+            vararg,
+            kwarg,
+            max_nesting_depth,
+            stmt_count,
+            decorator_count,
+            has_docstring,
+            total_params,
+            annotated_params,
+            return_type,
+            param_types,
+            fully_typed,
+            partial_typed,
+            untyped,
+            typedness_bucket,
+            typedness_source,
+            file_typed_ratio,
+            static_error_count,
+            has_static_errors
+            -- TODO: extend column list to include remaining analytics fields
+        )
+        VALUES (
+            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
+        )
+        """,
+        tuples,
+    )
+
+    return len(tuples)
+
+
+def build_function_profile_recipe(
+    gateway: StorageGateway,
+    cfg: ProfilesAnalyticsStepConfig,
+    *,
+    context: AnalyticsContext | None = None,
+) -> int:
+    """High-level orchestration: compute and persist function profile rows.
+
+    This is the function that :mod:`codeintel.analytics.profiles` will call.
+    """
+    inputs = compute_function_profile_inputs(gateway, cfg)
+
+    # NOTE: the order here is intentionally linear and easy to adjust.
+    base_by_func = load_function_base_info(inputs)
+    risk_by_func = join_function_risk(inputs)
+    cov_by_func = join_function_coverage(inputs)
+    tests_by_func = join_function_tests(inputs)
+    graph_by_func = join_function_graph_metrics(
+        inputs,
+        GraphMetricsStepConfig(snapshot=cfg.snapshot),
+        context=context,
+    )
+    docs_by_func = join_function_docs(inputs)
+    history_by_func = join_function_history(inputs)
+
+    rows = build_function_profile_rows(
+        inputs,
+        base_by_func=base_by_func,
+        risk_by_func=risk_by_func,
+        cov_by_func=cov_by_func,
+        tests_by_func=tests_by_func,
+        graph_by_func=graph_by_func,
+        docs_by_func=docs_by_func,
+        history_by_func=history_by_func,
+    )
+
+    return write_function_profile_rows(gateway, rows)
```

**Important:** all the `NotImplementedError` “TODO”s are *deliberate hook points* where the implementation agent should move the corresponding CTEs from your current SQL.

---

## Step 3 – Add profile row models to `storage/rows.py`

We now define a shared row contract for function_profile (and optionally test_profile later). This keeps column order stable.

### 3.1 Extend `src/codeintel/storage/rows.py`

Add near the top of the file (with other `TypedDict`s):

```diff
diff --git a/src/codeintel/storage/rows.py b/src/codeintel/storage/rows.py
index dddddddddddddddddddddddddddddddddddddddddddd..eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
--- a/src/codeintel/storage/rows.py
+++ b/src/codeintel/storage/rows.py
@@ -7,6 +7,7 @@ from __future__ import annotations
 
 from datetime import datetime
 from typing import TypedDict
+
@@ -XX,6 +YY,84 @@ __all__ = [
     "TestCoverageEdgeRow",
     "TestGraphMetricsRow",
+    "FunctionProfileRowModel",
+    "function_profile_row_to_tuple",
 ]
+
+
+class FunctionProfileRowModel(TypedDict):
+    """Row shape for ``analytics.function_profile`` inserts.
+
+    The fields and order must match the TableSchema entry in
+    :mod:`codeintel.config.schemas.tables`.
+    """
+
+    function_goid_h128: int
+    urn: str | None
+    repo: str
+    commit: str
+    rel_path: str
+    module: str | None
+    language: str | None
+    kind: str | None
+    qualname: str | None
+    start_line: int | None
+    end_line: int | None
+    loc: int
+    logical_loc: int
+    cyclomatic_complexity: int
+    complexity_bucket: str | None
+    param_count: int
+    positional_params: int
+    keyword_params: int
+    vararg: bool
+    kwarg: bool
+    max_nesting_depth: int | None
+    stmt_count: int | None
+    decorator_count: int | None
+    has_docstring: bool
+    total_params: int
+    annotated_params: int
+    return_type: object
+    param_types: object
+    fully_typed: bool
+    partial_typed: bool
+    untyped: bool
+    typedness_bucket: object
+    typedness_source: object
+    file_typed_ratio: float | None
+    static_error_count: int
+    has_static_errors: bool
+    # NOTE: as you migrate risk/coverage/tests/graph/history/docs into the
+    # recipe helpers, extend this TypedDict and the serializer below to cover
+    # the full schema.
+
+
+def function_profile_row_to_tuple(row: FunctionProfileRowModel) -> tuple[object, ...]:
+    """Serialize a FunctionProfileRowModel into INSERT column order."""
+    return (
+        row["function_goid_h128"],
+        row["urn"],
+        row["repo"],
+        row["commit"],
+        row["rel_path"],
+        row["module"],
+        row["language"],
+        row["kind"],
+        row["qualname"],
+        row["start_line"],
+        row["end_line"],
+        row["loc"],
+        row["logical_loc"],
+        row["cyclomatic_complexity"],
+        row["complexity_bucket"],
+        row["param_count"],
+        row["positional_params"],
+        row["keyword_params"],
+        row["vararg"],
+        row["kwarg"],
+        row["max_nesting_depth"],
+        row["stmt_count"],
+        row["decorator_count"],
+        row["has_docstring"],
+        row["total_params"],
+        row["annotated_params"],
+        row["return_type"],
+        row["param_types"],
+        row["fully_typed"],
+        row["partial_typed"],
+        row["untyped"],
+        row["typedness_bucket"],
+        row["typedness_source"],
+        row["file_typed_ratio"],
+        row["static_error_count"],
+        row["has_static_errors"],
+    )
```

You’ll extend this later as you migrate the rest of the columns (risk, coverage, etc.) into the recipe.

---

## Step 4 – Make `analytics/profiles.py` a thin orchestrator

Now we wire the new recipe into the existing public entrypoint **without** immediately deleting the old SQL. This gives you a clean switch-over point.

### 4.1 Update `src/codeintel/analytics/profiles.py`

At the top, add imports:

```diff
diff --git a/src/codeintel/analytics/profiles.py b/src/codeintel/analytics/profiles.py
index ffffffffffffffffffffffffffffffffffffffff..1111111111111111111111111111111111111111
--- a/src/codeintel/analytics/profiles.py
+++ b/src/codeintel/analytics/profiles.py
@@ -8,10 +8,20 @@ from __future__ import annotations
 
 import logging
 from datetime import UTC, datetime
 
 from codeintel.analytics.context import AnalyticsContext
-from codeintel.config import ProfilesAnalyticsStepConfig
+from codeintel.config import ProfilesAnalyticsStepConfig
+from codeintel.analytics.profiles.functions import (
+    build_function_profile_recipe,
+)
```

Then, in place of the current giant `build_function_profile` body, put a trampoline that *temporarily* calls either the new recipe or the legacy implementation. One good pattern is to keep the old implementation under a private name `_build_function_profile_legacy` so you can delete it once everything is ported.

Near your current `build_function_profile` definition:

```diff
@@
-def build_function_profile(
-    gateway: StorageGateway,
-    cfg: ProfilesAnalyticsStepConfig,
-    *,
-    catalog_provider: FunctionCatalogProvider | None = None,
-    context: AnalyticsContext | None = None,
-) -> None:
-    """
-    Build analytics.function_profile for the configured snapshot.
-
-    The profile denormalizes risk factors, coverage, tests, docstrings, and call
-    graph degrees into a single row per function GOID.
-    """
-    # ... ~460 LOC of logic ...
+def build_function_profile(
+    gateway: StorageGateway,
+    cfg: ProfilesAnalyticsStepConfig,
+    *,
+    catalog_provider: FunctionCatalogProvider | None = None,  # preserved for now
+    context: AnalyticsContext | None = None,
+) -> None:
+    """Public entrypoint for building ``analytics.function_profile``.
+
+    New implementations should go through
+    :func:`codeintel.analytics.profiles.functions.build_function_profile_recipe`.
+    The legacy in-place SQL implementation is retained under
+    :func:`_build_function_profile_legacy` while the refactor is in flight.
+    """
+    log = logging.getLogger(__name__)
+
+    # Temporary: delegate to the new recipe once helpers are implemented.
+    try:
+        inserted = build_function_profile_recipe(
+            gateway,
+            cfg,
+            context=context,
+        )
+        log.info(
+            "function_profile populated via recipe: %s rows for %s@%s",
+            inserted,
+            cfg.repo,
+            cfg.commit,
+        )
+        return
+    except NotImplementedError:
+        log.debug("function_profile recipe not fully implemented yet; falling back to legacy path")
+
+    _build_function_profile_legacy(
+        gateway,
+        cfg,
+        catalog_provider=catalog_provider,
+        context=context,
+    )
+
+
+def _build_function_profile_legacy(
+    gateway: StorageGateway,
+    cfg: ProfilesAnalyticsStepConfig,
+    *,
+    catalog_provider: FunctionCatalogProvider | None = None,
+    context: AnalyticsContext | None = None,
+) -> None:
+    """Previous monolithic implementation (kept for migration period).
+
+    NOTE: The body of the existing :func:`build_function_profile` should be
+    moved here verbatim by the implementation agent.  Helpers in
+    :mod:`codeintel.analytics.profiles.functions` will gradually absorb that
+    logic until this function can be deleted.
+    """
+    # TODO (migration): move the existing body of build_function_profile here
+    # unchanged, then incrementally rewrite in terms of the new recipe helpers.
+    raise NotImplementedError(
+        "_build_function_profile_legacy must be populated from the existing implementation"
+    )
```

👉 **Migration instruction for the implementation agent:**

1. Cut the entire original body of `build_function_profile` and paste it into `_build_function_profile_legacy`.
2. Confirm tests still pass (recipe path is catching `NotImplementedError` and falling back).
3. Step-by-step, peel off chunks of the SQL into the helpers in `functions.py`, replacing pieces of `_build_function_profile_legacy` with calls to those helpers, until the legacy function can be deleted.

---

## Step 5 – Test profile split (structural skeleton)

This is analogous, but we’ll keep it lighter (mostly structure + TODO hooks) so this answer doesn’t explode.

### 5.1 Create `src/codeintel/analytics/tests_profiles/__init__.py`

```diff
diff --git a/src/codeintel/analytics/tests_profiles/__init__.py b/src/codeintel/analytics/tests_profiles/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..2222222222222222222222222222222222222222
--- /dev/null
+++ b/src/codeintel/analytics/tests_profiles/__init__.py
@@ -0,0 +1,14 @@
+"""Composable recipes for test-centric analytics.
+
+This subpackage breaks :mod:`codeintel.analytics.tests.profiles` into smaller
+helpers grouped by concern:
+
+* coverage_inputs   – relationships between tests, functions, and subsystems
+* behavioral_tags   – behavioral tagging heuristics (AST/name/markers based)
+* importance        – flakiness/importance scores
+* rows              – assembly + write helpers for analytics.test_profile and
+                      analytics.behavioral_coverage.
+"""
+
+__all__ = ["coverage_inputs", "behavioral_tags", "importance", "rows"]
```

### 5.2 Create `coverage_inputs.py` skeleton

```diff
diff --git a/src/codeintel/analytics/tests_profiles/coverage_inputs.py b/src/codeintel/analytics/tests_profiles/coverage_inputs.py
new file mode 100644
index 0000000000000000000000000000000000000000..3333333333333333333333333333333333333333
--- /dev/null
+++ b/src/codeintel/analytics/tests_profiles/coverage_inputs.py
@@ -0,0 +1,66 @@
+"""Coverage-centric inputs for test profiles."""
+
+from __future__ import annotations
+
+from collections.abc import Mapping
+
+from duckdb import DuckDBPyConnection
+
+from codeintel.config import TestProfileStepConfig
+
+
+def aggregate_test_coverage_by_function(
+    con: DuckDBPyConnection,
+    cfg: TestProfileStepConfig,
+) -> Mapping[str, dict]:
+    """Aggregate coverage and execution stats per test→function edge.
+
+    Implementation note:
+        Move the parts of :mod:`codeintel.analytics.tests.profiles` that build
+        the `test_cov_fn` / `fn_cov` intermediates into this helper.  The
+        mapping values should be small dicts (or dataclasses) holding the
+        derived metrics needed to build `analytics.test_profile`.
+    """
+    raise NotImplementedError("aggregate_test_coverage_by_function must be implemented from existing logic.")
+
+
+def aggregate_test_coverage_by_subsystem(
+    con: DuckDBPyConnection,
+    cfg: TestProfileStepConfig,
+) -> Mapping[str, dict]:
+    """Aggregate coverage metrics used for behavioral_coverage.* datasets.
+
+    Implementation note:
+        Move the subsystem-level coverage aggregation from the current
+        behavioral_coverage builder into this helper.
+    """
+    raise NotImplementedError("aggregate_test_coverage_by_subsystem must be implemented from existing logic.")
```

### 5.3 Create `behavioral_tags.py` skeleton

```diff
diff --git a/src/codeintel/analytics/tests_profiles/behavioral_tags.py b/src/codeintel/analytics/tests_profiles/behavioral_tags.py
new file mode 100644
index 0000000000000000000000000000000000000000..4444444444444444444444444444444444444444
--- /dev/null
+++ b/src/codeintel/analytics/tests_profiles/behavioral_tags.py
@@ -0,0 +1,69 @@
+"""Behavioral tagging heuristics for tests."""
+
+from __future__ import annotations
+
+from collections.abc import Mapping
+
+from duckdb import DuckDBPyConnection
+
+from codeintel.config import TestProfileStepConfig
+
+
+def infer_behavior_tags(
+    con: DuckDBPyConnection,
+    cfg: TestProfileStepConfig,
+) -> Mapping[str, dict]:
+    """Infer behavioral tags and IO usage for each test.
+
+    Implementation note:
+        Move the AST/name/marker heuristics from
+        :mod:`codeintel.analytics.tests.profiles` into this helper.  The
+        returned mapping should be keyed by ``test_id`` and contain the
+        behavior tags, IO flags (uses_network/uses_db/etc.), and any other
+        fields needed to populate analytics.test_profile.
+    """
+    raise NotImplementedError("infer_behavior_tags must be implemented from existing logic.")
```

### 5.4 Create `importance.py` skeleton

```diff
diff --git a/src/codeintel/analytics/tests_profiles/importance.py b/src/codeintel/analytics/tests_profiles/importance.py
new file mode 100644
index 0000000000000000000000000000000000000000..5555555555555555555555555555555555555555
--- /dev/null
+++ b/src/codeintel/analytics/tests_profiles/importance.py
@@ -0,0 +1,60 @@
+"""Flakiness and importance scoring for tests."""
+
+from __future__ import annotations
+
+from collections.abc import Mapping
+
+from codeintel.config import TestProfileStepConfig
+
+
+def compute_importance_score(
+    cov_by_test: Mapping[str, dict],
+    behavior_by_test: Mapping[str, dict],
+    cfg: TestProfileStepConfig,
+) -> Mapping[str, dict]:
+    """Compute per-test importance and flakiness scores.
+
+    Implementation note:
+        Move the scoring logic (e.g. flakiness_score, importance_score) from
+        :mod:`codeintel.analytics.tests.profiles` into this helper.  The
+        returned mapping should be keyed by ``test_id`` and include the
+        scalar scores needed for `analytics.test_profile`.
+    """
+    raise NotImplementedError("compute_importance_score must be implemented from existing logic.")
```

### 5.5 Create `rows.py` skeleton

```diff
diff --git a/src/codeintel/analytics/tests_profiles/rows.py b/src/codeintel/analytics/tests_profiles/rows.py
new file mode 100644
index 0000000000000000000000000000000000000000..6666666666666666666666666666666666666666
--- /dev/null
+++ b/src/codeintel/analytics/tests_profiles/rows.py
@@ -0,0 +1,96 @@
+"""Row assembly helpers for test_profile and behavioral_coverage."""
+
+from __future__ import annotations
+
+from collections.abc import Iterable, Mapping
+
+from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
+from codeintel.storage.gateway import StorageGateway
+from codeintel.storage.rows import (
+    # TODO: add TestProfileRowModel / BehavioralCoverageRowModel when you define them.
+)
+
+
+def build_test_profile_rows(
+    cfg: TestProfileStepConfig,
+    cov_by_test: Mapping[str, dict],
+    behavior_by_test: Mapping[str, dict],
+    importance_by_test: Mapping[str, dict],
+) -> Iterable[dict]:
+    """Build row dicts for analytics.test_profile.
+
+    Implementation note:
+        Move the final row construction logic from
+        :mod:`codeintel.analytics.tests.profiles` into this function, mapping
+        directly to the analytics.test_profile schema.
+    """
+    raise NotImplementedError("build_test_profile_rows must be implemented from existing logic.")
+
+
+def write_test_profile_rows(
+    gateway: StorageGateway,
+    rows: Iterable[dict],
+) -> int:
+    """Insert rows into analytics.test_profile.
+
+    For the first iteration this can just use ``executemany`` with a fixed
+    column list matching the TableSchema.
+    """
+    raise NotImplementedError("write_test_profile_rows must be implemented.")
+
+
+def build_behavioral_coverage_rows(
+    cfg: BehavioralCoverageStepConfig,
+    cov_by_function: Mapping[str, dict],
+) -> Iterable[dict]:
+    """Build row dicts for analytics.behavioral_coverage."""
+    raise NotImplementedError("build_behavioral_coverage_rows must be implemented from existing logic.")
+
+
+def write_behavioral_coverage_rows(
+    gateway: StorageGateway,
+    rows: Iterable[dict],
+) -> int:
+    """Insert rows into analytics.behavioral_coverage."""
+    raise NotImplementedError("write_behavioral_coverage_rows must be implemented.")
```

### 5.6 Convert `analytics/tests/profiles.py` into an orchestrator

Finally, wire the new helpers into the existing test profiles entrypoints, again with a legacy fallback pattern.

At the top of `src/codeintel/analytics/tests/profiles.py`, add:

```diff
diff --git a/src/codeintel/analytics/tests/profiles.py b/src/codeintel/analytics/tests/profiles.py
index 7777777777777777777777777777777777777777..8888888888888888888888888888888888888888
--- a/src/codeintel/analytics/tests/profiles.py
+++ b/src/codeintel/analytics/tests/profiles.py
@@ -6,6 +6,12 @@ import logging
 from collections.abc import Callable, Iterable
 from dataclasses import dataclass, field
 from datetime import UTC, datetime
 from pathlib import Path
 from typing import cast
+
+from codeintel.analytics.tests_profiles.coverage_inputs import (
+    aggregate_test_coverage_by_function,
+    aggregate_test_coverage_by_subsystem,
+)
+from codeintel.analytics.tests_profiles import behavioral_tags, importance, rows as row_helpers
```

Then, replace the *top-level* builders (but **not** the internal helper definitions yet) with orchestrators. For example, around the existing `build_test_profile` function:

```diff
@@
-def build_test_profile(
-    gateway: StorageGateway,
-    cfg: TestProfileStepConfig,
-) -> None:
-    """Build analytics.test_profile for the configured snapshot."""
-    # ... ~1300 LOC of intertwined logic ...
+def build_test_profile(
+    gateway: StorageGateway,
+    cfg: TestProfileStepConfig,
+) -> None:
+    """Orchestrate analytics.test_profile using composable helpers.
+
+    New implementations should go through the :mod:`codeintel.analytics.tests_profiles`
+    helpers.  The legacy in-place implementation is retained under
+    :func:`_build_test_profile_legacy` while the refactor is in flight.
+    """
+    log = logging.getLogger(__name__)
+    con = gateway.con
+
+    try:
+        cov_by_test = aggregate_test_coverage_by_function(con, cfg)
+        behavior_by_test = behavioral_tags.infer_behavior_tags(con, cfg)
+        importance_by_test = importance.compute_importance_score(
+            cov_by_test,
+            behavior_by_test,
+            cfg,
+        )
+        profile_rows = row_helpers.build_test_profile_rows(
+            cfg,
+            cov_by_test=cov_by_test,
+            behavior_by_test=behavior_by_test,
+            importance_by_test=importance_by_test,
+        )
+        inserted = row_helpers.write_test_profile_rows(gateway, profile_rows)
+        log.info(
+            "test_profile populated via recipe: %s rows for %s@%s",
+            inserted,
+            cfg.repo,
+            cfg.commit,
+        )
+        return
+    except NotImplementedError:
+        log.debug("test_profile recipe not fully implemented; falling back to legacy path")
+
+    _build_test_profile_legacy(gateway, cfg)
+
+
+def _build_test_profile_legacy(
+    gateway: StorageGateway,
+    cfg: TestProfileStepConfig,
+) -> None:
+    """Previous monolithic implementation (kept for migration period).
+
+    NOTE: The existing body of :func:`build_test_profile` should be moved
+    here unchanged by the implementation agent, and then gradually rewritten
+    in terms of the helpers in :mod:`codeintel.analytics.tests_profiles`.
+    """
+    raise NotImplementedError("_build_test_profile_legacy must be populated from existing implementation.")
```

Do the same pattern for the function that currently builds `analytics.behavioral_coverage` — turn it into:

```python
def build_behavioral_coverage(gateway: StorageGateway, cfg: BehavioralCoverageStepConfig) -> None:
    # Try new recipe -> fallback to _build_behavioral_coverage_legacy
```

---

## How to use this with your implementation agent

You can now give an agent instructions like:

> 1. Move the body of `build_function_profile` into `_build_function_profile_legacy`.
> 2. For each helper in `analytics/profiles/functions.py` that raises `NotImplementedError`, locate the corresponding CTE/SQL chunk in `_build_function_profile_legacy` and port it into that helper.
> 3. Extend `FunctionProfileRowModel` and `function_profile_row_to_tuple` to cover all columns in `analytics.function_profile` and wire the helper outputs into `build_function_profile_rows`.
> 4. Once tests pass without hitting the legacy fallback, delete `_build_function_profile_legacy`.

And similarly for `tests.profiles`.

This keeps the migration **mechanically clear**, preserves behavior via the legacy path, and lands the structure you wanted:

* small, composable recipe units
* explicit value objects
* clear row models shared with storage
* orchestrator functions that are ~40 lines and easy for you + LLMs to reason about.

If you’d like, next I can focus *just* on Step 2.2 (e.g. “here is the exact SQL you should move into `load_function_base_info` and how to map columns → dataclass fields”) for one slice, and you (or the agent) can copy that pattern across the rest.


# deep dive on step 2.2 - exact SQL you should move into `load_function_base_info` and how to map columns → dataclass fields #

Nice, let’s zoom in on that one slice and actually wire it up. I’ll:

1. Nail down the **`FunctionBaseInfo` dataclass**.
2. Give you a concrete implementation of **`load_function_base_info`** that’s directly extracted from your existing `build_function_profile` SQL.
3. Show the **column→field mapping** explicitly so it’s easy to replicate for other helpers.

Everything below is aligned to what’s currently in:

* `analytics/analytics/profiles.py` → `build_function_profile`
* `storage/storage/gateway.py` (for `DuckDBConnection`)

---

## 1. `FunctionBaseInfo` dataclass (in `analytics/profiles/types.py`)

This matches the **first 36 columns** of the big `SELECT` in your current `build_function_profile`, up through `rf.has_static_errors`.

```python
# analytics/profiles/types.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from codeintel.storage.gateway import DuckDBConnection


@dataclass(frozen=True)
class FunctionProfileInputs:
    """
    Handle object for function profile computations.

    This does *not* eagerly materialize large relations; it just carries the
    connection + snapshot identity for queries.
    """

    con: DuckDBConnection
    repo: str
    commit: str
    created_at: datetime


@dataclass(frozen=True)
class FunctionBaseInfo:
    """
    Static / non-aggregated info for a function, used as the base of the profile.

    This is essentially the subset of the final function_profile SELECT that
    comes from rf/fm/ft/core.modules before any coverage/test/graph/doc/history
    joins kick in.
    """

    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    module: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    complexity_bucket: str | None
    param_count: int
    positional_params: int
    keyword_params: int
    vararg: bool
    kwarg: bool
    max_nesting_depth: int | None
    stmt_count: int | None
    decorator_count: int | None
    has_docstring: bool
    total_params: int
    annotated_params: int
    return_type: str | None
    param_types: str | None
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str | None
    typedness_source: str | None
    file_typed_ratio: float | None
    static_error_count: int
    has_static_errors: bool
```

If you already created `FunctionBaseInfo` from my earlier message, you can tweak it to exactly match this signature.

---

## 2. `load_function_base_info` implementation (in `analytics/profiles/functions.py`)

This is the “Step 2.2” slice: pulling the **static base attributes** out of your big SQL and into a helper.

### 2.1 Function implementation

Drop this into `analytics/profiles/functions.py`, replacing the `NotImplementedError` stub for `load_function_base_info`:

```python
# analytics/profiles/functions.py

from __future__ import annotations

from collections.abc import Mapping

from codeintel.storage.gateway import StorageGateway

from .types import FunctionBaseInfo, FunctionProfileInputs


def load_function_base_info(
    inputs: FunctionProfileInputs,
) -> Mapping[int, FunctionBaseInfo]:
    """
    Load static per-function info used as the base of the profile.

    This is extracted from the *front* of the final SELECT inside the existing
    build_function_profile:

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
            fm.keyword_only_params AS keyword_params,
            fm.has_varargs AS vararg,
            fm.has_varkw AS kwarg,
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
            ...   -- coverage/test/graph/doc/history start here

    We only select the first 36 columns (through has_static_errors) here.
    """
    con = inputs.con

    rows = con.execute(
        """
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
            fm.keyword_only_params AS keyword_params,
            fm.has_varargs AS vararg,
            fm.has_varkw AS kwarg,
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
            rf.has_static_errors
        FROM analytics.goid_risk_factors AS rf
        LEFT JOIN analytics.function_metrics AS fm
          ON rf.function_goid_h128 = fm.function_goid_h128
         AND rf.repo = fm.repo
         AND rf.commit = fm.commit
        LEFT JOIN analytics.function_types AS ft
          ON rf.function_goid_h128 = ft.function_goid_h128
         AND rf.repo = ft.repo
         AND rf.commit = ft.commit
        LEFT JOIN core.modules AS m
          ON m.path = rf.rel_path
         AND (m.repo IS NULL OR m.repo = rf.repo)
         AND (m.commit IS NULL OR m.commit = rf.commit)
        WHERE rf.repo = ? AND rf.commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()

    result: dict[int, FunctionBaseInfo] = {}

    for row in rows:
        (
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
        ) = row

        goid_int = int(function_goid_h128)

        result[goid_int] = FunctionBaseInfo(
            function_goid_h128=goid_int,
            urn=str(urn) if urn is not None else None,
            repo=str(repo),
            commit=str(commit),
            rel_path=str(rel_path),
            module=str(module) if module is not None else None,
            language=str(language) if language is not None else None,
            kind=str(kind) if kind is not None else None,
            qualname=str(qualname) if qualname is not None else None,
            start_line=int(start_line) if start_line is not None else None,
            end_line=int(end_line) if end_line is not None else None,
            loc=int(loc or 0),
            logical_loc=int(logical_loc or 0),
            cyclomatic_complexity=int(cyclomatic_complexity or 0),
            complexity_bucket=str(complexity_bucket) if complexity_bucket is not None else None,
            param_count=int(param_count or 0),
            positional_params=int(positional_params or 0),
            keyword_params=int(keyword_params or 0),
            vararg=bool(vararg),
            kwarg=bool(kwarg),
            max_nesting_depth=int(max_nesting_depth) if max_nesting_depth is not None else None,
            stmt_count=int(stmt_count) if stmt_count is not None else None,
            decorator_count=int(decorator_count) if decorator_count is not None else None,
            has_docstring=bool(has_docstring),
            total_params=int(total_params or 0),
            annotated_params=int(annotated_params or 0),
            return_type=str(return_type) if return_type is not None else None,
            param_types=str(param_types) if param_types is not None else None,
            fully_typed=bool(fully_typed),
            partial_typed=bool(partial_typed),
            untyped=bool(untyped),
            typedness_bucket=str(typedness_bucket) if typedness_bucket is not None else None,
            typedness_source=str(typedness_source) if typedness_source is not None else None,
            file_typed_ratio=float(file_typed_ratio) if file_typed_ratio is not None else None,
            static_error_count=int(static_error_count or 0),
            has_static_errors=bool(has_static_errors),
        )

    return result
```

### 2.2 Column → dataclass field mapping

For clarity (and to help you / an agent replicate the pattern for other helpers), here’s the mapping spelled out:

| SELECT expression                          | Dataclass field         | Notes                     |
| ------------------------------------------ | ----------------------- | ------------------------- |
| `rf.function_goid_h128`                    | `function_goid_h128`    | `int()`, used as dict key |
| `rf.urn`                                   | `urn`                   | `Optional[str]`           |
| `rf.repo`                                  | `repo`                  | `str`                     |
| `rf.commit`                                | `commit`                | `str`                     |
| `rf.rel_path`                              | `rel_path`              | `str`                     |
| `m.module`                                 | `module`                | `Optional[str]`           |
| `rf.language`                              | `language`              | `Optional[str]`           |
| `rf.kind`                                  | `kind`                  | `Optional[str]`           |
| `rf.qualname`                              | `qualname`              | `Optional[str]`           |
| `fm.start_line`                            | `start_line`            | `Optional[int]`           |
| `fm.end_line`                              | `end_line`              | `Optional[int]`           |
| `rf.loc`                                   | `loc`                   | `int`, default 0          |
| `rf.logical_loc`                           | `logical_loc`           | `int`, default 0          |
| `rf.cyclomatic_complexity`                 | `cyclomatic_complexity` | `int`, default 0          |
| `rf.complexity_bucket`                     | `complexity_bucket`     | `Optional[str]`           |
| `fm.param_count`                           | `param_count`           | `int`, default 0          |
| `fm.positional_params`                     | `positional_params`     | `int`, default 0          |
| `fm.keyword_only_params AS keyword_params` | `keyword_params`        | `int`, default 0          |
| `fm.has_varargs AS vararg`                 | `vararg`                | `bool`                    |
| `fm.has_varkw AS kwarg`                    | `kwarg`                 | `bool`                    |
| `fm.max_nesting_depth`                     | `max_nesting_depth`     | `Optional[int]`           |
| `fm.stmt_count`                            | `stmt_count`            | `Optional[int]`           |
| `fm.decorator_count`                       | `decorator_count`       | `Optional[int]`           |
| `fm.has_docstring`                         | `has_docstring`         | `bool`                    |
| `ft.total_params`                          | `total_params`          | `int`, default 0          |
| `ft.annotated_params`                      | `annotated_params`      | `int`, default 0          |
| `ft.return_type`                           | `return_type`           | `Optional[str]`           |
| `ft.param_types`                           | `param_types`           | `Optional[str]`           |
| `ft.fully_typed`                           | `fully_typed`           | `bool`                    |
| `ft.partial_typed`                         | `partial_typed`         | `bool`                    |
| `ft.untyped`                               | `untyped`               | `bool`                    |
| `rf.typedness_bucket`                      | `typedness_bucket`      | `Optional[str]`           |
| `rf.typedness_source`                      | `typedness_source`      | `Optional[str]`           |
| `rf.file_typed_ratio`                      | `file_typed_ratio`      | `Optional[float]`         |
| `rf.static_error_count`                    | `static_error_count`    | `int`, default 0          |
| `rf.has_static_errors`                     | `has_static_errors`     | `bool`                    |

Everything *after* `rf.has_static_errors` in the original SELECT is what you’re moving into other helpers:

* `rf.executable_lines`, `rf.covered_lines`, `rf.coverage_ratio`, `rf.tested`, `cf.untested_reason`, `t_stats.*` → **coverage & tests** helpers.
* `fh.*` → **history** helper.
* `cg.*` → **graph** helper.
* `rf.risk_score`, `rf.risk_level`, `risk_component_*` → **risk** helper.
* `fe.*`, `fc.*`, `fr.*`, `doc.*` → **effects/contracts/role/docs** helpers.

---

## How to use this as a pattern

Once this is in place, the remaining helpers are basically:

* Copy the relevant slice of the big SQL **into its own query**.
* Map columns to a small dataclass (like we just did).
* Use that mapping in `build_function_profile_rows`.




