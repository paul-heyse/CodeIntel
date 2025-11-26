Here’s a concrete implementation plan for the **Higher‑level architecture & subsystems** features, wired into the existing CodeIntel pipeline and code layout you shared. I’ll do:

1. **Graph metrics** (function/module‑level) → `analytics.graph_metrics_*`
2. **Subsystem clustering** → `analytics.subsystems*`

…with explicit notes on where to touch `analytics/`, `config/`, `orchestration/`, `storage/`, and `docs_export/`.

---

## 0. Context: what we’re building on

You already have:

* **Module registry**: `core.modules(module, path, repo, commit, tags, owners)`
* **Import graph**: `graph.import_graph_edges(src_module, dst_module, src_fan_out, dst_fan_in, cycle_group)`
* **Call graph**: `graph.call_graph_nodes`, `graph.call_graph_edges(caller_goid_h128, callee_goid_h128, ...)`
* **Symbol uses**: `graph.symbol_use_edges(symbol, def_path, use_path, same_file, same_module)`
* **Per‑function analytics**: `analytics.function_metrics`, `analytics.function_types`, `analytics.coverage_functions`, `analytics.goid_risk_factors`
* **Tags / ownership**: tags & owners already attached to modules; plus `analytics.tags_index` and `tags_index.yaml`.

And a pipeline shape:

* Ingestion → Graphs → Analytics → Risk → Export.

We’ll add:

* **Graph metrics tables**: per function and per module.
* **Subsystems tables**: inferred architectural clusters + membership.
* **Pipeline steps** to compute/populate them.
* **Exports** & optional `docs.*` views to make them agent‑friendly.

---

## 1. Graph metrics (`graph_metrics.*`)

### 1.1 Table design

Create **two analytics tables**:

1. `analytics.graph_metrics_functions`
2. `analytics.graph_metrics_modules`

#### 1.1.1 `analytics.graph_metrics_functions`

Purpose: graph‑theoretic metrics per function GOID from the **call graph**.

**Schema (add to `config/schemas/tables.py`):**

```python
"analytics.graph_metrics_functions": TableSchema(
    schema="analytics",
    name="graph_metrics_functions",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        # Degree / fan-in/out on call graph
        Column("call_fan_in", "INTEGER", nullable=False),
        Column("call_fan_out", "INTEGER", nullable=False),
        Column("call_in_degree", "INTEGER", nullable=False),
        Column("call_out_degree", "INTEGER", nullable=False),

        # Centrality (may be approximate)
        Column("call_pagerank", "DOUBLE"),
        Column("call_betweenness", "DOUBLE"),
        Column("call_closeness", "DOUBLE"),

        # Structure
        Column("call_cycle_member", "BOOLEAN", nullable=False),
        Column("call_cycle_id", "INTEGER"),
        Column("call_layer", "INTEGER"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Graph-theoretic metrics for functions computed from the call graph",
)
```

Definitions (how we’ll compute them):

* **fan_in**: number of **distinct callers** (unique `caller_goid_h128`) for this function.
* **fan_out**: number of **distinct callees** (unique `callee_goid_h128`) this function calls (excluding `NULL`).
* **in_degree / out_degree**: raw edge counts (multiple callsites to same callee count multiple times).
* **pagerank**: PageRank over the directed call graph.
* **betweenness / closeness**: standard directed centralities (approximate for large graphs).
* **cycle_member**: true if the function is in an SCC of size > 1.
* **cycle_id**: integer ID of the SCC.
* **layer**: inferred layer from condensation DAG (SCCs as nodes, longest‑path level from “entry” SCCs).

These will be keyed by **GOID** and repo/commit, consistent with other analytics tables.

#### 1.1.2 `analytics.graph_metrics_modules`

Purpose: graph metrics over **modules**, combining import graph + symbol‑use coupling.

Schema:

```python
"analytics.graph_metrics_modules": TableSchema(
    schema="analytics",
    name="graph_metrics_modules",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=False),

        # Import-graph degree / fan-in/out
        Column("import_fan_in", "INTEGER", nullable=False),
        Column("import_fan_out", "INTEGER", nullable=False),
        Column("import_in_degree", "INTEGER", nullable=False),
        Column("import_out_degree", "INTEGER", nullable=False),

        # Import-graph centralities
        Column("import_pagerank", "DOUBLE"),
        Column("import_betweenness", "DOUBLE"),
        Column("import_closeness", "DOUBLE"),

        # Import-graph SCC/layer
        Column("import_cycle_member", "BOOLEAN", nullable=False),
        Column("import_cycle_id", "INTEGER"),
        Column("import_layer", "INTEGER"),

        # Symbol-use coupling between modules
        Column("symbol_fan_in", "INTEGER", nullable=False),
        Column("symbol_fan_out", "INTEGER", nullable=False),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "module"),
    description="Per-module graph metrics from import and symbol-use graphs",
)
```

Notes:

* **import_* metrics** derive from `graph.import_graph_edges`. You already compute per‑edge fan‑in/fan‑out and `cycle_group`; we’ll aggregate to per‑module and add centralities + layers.
* **symbol_* metrics** come from a module‑collapsing of `graph.symbol_use_edges` joined to `core.modules` on path → module.

### 1.2 Analytics implementation (`analytics/graph_metrics.py`)

Create a new module: **`analytics/graph_metrics.py`**.

Imports will mirror `analytics/coverage_analytics.py` / `analytics/tests_analytics.py`: use `duckdb`, `ensure_schema`, and maybe `networkx` for graph algorithms.

#### 1.2.1 Config class

Add to `config/models.py`:

```python
@dataclass(frozen=True)
class GraphMetricsConfig:
    """Configuration for graph metrics analytics."""

    repo: str
    commit: str

    # Tunables for expensive centrality computations
    max_betweenness_sample: int | None = 200  # None = exact, else sample k nodes

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> Self:
        return cls(repo=repo, commit=commit)
```

Match style of `CoverageAnalyticsConfig` / `FunctionAnalyticsConfig`.

#### 1.2.2 Compute function graph metrics

In `analytics/graph_metrics.py`:

```python
import logging
from datetime import UTC, datetime

import duckdb
import networkx as nx  # new dependency

from codeintel.config.models import GraphMetricsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)
```

Algorithm:

1. **Ensure table**:

   ```python
   ensure_schema(con, "analytics.graph_metrics_functions")
   con.execute(
       "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
       [cfg.repo, cfg.commit],
   )
   ```

2. **Load call graph edges** (filter to resolved callees):

   ```sql
   SELECT caller_goid_h128, callee_goid_h128
   FROM graph.call_graph_edges
   WHERE callee_goid_h128 IS NOT NULL
   ```

3. **Build directed graph** `G` in Python (`networkx.DiGraph`):

   * Add nodes for every distinct caller/callee.
   * Add one edge per `(caller, callee)` (optionally dedup by pair).

4. **Compute degrees / fan‑in/out**:

   * `fan_in = len(predecessors(node))`
   * `fan_out = len(successors(node))`
   * `in_degree = G.in_degree(node)`
   * `out_degree = G.out_degree(node)`

5. **Compute centralities**:

   * **PageRank**: `nx.pagerank(G, alpha=0.85)`
   * **Closeness**: `nx.closeness_centrality(G)` (directed)
   * **Betweenness**:

     * If node count > threshold, use sampled version:
       `nx.betweenness_centrality(G, k=cfg.max_betweenness_sample, seed=0)`
     * Else full `nx.betweenness_centrality(G)`.

6. **Compute SCCs & layers**:

   * `sccs = list(nx.strongly_connected_components(G))`
   * Assign `call_cycle_id` per component (e.g., enumerated integer).
   * `call_cycle_member = len(component) > 1`.
   * Build condensation graph `C = nx.condensation(G, sccs)` (DAG).
   * Perform a longest‑path‐from‑sources style layering on `C` and propagate to original nodes as `call_layer`.

7. **Attach repo/commit**:

   * Simplest: look up repo/commit for each `function_goid_h128` via `analytics.goid_risk_factors` or `analytics.function_metrics`. Both contain `function_goid_h128`, `repo`, `commit`.

   * Query:

     ```sql
     SELECT function_goid_h128, repo, commit
     FROM analytics.function_metrics
     ```

   * Build `mapping: goid -> (repo, commit)`.

   * (In current design DB is per repo/commit, but this keeps the table future‑proof.)

8. **Insert rows**:

   Use `executemany` with a prepared `INSERT` over `analytics.graph_metrics_functions` (or bulk insert with DuckDB parameter arrays, matching style in `coverage_analytics`).

9. **Log row count** for this repo/commit.

#### 1.2.3 Compute module graph metrics

Still in `analytics/graph_metrics.py`:

1. **Ensure schema & clear rows**:

   ```python
   ensure_schema(con, "analytics.graph_metrics_modules")
   con.execute(
       "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
       [cfg.repo, cfg.commit],
   )
   ```

2. **Load import edges**:

   ```sql
   SELECT src_module, dst_module
   FROM graph.import_graph_edges
   ```

   Build directed graph `G_import` over modules.

3. **Load symbol‑use edges collapsed to modules**:

   ```sql
   SELECT
       su.def_path,
       su.use_path,
       m_def.module AS def_module,
       m_use.module AS use_module
   FROM graph.symbol_use_edges su
   LEFT JOIN core.modules m_def ON m_def.path = su.def_path
   LEFT JOIN core.modules m_use ON m_use.path = su.use_path
   WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
   ```

   Build `G_symbol` as directed edges from **using module → defining module** (A uses B).

4. **Compute import metrics**:

   * As with functions:

     * `import_fan_in = len(predecessors)` in `G_import`.
     * `import_fan_out = len(successors)`.
     * `import_in_degree` / `import_out_degree` from edge counts.
   * **Centralities** from `G_import`:

     * PageRank, betweenness (sampled for large graphs), closeness.
   * **Cycles**:

     * Either reuse `cycle_group` from `graph.import_graph_edges` aggregated per module (`MAX(cycle_group)`) or compute SCCs via networkx. You already produce `cycle_group` via Tarjan SCC in the builder; we can just aggregate that to `import_cycle_id`.
   * **Layering**:

     * Build condensation DAG (SCCs) as for functions and compute `import_layer` per module.

5. **Compute symbol‑use metrics**:

   * On `G_symbol`:

     * `symbol_fan_out = len(successors(module))`
     * `symbol_fan_in = len(predecessors(module))`

6. **Attach repo/commit**:

   * From `core.modules`:

     ```sql
     SELECT module, repo, commit FROM core.modules
     ```

   * Build `module -> (repo, commit)` mapping.

7. **Insert rows** into `analytics.graph_metrics_modules` with `created_at=NOW()`.

### 1.3 Wiring into pipeline

#### 1.3.1 Orchestration step

Extend `orchestration/steps.py` with a new **analytics step** (similar to `FunctionAnalyticsStep`, `CoverageAnalyticsStep`).

```python
from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.config.models import GraphMetricsConfig
```

Add:

```python
@dataclass
class GraphMetricsStep:
    """Compute graph-theoretic metrics over call and import graphs."""

    name: str = "graph_metrics"
    # Needs call graph and import graph; symbol_uses is optional but preferred.
    deps: Sequence[str] = ("callgraph", "import_graph", "symbol_uses")

    def run(self, ctx: PipelineContext, con: StorageGateway) -> None:
        _log_step(self.name)
        cfg = GraphMetricsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_graph_metrics(con, cfg)
```

Place this in the analytics section, before `RiskFactorsStep` or after – graph metrics don’t depend on risk.

#### 1.3.2 Prefect task + flow

In `orchestration/prefect_flow.py`, add a task:

```python
@task(name="graph_metrics", retries=1, retry_delay_seconds=2)
def t_graph_metrics(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = GraphMetricsConfig.from_paths(repo=repo, commit=commit)
    compute_graph_metrics(con, cfg)
    con.close()
```

Wire it into the `steps` sequence after `callgraph` and `import_graph` (and after `symbol_uses` if you want):

```python
("graph_metrics", lambda: _run_task("graph_metrics", t_graph_metrics, run_logger, ctx.repo, ctx.commit, ctx.db_path)),
```

#### 1.3.3 Export mapping

In **`docs_export/export_parquet.py`** and **`export_jsonl.py`**, extend the dataset maps:

```python
PARQUET_DATASETS.update({
    "analytics.graph_metrics_functions": "graph_metrics_functions.parquet",
    "analytics.graph_metrics_modules": "graph_metrics_modules.parquet",
})
```

and

```python
JSONL_DATASETS.update({
    "analytics.graph_metrics_functions": "graph_metrics_functions.jsonl",
    "analytics.graph_metrics_modules": "graph_metrics_modules.jsonl",
})
```

So your Document Output will now include `graph_metrics_functions.jsonl` and `graph_metrics_modules.jsonl`.

#### 1.3.4 Optional docs views

In `storage/views.py`, you can expose enriched views for agents:

* **`docs.v_function_architecture`**: join `analytics.graph_metrics_functions` with `analytics.goid_risk_factors` + `core.docstrings` to get “function + architecture metrics + risk” in one row, similar to `docs.v_function_summary`.
* **`docs.v_module_architecture`**: join `analytics.graph_metrics_modules` with `core.modules`, aggregated risk, hotspots, typedness.

---

## 2. Subsystems (`subsystems.*`)

Now we layer **architectural clusters** on top of these metrics.

### 2.1 Data model: subsystems + membership

We’ll create:

1. `analytics.subsystems` – one row per inferred subsystem/bounded context.
2. `analytics.subsystem_modules` – mapping from subsystem → modules.

#### 2.1.1 `analytics.subsystems`

Add to `config/schemas/tables.py`:

```python
"analytics.subsystems": TableSchema(
    schema="analytics",
    name="subsystems",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("subsystem_id", "VARCHAR", nullable=False),

        # Identity
        Column("name", "VARCHAR", nullable=False),
        Column("description", "VARCHAR"),

        # Membership summary
        Column("module_count", "INTEGER", nullable=False),
        Column("modules_json", "JSON", nullable=False),
        Column("entrypoints_json", "JSON"),

        # Graph stats at subsystem level
        Column("internal_edge_count", "INTEGER", nullable=False),
        Column("external_edge_count", "INTEGER", nullable=False),
        Column("fan_in", "INTEGER", nullable=False),   # # of other subsystems depending on this
        Column("fan_out", "INTEGER", nullable=False),  # # of other subsystems this depends on

        # Risk aggregation (from goid_risk_factors)
        Column("function_count", "INTEGER", nullable=False),
        Column("avg_risk_score", "DOUBLE"),
        Column("max_risk_score", "DOUBLE"),
        Column("high_risk_function_count", "INTEGER", nullable=False),
        Column("risk_level", "VARCHAR"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "subsystem_id"),
    description="Inferred architectural subsystems (bounded contexts) and their summary stats",
)
```

Notes:

* `modules_json`: JSON array of module strings (`["CodeIntel.app.routes.catalog_read", ...]`).
* `entrypoints_json`: JSON array of objects representing HTTP/CLI/cron entrypoints when you later build entrypoint analytics (for now, may be empty or based on tags).

#### 2.1.2 `analytics.subsystem_modules`

Schema:

```python
"analytics.subsystem_modules": TableSchema(
    schema="analytics",
    name="subsystem_modules",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("subsystem_id", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=False),
        Column("role", "VARCHAR"),  # e.g. 'api', 'infra', 'domain', derived from tags
    ],
    primary_key=("repo", "commit", "subsystem_id", "module"),
    description="Mapping from subsystems to member modules with optional roles",
)
```

This normalized table makes it easy to join subsystems to modules, functions, tests, config, etc.

### 2.2 Clustering algorithm

Implement in a new module **`analytics/subsystems.py`**.

#### 2.2.1 Config

In `config/models.py`:

```python
@dataclass(frozen=True)
class SubsystemsConfig:
    """Configuration for subsystem inference."""

    repo: str
    commit: str
    min_modules: int = 3           # drop tiny clusters
    max_subsystems: int | None = None
    import_weight: float = 1.0
    symbol_weight: float = 0.5
    config_weight: float = 0.3

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> Self:
        return cls(repo=repo, commit=commit)
```

#### 2.2.2 Build the module‑level similarity graph

Inputs:

* `graph.import_graph_edges(src_module, dst_module)`
* `graph.symbol_use_edges(def_path, use_path)` + `core.modules(path, module)` joined to get module↔module used.
* Optionally `analytics.config_values` to connect modules reading the same config keys.
* `core.modules.tags` / `analytics.tags_index` for tags/roles.

Algorithm:

1. **Import edges (structural)**:

   * Treat imports as undirected coupling for clustering.
   * Weight for an undirected edge `{A,B}`: `w_import * (#imports between A and B)`.
   * Build adjacency map `W[A][B] += w_import`.

2. **Symbol uses (semantic coupling)**:

   * After collapsing def/use paths to modules:

     * For each symbol use, connect `{using_module, defining_module}`.
     * Weight: `w_symbol` per use (maybe normalized).

3. **Config co‑usage (shared configuration)**:

   * For each config key row: `analytics.config_values(key, reference_modules)` (JSON array).
   * For each pair of modules in `reference_modules`, connect them with `w_config / (n-1)`.

4. Result: an **undirected weighted graph** over modules, representing how “tightly connected” they are across imports, symbol references, and config coupling.

Use networkx `Graph` again, but here the graph is undirected and weighted.

#### 2.2.3 Community detection (subsystems)

We want to find clusters that align with “bounded contexts”.

A simple, dependency‑light approach:

* Implement **label propagation** manually (no extra libs beyond networkx):

  1. Initialize `label[module] = module` (unique labels).
  2. Optionally seed labels from tags: modules tagged `"api"`, `"infra"`, `"ml"`, etc. get those tag names as initial labels and are **frozen** (don’t change).
  3. Shuffle node order each iteration.
  4. For each module:

     * Look at neighbors’ labels; pick the label with highest **total edge weight** into this node.
     * Update if the label changes.
  5. Repeat until convergence or `max_iters` (e.g. 20).

* After convergence, group modules by label → clusters.

Post‑processing:

* Drop clusters with `< cfg.min_modules` (reassign their modules to the best neighboring cluster or a “misc” cluster).
* If `max_subsystems` is set and we have more clusters than that, merge the smallest clusters into nearest larger clusters based on inter‑cluster edge weights.

Each final cluster becomes a **subsystem**.

#### 2.2.4 Subsystem IDs, names, descriptions

For each subsystem (cluster):

1. **Subsystem ID**:

   * Make it *stable* by hashing the sorted module list and repo:

     ```python
     import hashlib
     raw = f"{cfg.repo}:{','.join(sorted(modules))}"
     subsystem_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
     ```

2. **Name**:

   * Heuristics:

     * Take most common **module prefix** (e.g. `CodeIntel.app.routes`, `CodeIntel.analytics`) across member modules.
     * Fall back to `subsys_<short_hash>`.
   * You can also derive from dominant tags: e.g. `api_routes`, `config_ingest`.

3. **Description**:

   * For now, deterministic text:

     * `"Subsystem for {prefix}: {module_count} modules including {example_module1}, {example_module2}."`
   * Optionally, behind a feature flag, call an LLM in a later pass to generate a richer summary, but keep that separate from the core pipeline to avoid external dependencies.

#### 2.2.5 Subsystem graph stats

Using the **module import graph**:

1. For each pair of subsystems `(S, T)`:

   * `internal_edge_count(S)` = number of import edges where **src and dst both in S**.
   * `external_edge_count(S)` = number of edges where **exactly one endpoint is in S**.
   * `fan_in(S)` = distinct other subsystems `T != S` from which there is an edge `T→S`.
   * `fan_out(S)` = distinct subsystems `T` for which there is an edge `S→T`.

These are computed by aggregating import edges with module→subsystem mapping.

#### 2.2.6 Risk aggregation (per subsystem)

Use `analytics.goid_risk_factors` and `analytics.function_metrics` + `core.modules` to map functions → modules → subsystem.

SQL shape:

```sql
WITH function_modules AS (
    SELECT
        fm.function_goid_h128,
        m.module
    FROM analytics.function_metrics fm
    JOIN core.modules m
      ON m.path = fm.rel_path
),
function_subsystems AS (
    SELECT
        fm.function_goid_h128,
        sm.subsystem_id
    FROM function_modules fm
    JOIN analytics.subsystem_modules sm
      ON sm.module = fm.module
)
SELECT
    fs.subsystem_id,
    COUNT(*)                                  AS function_count,
    AVG(rf.risk_score)                        AS avg_risk_score,
    MAX(rf.risk_score)                        AS max_risk_score,
    SUM(CASE WHEN rf.risk_level = 'high'
             THEN 1 ELSE 0 END)              AS high_risk_function_count,
    CASE
      WHEN SUM(CASE WHEN rf.risk_level='high' THEN 1 ELSE 0 END) > 0 THEN 'high'
      WHEN SUM(CASE WHEN rf.risk_level='medium' THEN 1 ELSE 0 END) > 0 THEN 'medium'
      ELSE 'low'
    END                                       AS risk_level
FROM function_subsystems fs
JOIN analytics.goid_risk_factors rf
  ON rf.function_goid_h128 = fs.function_goid_h128
GROUP BY fs.subsystem_id;
```

Use this to populate the risk columns in `analytics.subsystems`.

#### 2.2.7 Entrypoints

Column `entrypoints_json` is designed to eventually hold HTTP/CLI/cron entrypoints per subsystem, but you can seed it *now* using tags:

* For modules within a subsystem:

  * If `modules.tags` or `tags_index` includes `"api"`, generate an entry like:

    ```json
    {"kind": "tag", "tag": "api", "module": "CodeIntel.app.routes.catalog_read"}
    ```

* Later, when you implement explicit `entrypoints.*` datasets, you can fill in richer descriptors (HTTP method/path, CLI command, schedule, etc.).

### 2.3 Implementation: `analytics/subsystems.py`

Structure:

```python
"""
Infer higher-level subsystems (bounded contexts) from module graphs and tags.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
import hashlib

import duckdb
import networkx as nx

from codeintel.config.models import SubsystemsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)
```

Core function:

```python
def build_subsystems(con: StorageGateway, cfg: SubsystemsConfig) -> None:
    """
    Populate analytics.subsystems and analytics.subsystem_modules for a repo/commit.
    """
    ensure_schema(con, "analytics.subsystems")
    ensure_schema(con, "analytics.subsystem_modules")

    con.execute(
        "DELETE FROM analytics.subsystems WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    # 1) Load modules + import/symbol/config relationships
    # 2) Build weighted module graph
    # 3) Run label-propagation clustering
    # 4) Post-process clusters (min_modules, max_subsystems)
    # 5) Compute subsystem stats & risk aggregation
    # 6) Insert into analytics.subsystems and analytics.subsystem_modules
```

Internals:

* Functions like `_load_module_graph(con, cfg)`, `_run_label_propagation(G, tags_by_module, cfg)`, `_compute_subsystem_stats(...)`.
* Use `json.dumps(modules)` and `json.dumps(entrypoints)` for JSON columns.

### 2.4 Wiring subsystems into pipeline

#### 2.4.1 Orchestration step

In `orchestration/steps.py`:

```python
from codeintel.analytics.subsystems import build_subsystems
from codeintel.config.models import SubsystemsConfig
```

Add:

```python
@dataclass
class SubsystemsStep:
    """Infer higher-level subsystems from module graph and analytics."""

    name: str = "subsystems"
    # Needs import graph + symbol uses + modules + risk factors
    deps: Sequence[str] = ("import_graph", "symbol_uses", "risk_factors")

    def run(self, ctx: PipelineContext, con: StorageGateway) -> None:
        _log_step(self.name)
        cfg = SubsystemsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        build_subsystems(con, cfg)
```

You *can* also add `graph_metrics` to deps if you want to eventually incorporate graph metrics into subsystem naming/weighting, but not strictly required for the first version.

#### 2.4.2 Prefect task + flow

In `orchestration/prefect_flow.py`:

```python
@task(name="subsystems", retries=1, retry_delay_seconds=2)
def t_subsystems(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = SubsystemsConfig.from_paths(repo=repo, commit=commit)
    build_subsystems(con, cfg)
    con.close()
```

Wire near the end of the analytics block, after `risk_factors`:

```python
("risk_factors", lambda: _run_task("risk_factors", t_risk_factors, run_logger, ctx.repo_root, ctx.repo, ctx.commit, ctx.db_path, ctx.build_dir)),
("subsystems", lambda: _run_task("subsystems", t_subsystems, run_logger, ctx.repo, ctx.commit, ctx.db_path)),
```

#### 2.4.3 Export mapping

In `docs_export/export_parquet.py` and `export_jsonl.py`, add:

```python
PARQUET_DATASETS.update({
    "analytics.subsystems": "subsystems.parquet",
    "analytics.subsystem_modules": "subsystem_modules.parquet",
})

JSONL_DATASETS.update({
    "analytics.subsystems": "subsystems.jsonl",
    "analytics.subsystem_modules": "subsystem_modules.jsonl",
})
```

So Document Output now has **subsystems.jsonl** and **subsystem_modules.jsonl**.

#### 2.4.4 Optional docs view: `docs.v_subsystem_summary`

In `storage/views.py`, add something like:

```sql
CREATE OR REPLACE VIEW docs.v_subsystem_summary AS
SELECT
    s.repo,
    s.commit,
    s.subsystem_id,
    s.name,
    s.description,
    s.module_count,
    s.modules_json,
    s.entrypoints_json,
    s.internal_edge_count,
    s.external_edge_count,
    s.fan_in,
    s.fan_out,
    s.function_count,
    s.avg_risk_score,
    s.max_risk_score,
    s.high_risk_function_count,
    s.risk_level
FROM analytics.subsystems s;
```

This gives the agent a **single query** to see “all subsystems and their health”.

You can also add `docs.v_module_with_subsystem` that joins `core.modules`, `analytics.graph_metrics_modules`, `analytics.subsystem_modules`, and aggregated risk, similar to your existing `docs.v_call_graph_enriched`.

---

## 3. How an AI agent would use these

Once implemented, an agent can:

* **Understand architecture topology quickly**:

  * Call `docs.v_subsystem_summary` to see the main subsystems, their size, and risk profile.
  * Use `subsystems.modules_json` or `analytics.subsystem_modules` to drill into modules.
* **Locate safe places for new code**:

  * Use `analytics.graph_metrics_modules` to avoid modules with very high `import_pagerank` or large `import_fan_in` when adding new features.
  * Use `analytics.graph_metrics_functions` for call graph‑aware placement/refactoring.
* **Plan refactors**:

  * Identify subsystems with high `external_edge_count` and high `high_risk_function_count` as candidates for boundary hardening.
* **Route tasks to the right “bounded context”**:

  * Given a target function or module, map to its subsystem via `analytics.subsystem_modules` and work mostly within that cluster.

---

If you’d like, I can next sketch the **exact SQL and Python signatures** for `compute_graph_metrics` and `build_subsystems` so your team can slot them directly into the codebase, or propose some example queries that an MCP tool or FastAPI endpoint would expose on top of these new datasets.
