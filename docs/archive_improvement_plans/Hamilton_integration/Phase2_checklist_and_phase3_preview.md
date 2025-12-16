Absolutely — here are both:

1. a **Phase 2 “Definition of Done” (DoD) checklist**, written as concrete acceptance criteria you can literally tick off in CI + via CLI usage, and
2. a **Phase 3 preview** showing how to migrate selected targets (starting with `risk_factors` + `function_metrics_ext`) into **pure Hamilton compute nodes + explicit materializers**, with realistic code snippets and a practical migration strategy.

I’m treating your **Phase 1 state** as: full DAG coverage (generated nodes), closure execution + correct computed/skipped/failed reporting, upstream failure gating, `--force`, run tracking in `build.runs`, universal dataset nodes (`d__*`) populated for all contracts, and DAG export tooling. 

---

# Phase 2 Definition of Done checklist

This is intentionally written as “ship criteria”. If all of these are true, Phase 2 is done.

## A. Engine behavior and CLI contract

### A1. Hamilton is the default

* [ ] `codeintel build run ...` defaults to Hamilton engine without requiring `--engine hamilton`.
* [ ] Legacy engine is either removed or gated behind an explicit flag:

  * `--engine legacy` (explicit only), OR
  * `CODEINTEL_ALLOW_LEGACY=1` for compatibility.

### A2. Explicit Hamilton runtime mode is supported

* [ ] CLI supports `--hamilton-mode generated|phase0`.
* [ ] `generated` is the default.
* [ ] `phase0` mode still works for debugging and runs the Phase 0 chain end-to-end.

### A3. CLI output parity is maintained

* [ ] Build output (human and JSON, if applicable) reports:

  * requested targets
  * closure targets
  * computed targets
  * skipped targets
  * failed targets
* [ ] These lists are closure-complete (not only requested) and match stored run records.

---

## B. Planning and explainability: best-in-class “what will run and why?”

### B1. Planner produces deterministic, actionable plans

* [ ] `codeintel build run --dry-run ...` produces a **plan** that includes for each target in closure:

  * status: `compute|skip|missing|blocked`
  * reason: `forced|no_manifest|hash_changed|up_to_date|upstream_failed|missing_dep_manifest|unknown_target`
  * computed `options_hash`
  * computed `input_hash`
  * `prior_input_hash` when available
  * dependencies list
  * produced table_keys and artifact_keys from the contract
* [ ] Plan ordering is topological and stable.

### B2. Plan matches execution

* [ ] If you run:

  1. `--dry-run` and then
  2. the real `build run`

  the set of targets marked `compute` equals the set that actually compute (except for failures turning downstream into blocked/skipped).

### B3. Explain mode exists (or “plan diff”)

* [ ] `codeintel build explain <target>` (or `build plan --diff`) surfaces:

  * *which dependency changed* (dep input hash mismatch)
  * *which config changed* (options hash mismatch)
  * *forced vs hash-based recompute*
* [ ] Output can be emitted as JSON for tooling.

---

## C. Incremental build performance: manifest index + stable hash cascade

### C1. Manifest prefetch is used (no N× DB round trips)

* [ ] For any run, the system fetches prior manifests **once** (or in a small bounded set of queries).
* [ ] Skip checks and hash computation use a manifest index stored in `BuildEnv` (or equivalent).

### C2. Hash semantics cascade through dependencies correctly

* [ ] If an upstream dependency changes (options/input), downstream targets’ computed `input_hash` changes deterministically.
* [ ] Hash computation for a target uses:

  * target identity
  * repo+commit
  * options hash
  * dependency input hashes (or dependency manifest input hashes)

### C3. Planner and runner share the same hashing logic

* [ ] The planner and executor call the same helper functions for:

  * options hashing
  * input hashing
  * skip evaluation

---

## D. Asset-centric DAG: DatasetRef v2 and loader nodes

### D1. DatasetRef includes snapshot identity

* [ ] `DatasetRef` includes `repo` + `commit` (or snapshot id) so loaders can filter safely.

### D2. Skipped targets still produce dataset refs

* [ ] When a target is skipped as up-to-date, its `TargetRunRecord.datasets` is still populated (row_count may be unknown/None).
* [ ] This allows dataset nodes (`d__*`) to still work for downstream consumers even when upstream was skipped.

### D3. Loader nodes exist for each dataset

For every dataset key in the contract graph:

* [ ] `d__schema__table` exists and yields `DatasetRef`
* [ ] `q__schema__table` exists and yields an Ibis expression (or equivalent)
* [ ] `df__schema__table` exists and yields a pandas DataFrame (for debugging/validation paths)

### D4. Optional validation switch works

* [ ] CLI supports `--validate-outputs` (or config setting).
* [ ] When enabled, produced datasets are validated post-write (Pandera via SCHEMA_REGISTRY).
* [ ] Validation failures:

  * mark the target failed
  * prevent downstream computation

---

## E. Run and target observability: persisted per-target records

### E1. `build.runs` remains the canonical run header

* [ ] Each Hamilton run writes to `build.runs` at start and completion (Phase 1 behavior preserved).

### E2. New `build.run_targets` exists and is populated

* [ ] A `build.run_targets` dataset/table exists with per-target rows:

  * run_id
  * repo, commit
  * target
  * plugin_name / implementation
  * status
  * duration_ms
  * input_hash, options_hash
  * error
  * row_counts (json)
* [ ] A run with closure N targets produces N rows (including skipped/blocked/fail).

### E3. CLI can display per-target breakdown

* [ ] `codeintel build history --run-id X` (or equivalent) can show per-target rows.
* [ ] Slowest targets can be summarized.

---

## F. DAG export and visualization: JSON + Mermaid + DOT

### F1. Graph export is stable and complete

* [ ] `codeintel build graph ... --format json|mermaid|dot` produces a closure-complete representation.
* [ ] Export includes:

  * nodes: targets + datasets (+ loaders if you decide)
  * edges: dependencies
  * metadata: module/domain, table_keys, tags, etc.

### F2. Export is useful for PR review

* [ ] Mermaid output renders cleanly in GitHub Markdown.
* [ ] DOT output works with graphviz.

---

## G. Test and CI gates (hard requirements)

### G1. Unit tests under `tests/build`

* [ ] Planner tests:

  * forced, no manifest, hash_changed, up_to_date
  * blocked due to upstream failure
* [ ] Manifest-index tests:

  * verify no per-target manifest loads occur
* [ ] DatasetRef v2 tests:

  * skipped targets still populate datasets
* [ ] Loader nodes tests:

  * generated `q__` and `df__` nodes exist and are callable
* [ ] Graph export tests:

  * mermaid/dot/json contain expected edges for a tiny graph
* [ ] Run targets persistence test:

  * run inserts rows into `build.run_targets`

### G2. One integration “golden run” test

* [ ] A single test that:

  1. runs a small closure
  2. runs again and confirms everything skips
  3. forces a target and confirms that exact target recomputes
  4. validates output tables (optional but recommended)

---

## H. Documentation (still part of DoD)

* [ ] `docs/build/hamilton.md` (or similar) documents:

  * node naming conventions
  * how to plan/explain
  * how to export graphs
  * how to interpret `build.run_targets`
  * how to enable validations
* [ ] Add a troubleshooting section:

  * how to diagnose “why did this recompute?”
  * how to view dependency chain

---

# Phase 3 preview: migrate selected targets to pure Hamilton compute + explicit materializers

Phase 2 makes the system *observable and asset-centric*. Phase 3 is where you get “best-in-class” performance and design:

> Targets stop being “plugins with side effects” and become **Hamilton pipelines that compute typed artifacts and materialize them explicitly**.

## Phase 3 guiding principles

1. **Computation nodes return data, not side effects**

   * return an Ibis table expression, pandas DataFrame, Arrow table, or structured Python object.

2. **Materialization is explicit**

   * one (or a few) nodes are responsible for writing outputs to DuckDB.
   * those nodes are the only ones with DB write side effects.

3. **Contracts and validations live at boundaries**

   * validate pre-materialization (type/schema)
   * validate post-materialization (row count, keys, constraints)

4. **Caching becomes meaningful**

   * pure compute nodes can be cached safely
   * materializer nodes can be configured as “always run” or “recompute” depending on policy

5. **Migration is target-by-target**

   * you don’t rewrite everything.
   * you “flip” one target at a time from “plugin runner node” → “pure pipeline”.

---

## Phase 3 architecture additions

### New package layout

Add a **dataflow** layer that is Hamilton-native:

```
codeintel/build/hamilton/
  dataflow/
    __init__.py
    analytics/
      __init__.py
      function_metrics_ext.py
      risk_factors.py
    graphs/
      __init__.py
      call_graph_views.py
  materializers/
    __init__.py
    duckdb.py
  adapters/
    __init__.py
    duckdb_io.py
```

### New registry: “native implementations”

Add a registry that maps a target name to the Hamilton module implementing it natively:

```python
# codeintel/build/hamilton/native_registry.py
NATIVE_TARGET_MODULES: dict[str, str] = {
    "function_metrics_ext": "codeintel.build.hamilton.dataflow.analytics.function_metrics_ext",
    "risk_factors": "codeintel.build.hamilton.dataflow.analytics.risk_factors",
}
```

Then in `driver_factory.build_driver(...)`, include:

* generated nodes module (for everything)
* plus the native modules, which *override* the generated wrapper nodes by defining the same `t__...` names.

Hamilton resolves function names by last module loaded; if your driver construction order ensures the “native” module functions win, you can override seamlessly.

---

## Phase 3 migration target selection

### Best first candidates (high ROI, low risk)

* **`risk_factors`**: usually a deterministic derived dataset from `call_graph` + `function_metrics`
* **`function_metrics_ext`** (or create it): derived metrics computed from AST/goids tables and/or intermediate views

These are typically:

* well-defined table outputs
* “data transforms” that are naturally expressed in Ibis/SQL
* easy to validate with schema registry
* benefit greatly from caching

### Harder later candidates

* ingestion targets (SCIP, AST extraction, etc.) that touch filesystem/tools.

  * still doable — but you’ll want dynamic execution and careful caching policies.

---

## Phase 3 example: `risk_factors` as pure Hamilton pipeline

### What it should look like

**Inputs**: dataset loaders for:

* `q__analytics__function_metrics_ext` (or function_metrics)
* `q__graph__call_graph_edges` (or call_graph views)

**Compute**:

* join/aggregate → compute risk factors as an Ibis expression

**Validate**:

* optional: materialize to pandas and validate with Pandera (or validate row-level invariants with Ibis)

**Materialize**:

* write to DuckDB via `IbisGateway.write()` (or a Hamilton DataSaver)

### Code sketch: compute node returning an Ibis table

```python
# codeintel/build/hamilton/dataflow/analytics/risk_factors.py
from __future__ import annotations

import ibis.expr.types as ir
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv

# This node is "pure": no DB writes.
@tag(domain="analytics", target="risk_factors", kind="compute")
def t__risk_factors_compute(
    env: BuildEnv,
    q__analytics__function_metrics_ext: ir.Table,   # from Phase 2 loader nodes
    q__graph__call_graph_edges: ir.Table,           # from Phase 2 loader nodes
) -> ir.Table:
    fm = q__analytics__function_metrics_ext
    edges = q__graph__call_graph_edges

    # Example: compute fan-in / fan-out from edges and join onto function metrics
    fan_out = (
        edges.group_by(edges.caller_id)
             .aggregate(fan_out=edges.callee_id.nunique())
             .rename({"caller_id": "function_id"})
    )
    fan_in = (
        edges.group_by(edges.callee_id)
             .aggregate(fan_in=edges.caller_id.nunique())
             .rename({"callee_id": "function_id"})
    )

    joined = (
        fm.left_join(fan_out, ["function_id"])
          .left_join(fan_in, ["function_id"])
          .mutate(
              fan_out=fm.fan_out.fill_null(0),
              fan_in=fm.fan_in.fill_null(0),
          )
    )

    # Example risk score (placeholder): weight metrics + graph centrality proxies
    risk = joined.mutate(
        risk_score=(
            joined.cyclomatic_complexity.fill_null(0) * 0.4 +
            joined.fan_in.fill_null(0) * 0.3 +
            joined.fan_out.fill_null(0) * 0.3
        )
    )

    return risk
```

### Code sketch: explicit materializer node

```python
# codeintel/build/hamilton/dataflow/analytics/risk_factors.py
from hamilton.function_modifiers import tag

@tag(domain="analytics", target="risk_factors", kind="materialize")
def t__risk_factors(
    env: BuildEnv,
    graph,  # TargetGraph still passed through for metadata/contract lookup
    t__risk_factors_compute,  # ibis table
):
    # Write to DuckDB (explicit side effect boundary)
    table_key = "analytics.risk_factors"
    env.gateway.ibis.write(
        table_key,
        t__risk_factors_compute,
        overwrite=True,  # or whatever your policy is
    )

    # Build a TargetRunRecord with datasets, hashes, manifests, etc.
    # (You can reuse your existing _run_target record builder helpers, but
    # skip plugin.execute and treat this pipeline as the "plugin".)
    ...
```

### Why this is powerful

* The compute node becomes cacheable and testable.
* The materializer is a single clean side-effect boundary.
* You can validate `t__risk_factors_compute` without writing anything.

---

## Phase 3 example: `function_metrics_ext` as pure pipeline

This is typically a transform over AST/goids tables (or intermediate extracted tables).

### Compute node

```python
# codeintel/build/hamilton/dataflow/analytics/function_metrics_ext.py
from __future__ import annotations

import ibis.expr.types as ir
from hamilton.function_modifiers import tag
from codeintel.build.hamilton.env import BuildEnv

@tag(domain="analytics", target="function_metrics_ext", kind="compute")
def t__function_metrics_ext_compute(
    env: BuildEnv,
    q__graphs__goids: ir.Table,         # adjust to your actual table keys
    q__ingestion__ast: ir.Table,        # adjust to your actual table keys
) -> ir.Table:
    # Example: aggregate AST-derived metrics by function_id
    # (placeholder structure; real impl depends on your schema)
    ast = q__ingestion__ast
    goids = q__graphs__goids

    # Suppose ast contains (function_id, node_type, ...)
    counts = (
        ast.group_by(ast.function_id)
           .aggregate(
               stmt_count=ast.node_type.count(),
               branch_count=(ast.node_type == "If").sum(),
           )
    )

    enriched = goids.left_join(counts, ["function_id"])
    return enriched
```

### Materializer node

```python
@tag(domain="analytics", target="function_metrics_ext", kind="materialize")
def t__function_metrics_ext(env: BuildEnv, graph, t__function_metrics_ext_compute: ir.Table):
    env.gateway.ibis.write("analytics.function_metrics_ext", t__function_metrics_ext_compute, overwrite=True)
    ...
```

---

## How Phase 3 integrates with your existing Phase 1/2 machinery

### Keep what already works

* Closure computation, run tracking, upstream gating, `--force` are still valid and should remain.
* DatasetRef + dataset nodes remain the connective tissue.

### Replace only the “execution core” for selected targets

Right now, target nodes probably do:

* hash/skip
* if compute: run plugin.execute(ctx)
* persist manifests
* populate datasets

In Phase 3, for a migrated target:

* hash/skip stays
* compute becomes: execute Hamilton-native subgraph (compute node(s) + materialize node)
* manifests/datasets are still recorded

So you essentially swap:

* `plugin.execute(ctx)`
  with
* `materialize_node(...)` built from pure compute DAG nodes

---

## Phase 3 migration checklist for each target

For each migrated target (e.g., `risk_factors`):

1. **Define the asset boundary**

   * table keys it produces (`analytics.risk_factors`)

2. **Create loader dependencies**

   * use Phase 2 `q__*` nodes for upstream tables

3. **Implement pure compute node(s)**

   * return ibis expressions or dataframes

4. **Add validation**

   * Pandera schema check (optional at first, but recommended)

5. **Add explicit materializer**

   * write to DuckDB via gateway (single boundary)

6. **Integrate into driver**

   * native module overrides generated wrapper for `t__risk_factors`

7. **Add tests under `tests/build`**

   * unit test compute node logic (small in-memory ibis / mock tables)
   * integration test: run just that target and confirm table exists + schema valid

---

## Phase 3 “Definition of Done” (preview)

You can treat Phase 3 as complete when:

* [ ] At least 2 “serving-facing” targets (e.g., `risk_factors` + `function_metrics_ext`) are **pure compute + explicit materialize**
* [ ] For those targets:

  * plugin execution path is no longer used
  * caching is safe and yields speedups
  * validation is supported
  * dataset nodes/loader nodes remain consistent

---

If you want, next I can turn this into a **Phase 2 PR-by-PR DoD gate sheet** (PR‑08…PR‑15) that lists:

* exact new/modified tests per PR under `tests/build/`
* exact CLI snapshots (“golden output”) you can assert against
* an explicit “migration map” for Phase 3: which target uses native pipeline vs plugin wrapper (and how to flip them gradually).
