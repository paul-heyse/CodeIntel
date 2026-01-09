# Analytics Acero/DSL Best-in-Class Implementation Plan

## Goal
Fully refactor analytics compute loops into a modular Acero/DSL architecture that
standardizes plan → execute → finalize, enforces deterministic outputs, and minimizes
Python row loops.

## Scope Principles
- Prefer Acero Plan nodes (scan/filter/project/join/aggregate/order_by) for all bulk work.
- Allow Python loops only for AST/graph algorithms that cannot be expressed as Arrow kernels.
- Centralize determinism, dedupe, and invariant checks in finalize policies.
- Keep streaming boundaries explicit; avoid accidental materialization.

---

## 1) Shared analytics plan builder (snapshot-scoped scans)

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import Plan, materialize_plan

def snapshot_plan(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    columns: Sequence[str],
) -> Plan:
    plan = Plan.table(table).filter(
        E.and_(
            E.field("repo") == E.scalar(repo),
            E.field("commit") == E.scalar(commit),
        )
    )
    return plan.project({name: E.field(name) for name in columns})

def snapshot_table(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    columns: Sequence[str],
    order_by: Sequence[str] | None = None,
) -> pa.Table:
    plan = snapshot_plan(table, repo=repo, commit=commit, columns=columns)
    if order_by:
        plan = plan.order_by(sort_keys=[(name, "ascending") for name in order_by])
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/analytics/utilities/catalogs.py
- src/codeintel/build/analytics/cfg_dfg/helpers.py

Checklist
- [x] Introduce a snapshot_plan/snapshot_table helper for analytics modules.
- [x] Replace duplicated repo/commit filter + select code with the helper.
- [x] Ensure Plan.order_by is applied when a deterministic worklist is required.
- Status: Completed (datasets/catalogs/cfg_dfg helpers updated).

---

## 2) Rowset builders for grouped lists and adjacency inputs

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import Plan, materialize_plan

def edges_by_fn_table(edges: pa.Table, *, repo: str, commit: str) -> pa.Table:
    plan = (
        Plan.table(edges)
        .filter(
            E.and_(
                E.field("repo") == E.scalar(repo),
                E.field("commit") == E.scalar(commit),
                E.is_valid("function_goid_h128"),
                E.is_valid("src_block_id"),
                E.is_valid("dst_block_id"),
            )
        )
        .project(
            {
                "function_goid_h128": E.field("function_goid_h128"),
                "src_block_id": E.field("src_block_id"),
                "dst_block_id": E.field("dst_block_id"),
                "edge_kind": E.field("edge_kind"),
            }
        )
        .aggregate(
            keys=[E.field("function_goid_h128")],
            aggregates=[("src_block_id", "list", None, "src_block_id")],
        )
        .order_by(sort_keys=[("function_goid_h128", "ascending")])
    )
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/cfg_dfg/cfg_core.py
- src/codeintel/build/analytics/cfg_dfg/dfg_core.py
- src/codeintel/build/analytics/functions/function_effects.py
- src/codeintel/build/analytics/graphs/config_graph_metrics.py
- src/codeintel/build/analytics/graphs/config_data_flow.py

Checklist
- [x] Replace per-row edge/block parsing with grouped list rowsets.
- [x] Use Plan.aggregate(list) to build adjacency inputs deterministically.
- [x] Convert list/struct outputs to graph inputs only at the final step.
- Status: Completed (CFG/DFG loaders + config_* graphs + function_effects).

---

## 3) Deterministic worklists for AST-heavy analytics

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import Plan, materialize_plan

def function_worklist(goids: pa.Table, *, repo: str, commit: str) -> pa.Table:
    plan = (
        Plan.table(goids)
        .filter(
            E.and_(
                E.field("repo") == E.scalar(repo),
                E.field("commit") == E.scalar(commit),
                E.in_("kind", ["function", "method"]),
            )
        )
        .project(
            {
                "goid_h128": E.field("goid_h128"),
                "rel_path": E.field("rel_path"),
                "qualname": E.field("qualname"),
                "start_line": E.field("start_line"),
                "end_line": E.field("end_line"),
            }
        )
        .aggregate(
            keys=[E.field("goid_h128")],
            aggregates=[
                ("rel_path", "min", None, "rel_path"),
                ("qualname", "min", None, "qualname"),
                ("start_line", "min", None, "start_line"),
                ("end_line", "max", None, "end_line"),
            ],
        )
        .order_by(sort_keys=[("rel_path", "ascending"), ("start_line", "ascending")])
    )
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/functions/metrics.py
- src/codeintel/build/analytics/functions/function_contracts.py
- src/codeintel/build/analytics/semantic_roles/core.py
- src/codeintel/build/analytics/compute/functions/goids.py

Checklist
- [x] Build deterministic worklists (goid + rel_path + qualname) in Plan.
- [x] Drive AST loops from the worklist table only.
- [x] Replace ad-hoc list dedupe with aggregate + order_by.
- Status: Completed (metrics/function_contracts/semantic_roles/goids).

---

## 4) Graph analytics rowsets (config graphs, subsystem graphs)

Representative pattern
```python
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import Plan, materialize_plan

def config_reference_rowset(table: pa.Table, *, repo: str, commit: str) -> pa.Table:
    plan = (
        Plan.table(table)
        .filter(
            E.and_(
                E.field("repo") == E.scalar(repo),
                E.field("commit") == E.scalar(commit),
                E.is_valid("config_path"),
                E.is_valid("key"),
            )
        )
        .project({"config_path": E.field("config_path"), "key": E.field("key")})
        .aggregate(
            keys=[E.field("config_path"), E.field("key")],
            aggregates=[],
        )
        .order_by(sort_keys=[("config_path", "ascending"), ("key", "ascending")])
    )
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/graphs/config_graph_metrics.py
- src/codeintel/build/analytics/graphs/config_references.py
- src/codeintel/build/analytics/graphs/config_data_flow.py
- src/codeintel/build/analytics/subsystems/affinity.py
- src/codeintel/build/analytics/subsystems/materialize.py

Checklist
- [x] Build config/reference rowsets via Plan.aggregate.
- [x] Replace list-building loops with grouped Arrow tables.
- [x] Only convert to rustworkx inputs at a single boundary function.
- Status: Completed (config_graph_metrics/config_references/config_data_flow + affinity/materialize).

---

## 5) Finalize policies for analytics outputs (determinism + dedupe)

Representative pattern
```python
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table

def finalize_analytics(table_key: str, table: pa.Table) -> pa.Table:
    spec = FinalizeSpec(
        table_key=table_key,
        mode="tolerant",
        order_by=resolve_stable_sort_keys(get_schema_service().get_table_schema(table_key)),
    )
    return finalize_table(table, spec=spec).good
```

Target files
- src/codeintel/build/hamilton/native/analytics/finalize_helpers.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/post_run_quality_outputs.py

Checklist
- [x] Ensure every analytics write path finalizes with order_by.
- [x] Centralize per-table determinism policy in finalize helpers.
- [x] Remove any remaining ad-hoc ordering from analytics nodes.
- Status: Completed (finalize helpers + datasets + post-run outputs).

---

## 6) CFG/DFG edge and block loaders (full plan-based rowsets)

Representative pattern
```python
def cfg_blocks_by_fn(blocks: pa.Table, *, repo: str, commit: str) -> pa.Table:
    plan = (
        Plan.table(blocks)
        .filter(E.and_(E.field("repo") == E.scalar(repo), E.field("commit") == E.scalar(commit)))
        .project(
            {
                "function_goid_h128": E.field("function_goid_h128"),
                "block_idx": E.field("block_idx"),
                "kind": E.field("kind"),
                "in_degree": E.field("in_degree"),
                "out_degree": E.field("out_degree"),
            }
        )
        .aggregate(
            keys=[E.field("function_goid_h128")],
            aggregates=[
                ("block_idx", "list", None, "block_idx"),
                ("kind", "list", None, "kind"),
                ("in_degree", "list", None, "in_degree"),
                ("out_degree", "list", None, "out_degree"),
            ],
        )
        .order_by(sort_keys=[("function_goid_h128", "ascending")])
    )
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/cfg_dfg/cfg_core.py
- src/codeintel/build/analytics/cfg_dfg/dfg_core.py
- src/codeintel/build/analytics/cfg_dfg/helpers.py

Checklist
- [x] Build blocks/edges grouped rowsets via Plan.aggregate(list).
- [x] Replace per-row parsing loops with list/struct decoding.
- [x] Keep rustworkx graph construction as the only Python-heavy step.
- Status: Completed (cfg_core/dfg_core/helpers).

---

## 7) Quality reports (plan-first aggregation)

Representative pattern
```python
def edge_counts(cpg_edges: pa.Table, *, repo: str, commit: str) -> pa.Table:
    plan = (
        Plan.table(cpg_edges)
        .filter(E.and_(E.field("repo") == E.scalar(repo), E.field("commit") == E.scalar(commit)))
        .project({"edge_kind": E.field("edge_kind"), "edge_layer": E.field("edge_layer")})
        .aggregate(
            keys=[E.field("edge_kind"), E.field("edge_layer")],
            aggregates=[("edge_kind", "count", None, "edge_count")],
        )
        .order_by(sort_keys=[("edge_kind", "ascending"), ("edge_layer", "ascending")])
    )
    return materialize_plan(plan, use_threads=True)
```

Target files
- src/codeintel/build/analytics/py_cpg_quality_report.py
- src/codeintel/build/analytics/scip_diagnostics_rollups.py

Checklist
- [x] Convert counter loops to Plan.aggregate.
- [x] Use grouped counts instead of Python accumulation.
- [x] Finalize and order outputs before persistence.
- Status: Completed (py_cpg_quality_report + scip_diagnostics_rollups).

---

## 8) Integration + safety checks

Representative pattern
```python
def require_columns(table: pa.Table, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
```

Target files
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/analytics/utilities/catalogs.py
- src/codeintel/build/analytics/cfg_dfg/helpers.py

Checklist
- [x] Add explicit schema guards before plan construction.
- [x] Ensure list aggregates are only used after canonical ordering.
- [x] Add static guardrails for deterministic ordering and list aggregation.
- Status: Completed (snapshot guards + analytics rowset guardrail).

---

## Sequencing Recommendation
1) Shared analytics plan builder + finalize policy unification.
2) CFG/DFG loaders + function worklists.
3) Semantic roles + function effects + config graphs.
4) Quality report aggregation upgrades.

## Expected Outcome
Analytics compute becomes Acero-first, with Python loops only for AST/graph logic.
All outputs are deterministic, contract-aligned, and easy to integrate into new
pipelines with shared plan utilities.

---

## Completion Summary (latest)
- Action set 1 (finalize compliance): Completed via analytics finalize write guardrail.
- Action set 2 (iter_rows guardrail): Completed via analytics iter_rows guardrail.
- Remaining scope: None in this plan.
