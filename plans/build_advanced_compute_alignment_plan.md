# Build Advanced Compute Alignment Plan (CPG + Analytics)

This plan integrates the latest compute design decisions and the remaining
alignment opportunities across CPG2, analytics, and graph validation. The
goal is to converge on the Arrow-first pipeline pattern:
plan -> reader -> finalize (with deterministic ordering and diagnostics).

## Scope items

### 1) Adopt a single join-precheck diagnostics dataset (CPG + analytics)
Use `build.join_precheck_issues` as the shared diagnostics dataset for all
pipeline types. Avoid creating a graph-specific dataset unless a strict
retention or privacy boundary requires it.

**Code pattern**
```python
precheck = finalize_join_keys(
    table,
    required_non_null=join_keys,
    key_fields=key_fields,
    stage="join_precheck",
)
record_join_precheck_errors(
    precheck,
    table_key=table_key,
    target_name=target_name,
    join_keys=join_keys,
)
clean = precheck.good
```

**Target files**
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/analytics/subsystems/cache.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`

**Checklist**
- [ ] Add join-precheck finalize + record calls before all hash-join inputs.
- [ ] Ensure `table_key` and `target_name` are wired for CPG/analytics joins.
- [ ] Confirm join-precheck errors persist via `build.join_precheck_issues`.
- [ ] Remove any ad hoc join-key filters that drop invalid rows silently.

### 2) Determinism tier policy for CPG/analytics outputs
Define canonical ordering for persisted/shared outputs and stable-set ordering
for internal intermediates. Canonical outputs must apply explicit `order_by`
with stable tie-breakers before finalize.

**Code pattern**
```python
plan = Plan.table(left).hash_join(right=Plan.table(right), spec=spec)
if determinism == "canonical":
    sort_keys = [(key, "ascending") for key in spec.left_keys] + [
        ("edge_ordinal", "ascending"),
    ]
    plan = plan.order_by(sort_keys=sort_keys)
result = finalize_reader(
    plan.to_reader(),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
)
```

**Target files**
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/analytics/subsystems/cache.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`

**Checklist**
- [ ] Enumerate CPG/analytics outputs that are persisted or reused downstream.
- [ ] Add explicit tie-breakers (cpg_id, edge_ordinal, or a stable row id).
- [ ] Require `order_by` for canonical outputs and skip for internal outputs.
- [ ] Avoid order-dependent aggregations without pre-sorting.

### 3) Replace CPG2 join filters with join-precheck + plan ordering
CPG2 plane joins still use `E.is_valid` filters and post-materialization
sorting, which hides join-key errors and forces full table loads. Replace
filters with `finalize_join_keys`, record precheck errors, and move ordering
into the plan.

**Code pattern**
```python
precheck = finalize_join_keys(
    left_rows,
    required_non_null=spec.left_keys,
    key_fields=key_fields,
    stage="join_precheck",
)
record_join_precheck_errors(
    precheck,
    table_key=table_key,
    target_name=target_name,
    join_keys=spec.left_keys,
)
plan = Plan.table(precheck.good).hash_join(right=Plan.table(right_rows), spec=spec)
ordered = plan.order_by(sort_keys=[(key, "ascending") for key in spec.left_keys])
result = finalize_reader(
    ordered.to_reader(),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
)
```

**Target files**
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`

**Checklist**
- [ ] Replace `E.is_valid` join filters with `finalize_join_keys` prechecks.
- [ ] Record join-precheck errors for all CPG2 join inputs.
- [ ] Push sorting into the plan (`order_by`) before finalize.
- [ ] Prefer `finalize_reader` to preserve streaming until finalize.

### 4) Upgrade analytics cache join to join-precheck + plan ordering
Analytics cache joins currently mirror the legacy key-filter pattern. Adopt
join-precheck error routing and plan-level ordering for deterministic cache
rows.

**Code pattern**
```python
precheck = finalize_join_keys(
    rows,
    required_non_null=spec.left_keys,
    key_fields=key_fields,
    stage="join_precheck",
)
record_join_precheck_errors(
    precheck,
    table_key=table_key,
    target_name=target_name,
    join_keys=spec.left_keys,
)
joined = Plan.table(precheck.good).hash_join(right=Plan.table(lookup), spec=spec)
ordered = joined.order_by(sort_keys=[(key, "ascending") for key in spec.left_keys])
cache_rows = finalize_reader(
    ordered.to_reader(),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
)
```

**Target files**
- `src/codeintel/build/analytics/subsystems/cache.py`

**Checklist**
- [ ] Replace join-key filters with `finalize_join_keys` + precheck errors.
- [ ] Apply `order_by` before finalize for canonical cache outputs.
- [ ] Ensure cache rows are finalized via `finalize_reader`.

### 5) Move graph validation to reader-first, plan-driven checks
Current validation scans materialize tables and iterate row-wise. Replace
row loops with `Plan.scan` + vectorized aggregates, and keep scans streaming.

**Code pattern**
```python
plan = Plan.scan(
    dataset,
    columns={"repo": E.field("repo"), "commit": E.field("commit")},
    filter_expr=filter_expr,
    implicit_ordering=True,
)
plan = plan.aggregate(
    keys=[E.field("repo"), E.field("commit")],
    aggregates=[("repo", "count", None, "row_count")],
)
result = finalize_reader(
    plan.to_reader(),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
)
```

**Target files**
- `src/codeintel/build/graphs/validation/checks/database.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Checklist**
- [ ] Identify row-wise validation loops and map them to aggregations.
- [ ] Add projection + filter pushdown to scan requests.
- [ ] Use reader-first pipelines and finalize via `finalize_reader`.
- [ ] Enable scan telemetry for validation scans.

### 6) Replace call-wiring row lists with columnar collectors or plan aggregates
Call wiring currently builds rows in Python lists and uses `pa.Table.from_pylist`.
Switch to contract-aware collectors or Arrow-native aggregation patterns.

**Code pattern**
```python
collector = columnar_batch_collector_for_table_key(CALL_WIRING_TABLE_KEY)
collector.extend(rows)
edges = collector.to_table()
result = finalize_table(
    edges,
    spec=FinalizeSpec(table_key=CALL_WIRING_TABLE_KEY, mode="tolerant"),
)
```

**Target files**
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`

**Checklist**
- [ ] Replace `pa.Table.from_pylist` with columnar collectors.
- [ ] Move dedupe steps to Plan.aggregate or finalize dedupe policies.
- [ ] Finalize results against table contracts and emit artifacts.

### 7) Replace scip_resolution occurrence-syntax loops with Arrow joins
The occurrence-syntax xref path still uses Python loops and dict joins.
Rebuild it as a hash-join pipeline with join-precheck routing and plan ordering.

**Code pattern**
```python
precheck = finalize_join_keys(
    occurrences,
    required_non_null=spec.left_keys,
    key_fields=key_fields,
    stage="join_precheck",
)
record_join_precheck_errors(
    precheck,
    table_key=table_key,
    target_name=target_name,
    join_keys=spec.left_keys,
)
joined = Plan.table(precheck.good).hash_join(right=Plan.table(syntax_nodes), spec=spec)
ordered = joined.order_by(sort_keys=[(key, "ascending") for key in spec.left_keys])
result = finalize_ingest_reader(
    table_key,
    ordered.to_reader(),
    target_name=target_name,
)
```

**Target files**
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`

**Checklist**
- [ ] Build join-key tables for occurrences and syntax nodes.
- [ ] Route join-precheck errors to `build.join_precheck_issues`.
- [ ] Replace Python loops with Plan.hash_join + order_by.
- [ ] Finalize via `finalize_ingest_reader` for ingestion policy alignment.
