# CPG Symbol Gap Resolution Plan

This plan eliminates missing SYMBOL-layer destinations by introducing explicit
external/unresolved node planes and routing edges to them without forced
matching. The goal is 0 missing dst CPG edges while preserving uncertainty and
provenance.

Each scope item includes representative code patterns and a target file list.

## 0. Schema + contract additions

Status: Planned

Goal:
- Add contract-backed tables for external SCIP symbols and unresolved symtable
  bindings.
- Ensure schema generation picks up the new tables and row models.

Proposed tables:
- `core.scip_external_symbols`
  - PK: (`repo`, `commit`, `symbol`)
  - Columns: `repo`, `commit`, `symbol`, `origin_kind`, `sample_rel_path`,
    `created_at`
- `core.py_sym_unresolved_bindings`
  - PK: (`repo`, `commit`, `rel_path`, `binding_id`)
  - Columns: `repo`, `commit`, `rel_path`, `binding_id`, `resolution_kind`,
    `confidence`, `reason`, `created_at`

Representative code pattern:
```python
TableSchema(
    schema="core",
    name="scip_external_symbols",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("symbol", "VARCHAR", nullable=False),
        Column("origin_kind", "VARCHAR", nullable=False),
        Column("sample_rel_path", "VARCHAR"),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "symbol"),
    description="SCIP symbols referenced by occurrences/relationships without\n"
    "matching scip_symbol_information rows.",
)
```

File targets:
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/data_models/rows.py`
- `src/codeintel/core/schemas/row_models.py` (regenerated)
- `tools/schema_diff.py` (run after schema changes)

## 1. Build `core.scip_external_symbols`

Status: Planned

Goal:
- Derive external SCIP symbols from occurrences + relationships that are not
  present in `core.scip_symbol_information`.
- Preserve origin and a sample rel_path for traceability.

Implementation:
1. Select distinct symbols from `core.scip_occurrences` and
   `core.scip_symbol_relationships`.
2. Left-anti join against `core.scip_symbol_information` on
   (`repo`, `commit`, `symbol`).
3. Emit a table with origin metadata and align to the contract.

Representative code pattern:
```python
symbols = concat_tables_unified(
    [
        occ.select(["repo", "commit", "symbol", "rel_path"]),
        rels.select(["repo", "commit", "symbol", "rel_path"]),
    ]
)
normalized = normalize_table_for_join(symbols)
info = normalize_table_for_join(symbol_info.select(["repo", "commit", "symbol"]))
missing = arrow_join_tables(
    normalized,
    info,
    spec=ArrowJoinSpec(on=["repo", "commit", "symbol"], how="anti"),
    options=build_join_options(normalized, info, normalize_inputs=False),
)
external = append_constant_columns(
    missing,
    {"origin_kind": "missing_symbol_info", "created_at": created_at},
)
external = dedupe_table_for_table("core.scip_external_symbols", external)
external = align_table_to_contract(
    "core.scip_external_symbols",
    external,
    target_name=SCIP_RESOLUTION_TARGET_NAME,
    reporter=emit_alignment_report,
)
```

File targets:
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py` (if wiring in ingestion)
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py` (table context)

## 2. Build `core.py_sym_unresolved_bindings`

Status: Planned

Goal:
- Materialize unresolved dst bindings referenced by
  `core.py_sym_resolution_edges` (e.g., `*:unknown`, `kind=UNKNOWN`).
- Ensure all unresolved dst nodes are explicit and referenceable.

Implementation:
1. Filter `core.py_sym_resolution_edges` where `dst_binding_id` is unknown or
   `kind == "UNKNOWN"`.
2. Left-anti join against `core.py_sym_bindings` to guarantee the binding does
   not exist.
3. Emit a deduped table with metadata (`resolution_kind`, `confidence`, `reason`).

Representative code pattern:
```python
unknown_mask = or_kleene(
    require_array(call_compute("ends_with", [edges["dst_binding_id"], ":unknown"]), name="ends_with"),
    equal_mask(edges["kind"], "UNKNOWN"),
)
unknown = safe_filter(edges, unknown_mask)
unknown = unknown.select(
    ["repo", "commit", "rel_path", "dst_binding_id", "kind", "confidence", "reason"]
).rename_columns(
    ["repo", "commit", "rel_path", "binding_id", "resolution_kind", "confidence", "reason"]
)
bindings = py_sym_bindings.select(["repo", "commit", "rel_path", "binding_id"])
missing = arrow_join_tables(
    normalize_table_for_join(unknown),
    normalize_table_for_join(bindings),
    spec=ArrowJoinSpec(
        on=["repo", "commit", "rel_path", "binding_id"],
        how="anti",
    ),
    options=build_join_options(unknown, bindings, normalize_inputs=False),
)
missing = append_constant_columns(missing, {"created_at": created_at})
```

File targets:
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/symtable.py` (if a dedicated module exists)

## 3. Add CPG nodes for external symbols + unresolved bindings

Status: Planned

Goal:
- Emit explicit CPG nodes for the new tables with distinct `node_kind` values.
- Keep `source_table_key` pointing to the new dataset keys for traceability.

Implementation:
1. Add new node builders in `cpg2/planes/scip.py` and `cpg2/planes/py_sym.py`.
2. Ensure `source_pk_json` is stored (as with other nodes) and `rel_path` is set
   when available.

Representative code pattern:
```python
def cpg2_nodes__scip_external_symbols(symbols: pa.Table) -> pa.Table:
    required = {"repo", "commit", "symbol"}
    if not required.issubset(symbols.column_names):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized = canonicalize_for_table(symbols, table_key="core.scip_external_symbols")
    anchors = build_anchor_map(
        normalized,
        table_key="core.scip_external_symbols",
        pk_columns=("repo", "commit", "symbol"),
        include_source_pk_json=True,
    )
    anchors = append_constant_columns(
        anchors,
        {"node_kind": "SCIP_SYMBOL_EXTERNAL", "source_table_key": "core.scip_external_symbols"},
    )
    return anchors.select(
        [
            "repo",
            "commit",
            "cpg_node_id",
            "node_kind",
            "source_table_key",
            "source_pk_json",
            "rel_path",
            "start_byte",
            "end_byte",
            "extras_json",
        ]
    )
```

File targets:
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/py_sym.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/assemble.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/types.py`

## 4. Route SYMBOL edges to internal/external/unresolved nodes

Status: Planned

Goal:
- Route CPG SYMBOL edges to internal nodes when present, otherwise to external
  or unresolved nodes. Do not force matches.
- Preserve provenance in `extras_json` for downstream filtering.

Implementation:
1. Build two anchor maps for SCIP symbols (internal + external).
2. Join edges against internal anchors first; fallback to external anchors when
   internal is missing.
3. Update symtable RESOLVES_TO edges to use unresolved binding table key when
   `dst_binding_id` is unknown.

Representative code patterns:
```python
# SCIP occurrence edges: choose internal or external target.
joined = arrow_join_tables(occ_rows, internal_anchor, spec=spec, options=options)
joined = arrow_join_tables(joined, external_anchor, spec=spec_ext, options=options_ext)
joined = joined.append_column(
    "dst_cpg_node_id",
    pc.coalesce(joined["dst_cpg_node_id"], joined["dst_cpg_node_id_ext"]),
)
joined = joined.append_column(
    "extras_json",
    pa.array([
        encode_payload({"symbol_origin": "external"}) if is_ext else None
        for is_ext in pc.is_null(joined["dst_cpg_node_id"]).to_pylist()
    ]),
)
```

```python
# Symtable resolution edges: route unknown dst bindings.
def _dst_table_key(row: Mapping[str, object]) -> str:
    dst = row.get("dst_binding_id")
    if isinstance(dst, str) and dst.endswith(":unknown"):
        return "core.py_sym_unresolved_bindings"
    return PY_SYM_BINDINGS_TABLE_KEY

rows.append(
    {
        "dst_cpg_node_id": cpg_node_id(_dst_table_key(row), dst_pk),
        "edge_kind": "RESOLVES_TO",
        "edge_layer": "SYMBOL",
    }
)
```

File targets:
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/overlays_symtable.py`

## 5. Diagnostics + quality reporting

Status: Planned

Goal:
- Track external/unresolved rates without penalizing correctness.
- Confirm missing dst edges go to 0 after node creation.

Implementation:
1. Extend `analytics.py_cpg_quality_report` to include:
   - `external_symbol_rate`
   - `unresolved_binding_rate`
2. Add a small check in `cpg_edge_integrity` or post-run audit to confirm
   missing dst is now 0 and record the external/unresolved counts.

Representative code pattern:
```python
report["external_symbol_rate"] = external_count / max(symbol_edges, 1)
report["unresolved_binding_rate"] = unresolved_count / max(resolve_edges, 1)
```

File targets:
- `src/codeintel/build/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/causal_analysis/cpg_symbol_destination_audit.py`
- `src/codeintel/build/hamilton/post_run_quality_outputs.py` (if surfaced post-run)

## 6. Compute improvement refactor checklist (post-implementation)

Status: Planned

Goal:
- Apply the compute/Acero patterns in `compute_improvement_deepdive.md` to remove
  Python loops, centralize list explode logic, and standardize expression/compute
  vocab usage across the new symbol gap pipelines.

### File: `src/codeintel/build/tabular/expr_vocab.py`

Checklist:
- Add a minimal expression vocabulary for pushdown-safe filters and projections.
- Use this in dataset scans and Acero filter/project nodes.

Representative code pattern:
```python
class E:
    @staticmethod
    def field(name: str) -> pc.Expression:
        return pc.field(name)

    @staticmethod
    def scalar(value: object) -> pc.Expression:
        return pc.scalar(value)

    @staticmethod
    def in_(expr: pc.Expression, values: list[object]) -> pc.Expression:
        return expr.isin(values)
```

### File: `src/codeintel/build/tabular/explode_ops.py`

Checklist:
- Add a canonical list explode helper using `list_parent_indices` + `list_flatten`.
- Reuse for any list->edges expansion in CPG or symbol tooling.

Representative code pattern:
```python
def explode_edges(table: pa.Table, *, src_col: str, dst_list_col: str) -> pa.Table:
    parent_idx = require_array(
        call_compute("list_parent_indices", [table[dst_list_col]]),
        name="list_parent_indices",
    )
    dst_flat = require_array(
        call_compute("list_flatten", [table[dst_list_col]]),
        name="list_flatten",
    )
    src_rep = require_array(call_compute("take", [table[src_col], parent_idx]), name="take")
    return pa.table({"src_id": src_rep, "dst_id": dst_flat})
```

### File: `src/codeintel/build/hamilton/native/ingestion/scip.py`

Checklist:
- Build external symbol derivation as an Acero plan (project -> aggregate -> hashjoin).
- Use projection pushdown to scan only `repo/commit/symbol` where possible.

Representative code pattern:
```python
symbols = acero.Declaration(
    "project",
    acero.ProjectNodeOptions(
        expressions=[pc.field("repo"), pc.field("commit"), pc.field("symbol")],
        names=["repo", "commit", "symbol"],
    ),
    inputs=[acero.Declaration("table_source", acero.TableSourceNodeOptions(occ_table))],
)
distinct = acero.Declaration(
    "aggregate",
    acero.AggregateNodeOptions(keys=[pc.field("repo"), pc.field("commit"), pc.field("symbol")]),
    inputs=[symbols],
)
missing = acero.Declaration(
    "hashjoin",
    acero.HashJoinNodeOptions(
        join_type="left anti",
        left_keys=["repo", "commit", "symbol"],
        right_keys=["repo", "commit", "symbol"],
    ),
    inputs=[distinct, info_table],
)
```

### File: `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

Checklist:
- Use compute helpers for `ends_with`/`equal` to avoid unsupported kernels.
- Keep joins normalized and use `how="anti"` for left-anti semantics.

Representative code pattern:
```python
unknown_mask = or_kleene(
    require_array(call_compute("ends_with", [edges["dst_binding_id"], ":unknown"]), name="ends_with"),
    equal_mask(edges["kind"], "UNKNOWN"),
)
unknown = safe_filter(edges, unknown_mask)
```

### File: `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`

Checklist:
- Replace row loops with vectorized kernels for edge_kind selection.
- Use `call_compute("coalesce")` for dst node fallback.

Representative code pattern:
```python
edge_kind = require_array(
    call_compute("if_else", [is_def, pa.scalar("DEFINES"), pa.scalar("REFERS_TO")]),
    name="if_else",
)
dst_id = require_array(
    call_compute("coalesce", [joined["dst_cpg_node_id"], joined["dst_cpg_node_id_ext"]]),
    name="coalesce",
)
```

### File: `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`

Checklist:
- Coalesce internal/external IDs via compute helper rather than Python loops.
- Prefer struct payloads over per-row JSON building.

Representative code pattern:
```python
src_id = require_array(
    call_compute("coalesce", [joined["src_cpg_node_id"], joined["src_cpg_node_id_ext"]]),
    name="coalesce",
)
joined = joined.set_column(joined.schema.get_field_index("src_cpg_node_id"), "src_cpg_node_id", src_id)
```

### File: `src/codeintel/build/analytics/py_cpg_quality_report.py`

Checklist:
- Use masks + filters for edge counts instead of row loops.
- Join edge->node kind once, then compute rates with vectorized masks.

Representative code pattern:
```python
symbol_mask = and_kleene(
    equal_mask(edges["edge_layer"], "SYMBOL"),
    is_in_mask(edges["dst_node_kind"], value_set=["SCIP_SYMBOL", "SCIP_SYMBOL_EXTERNAL"]),
)
symbol_edge_count = edges.filter(symbol_mask).num_rows
```

### File: `src/codeintel/build/causal_analysis/cpg_edge_integrity.py`

Checklist:
- Drop pandas usage; aggregate counts with iterators and dicts.
- Use iter_rows/iter_array_values to avoid to_pylist.

Representative code pattern:
```python
missing_src = sum(1 for row in iter_rows(edges) if row.get("missing_src"))
```

### File: `src/codeintel/build/causal_analysis/cpg_symbol_destination_audit.py`

Checklist:
- Replace `to_pylist` with iterators for symbol sets and batch scans.

Representative code pattern:
```python
symbols = {
    value
    for value in iter_array_values(symbols_table["symbol"])
    if isinstance(value, str)
}
```

### File: `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`

Checklist:
- Replace `to_pylist` in coverage logging with iterators.

Representative code pattern:
```python
for value in iter_array_values(table.column("match_kind")):
    key = "none" if value is None else str(value)
    counts[key] = counts.get(key, 0) + 1
```

## Acceptance + validation

- `cpg_edge_integrity` reports `missing_dst == 0` for SYMBOL edges.
- All SYMBOL edges still resolve to a node (`SCIP_SYMBOL` internal,
  `SCIP_SYMBOL_EXTERNAL`, or `BINDING_UNRESOLVED`) with provenance metadata.
- Full build completes without new quality report errors.

Recommended commands:
```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run codeintel build run --all --verbose=1
uv run python -m codeintel.build.causal_analysis.cpg_symbol_destination_audit
```
