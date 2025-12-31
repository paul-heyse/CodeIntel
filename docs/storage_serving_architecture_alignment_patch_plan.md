# Storage/Serving Architecture Alignment Patch Plan

## Scope
Translate the approved design changes into a concrete, ordered implementation plan that resolves
all outstanding Pyright/Pyrefly errors while aligning with:
- `docs/storage_serving_architecture_alignment_plan.md`
- `docs/hamilton_inference_first_implementation_plan.md`

## Phase 1: DuckDB relation scan adapter (typing + backbone enforcement)
Goal: Preserve DuckDB as the execution backbone and satisfy strict typing for relation scans.

Changes
- Add a typed adapter for `DuckDBPyConnection.from_parquet` and `from_arrow` to eliminate
  `dict[str, object]` call sites and provide explicit, typed signatures.
- Use the adapter inside `src/codeintel/serving/semantic/duckdb_relation_builder.py` so
  projection/pushdown is explicit and type-safe.

Files
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- New helper module (e.g. `src/codeintel/serving/semantic/duckdb_scan_adapter.py`)

Acceptance
- Pyright/Pyrefly `from_parquet` overload errors resolved.
- Serving still constructs relations via SQLGlot AST -> Expression API only.

## Phase 2: Inference-first aggregation helpers (build-time alignment)
Goal: Ensure inference-first pipelines avoid materialization and keep aggregation type-safe.

Changes
- Introduce a small helper that calls `relation.aggregate(*aggs, by)` with varargs
  instead of `list[str]`, then replace current aggregate call sites.

Files
- `src/codeintel/analytics/compute/coverage/compute.py`
- `src/codeintel/analytics/profiles/graph_features.py`
- Optional shared helper module (e.g. `src/codeintel/analytics/duckdb_helpers.py`)

Acceptance
- Pyright/Pyrefly aggregate argument errors resolved.
- No change in aggregation semantics.

## Phase 3: Graph goids typing and symbol-use alignment
Goal: Fix type safety around AST-derived nodes and symbol-use mapping while preserving
inference-first design (observed data remains authoritative).

Changes
- Narrow `node_type` to `str` before calling `_resolve_start_line`, `_resolve_qualname`,
  and `determine_kind`.
- Ensure `dataclasses.asdict` only receives dataclass instances (guard/cast).
- Align `build_use_edges` signature with actual mapping (tuple value including line),
  or strip line numbers before passing if not needed.

Files
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`

Acceptance
- Pyright/Pyrefly argument errors resolved.
- No change in graph output semantics.

## Phase 4: Contract resolver alignment (DuckDB authoritative)
Goal: Replace Arrow schema contract resolution with DuckDB metadata per architecture plan.

Changes
- Implement DuckDB-backed contract resolver (metadata tables + information_schema).
- Deprecate or thin-wrap `src/codeintel/serving/semantic/schema_contracts.py`.
- Update engine call sites to use the DuckDB resolver.

Files
- `src/codeintel/serving/semantic/schema_contracts.py` (deprecate/replace)
- `src/codeintel/serving/semantic/duckdb_contracts.py` (new)
- `src/codeintel/serving/semantic/engines/duckdb_engine.py`
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`

Acceptance
- Serving never consults Arrow metadata directly for contracts.
- Contract types are obtained from DuckDB catalog + metadata tables.

## Phase 5: Minimal gateway protocol alignment
Goal: Allow observation tracking to use minimal gateway without violating protocol typing.

Changes
- Narrow `SchemaCatalogTracking.__init__` to `MinimalGateway` (or new minimal protocol)
  since it only uses `con` + `policy`.

Files
- `src/codeintel/storage/tracking/schema_catalog.py`
- `src/codeintel/storage/gateway/factory.py`
- `src/codeintel/storage/gateway/protocol.py` (if new protocol needed)

Acceptance
- Pyright/Pyrefly `MinimalStorageGateway` incompatibility resolved.
- No runtime behavior changes.

## Phase 6: Export metrics + fingerprint consistency
Goal: Align serving export metadata with canonical AST and fingerprints.

Changes
- Standardize `export_fingerprint` return to include `(query_hash, schema_hash,
  ast_fingerprint, sql_fingerprint)`.
- Make `ExportMetricsContext` optional fingerprint fields default to `None`,
  or update call sites to pass them.

Files
- `src/codeintel/serving/http/export_dispatch.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/mcp` and test fixtures

Acceptance
- Pyright/Pyrefly errors in export tests resolved.
- Fingerprint fields match canonical AST requirements from Phase 1/2.

## Phase 7: Remove legacy SQLGlot view registry usage
Goal: Enforce Hamilton as the canonical view source (Phase 3 of alignment plan).

Changes
- Replace `codeintel.storage.views.sqlglot_views` usage in tests/helpers with
  Hamilton-derived registry artifacts.
- Remove references to `view_plan_map` / SQLGlot view registries.

Files
- `tests/_helpers/docs_views.py`
- `tests/build/serving/test_pr84_semantic_view_hamilton_tags.py`
- `tests/build/serving/test_pr85_semantic_registry_compiles_from_tags.py`
- `tests/storage/views/test_pr86_no_ibis_view_registry.py`

Acceptance
- Pyright/Pyrefly missing import errors resolved.
- Tests validate Hamilton outputs instead of legacy view registry.

## Phase 8: Routing + Polars test updates
Goal: Ensure routing tests and query builder tests match the DuckDB-first architecture.

Changes
- Remove `ast_supports_polars` expectations; routing should prefer only DuckDB.
- Replace `polars_query_builder` test imports with DuckDB relation builder
  (`apply_query_ast` / `build_relation_plan`).
- Update any test calling `EngineContext` with removed params.

Files
- `src/codeintel/serving/semantic/routing.py`
- `tests/serving/semantic/test_routing.py`
- `tests/serving/semantic/test_query_builder.py`

Acceptance
- Pyright/Pyrefly missing import/attribute errors resolved.
- Tests reflect the canonical AST + DuckDB relation plan.

## Validation Steps
- `uv run ruff check src/codeintel/serving src/codeintel/storage`
- `uv run pyright --warnings --pythonversion=3.13`
- `uv run pyrefly check`

## Notes
- No new tests are required in this plan; existing tests should be updated to reflect
  the new architecture direction.
- All changes avoid raw SQL templates in serving paths and preserve SQLGlot AST
  as the canonical IR.
