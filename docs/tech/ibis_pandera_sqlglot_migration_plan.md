# Ibis + Pandera SSOT + SQLGlot Migration Plan

This document describes a phased plan to complete the migration to the go-forward storage architecture:

- Ibis as the query front-end
- DuckDB as the backend storage engine
- SQLGlot-based mutation SQL generation (centralized in `DuckDBPolicyBackend`)
- Pandera-backed `DatasetSchema` as the single source of truth (`SCHEMA_REGISTRY`)

## Goal

End state:

- Ibis is the read/query front-end.
- DuckDB is the backend storage engine.
- All mutations flow through `DuckDBPolicyBackend` (SQLGlot-built).
- All schema + row validation flows through `codeintel.config.datasets.schema_registry.SCHEMA_REGISTRY`
  (Pandera-backed).

Cleanup target:

- Delete (or fully isolate) macro-era, legacy SQL-builder, and duplicate row-model compatibility layers.

## Phase 0 — Lock invariants + add guardrails (no behavior changes)

### Work

- Write down explicit invariants (docs/tech or openspec addendum):
  - No “normalized macros” surface area (`MacroRequirement`, `require_normalized_macros`,
    `requires_normalized_macro`, env flags).
  - Pandera SSOT: no direct imports from `codeintel.storage.pandera_schemas` outside the schema-building
    bridge (and eventually not even there).
  - Mutations: no `con.execute`/`executemany` for DML outside `DuckDBPolicyBackend` / `IbisGateway`
    (allowlist only system/DDL).
- Add guardrail enforcement in the quality gate:
  - `uv run python -m tools.guardrails` (wired into `tools/quality_report.py`)
  - Blocks: `MacroRequirement|require_normalized_macros|requires_normalized_macro`,
    `SafeTable|SafeColumn|QueryBuilder|codeintel.storage.sql`,
    `macro_exists|safe_macro_exists|INGEST_MACRO_TABLES`,
    raw `.con.execute(` outside storage internals/tests.
- Add grep-gates (CI or `tools/quality_report` optional step) with an allowlist:
  - Block: `require_normalized_macros|requires_normalized_macro|MacroRequirement`
  - Block: `SafeTable|SafeColumn|QueryBuilder|codeintel.storage.sql`
  - Block: `macro_exists|INGEST_MACRO_TABLES|safe_macro_exists` (once phased out)
  - Block: `\\.con\\.execute\\(` outside allowlisted storage internals.
- Baseline test command after each phase:
  - `uv run pytest -q -n auto`

### Exit criteria

- Guardrails merged + full test suite still green.

## Phase 1 — Remove normalized-macro compatibility (docs/export/contracts)

### Work

- CLI/handlers:
  - Remove `MacroRequirement` and `--macro-requirement` from `src/codeintel/cli/commands/docs.py`.
  - Remove `MacroRequirement` and related param plumbing from `src/codeintel/cli/handlers/docs.py`.
  - Decide compat strategy:
    - Strict: remove flag entirely (preferred for “100%” migration).
    - Grace: keep flag accepted but no-op + warning for 1 release, then remove.
- Export surface:
  - Remove `require_normalized_macros` parameters and env flags from:
    - `src/codeintel/export/export_jsonl.py`
    - `src/codeintel/export/export_parquet.py`
  - Remove any “deprecated; macro-free” warning logic once the param is gone.
- Dataset capabilities:
  - Remove `requires_normalized_macro` from `src/codeintel/config/datasets/contracts.py`
    (and any serialization/consumers).
- Update tests/docs referencing these flags/fields.

### Exit criteria

- `rg "require_normalized_macros|MacroRequirement|requires_normalized_macro"` returns nothing.
- Docs export + exports still work; tests green.

## Phase 2 — Pandera SSOT via `SCHEMA_REGISTRY` (schema + validation routing)

### Work

- Introduce a single validation API in config-layer (example module):
  - `codeintel.config.datasets.validation`:
    - `get_pandera_schema(table_key) -> pa.DataFrameSchema | None`
    - `validate_df(table_key, df, *, mode=...) -> df`
    - `validate_rows(table_key, rows) -> list[dict[str, object]]`
- Migrate callsites to that API:
  - Ingestion validation (`IngestStorageService._validate_rows`)
  - Analytics row validators (`src/codeintel/analytics/utilities/datasets.py`)
  - Export validation paths
  - Any remaining direct `validate_dataset_df` / `get_dataset_schema` usage
- Make `SCHEMA_REGISTRY` authoritative:
  - Ensure `build_all_schemas()` does not require “storage-layer globals” long-term.
  - Option A (cleanest): move Pandera schema definitions out of `src/codeintel/storage/pandera_schemas.py`
    into `src/codeintel/config/datasets/pandera_schemas.py` (or similar), and have schema_builder build
    from there.
  - Keep `src/codeintel/storage/pandera_schemas.py` temporarily as a thin import-forwarder (with
    deprecation), then delete in Phase 6.

### Exit criteria

- No runtime code depends on `codeintel.storage.pandera_schemas` for lookups/validation.
- `SCHEMA_REGISTRY` is the only entrypoint for schema/validation in production code.

## Phase 3 — Row-model dedupe (single source + generated types)

### Work

- Choose one canonical row-model namespace (recommend: config-layer):
  - Canonical: `src/codeintel/config/datasets/rows/*`
  - Decommission: `src/codeintel/storage/gateway/rows/*`
- Implement generation per openspec Phase 3:
  - Generate TypedDicts (and optional tuple serializers) from Pandera/DatasetSchema:
    - e.g. `codeintel.config.datasets.generated_rows`
  - Provide stable aliases for existing names (`GoidRow`, `CallGraphEdgeRow`, etc.) so callsites don’t
    churn.
- Migrate imports:
  - `src/codeintel/storage/gateway/accessors.py`
  - tests and helper seed modules
- Delete the non-canonical row-model tree.

### Exit criteria

- Only one row-model tree remains, or all row models are generated + re-exported from one place.
- Tests green.

## Phase 4 — SQL builder decommission + “sqlglot mutations only”

### Work

- Make `DuckDBPolicyBackend` the only mutation builder and ensure it is SQLGlot-based:
  - Replace any remaining hand-built DML SQL in `src/codeintel/storage/duckdb_policy_backend.py` with
    SQLGlot AST → SQL.
  - Ensure parameterization conventions are consistent (`?` placeholders, no string interpolation of
    values).
- Remove legacy SQL builder stack:
  - Replace remaining usage of:
    - `SafeTable`, `SafeColumn`, `QueryBuilder` (`src/codeintel/storage/sql/primitives.py`)
    - `src/codeintel/config/datasets/sql.py` (and its re-exports)
    - `src/codeintel/storage/sql/builder.py` (especially `prepared_statements_dynamic`)
  - Exports should compile from Ibis expressions (and/or backend policy helpers), not from “prepared
    SQL builder” modules.
- Keep only minimal quoting/utilities if truly needed, but locate them under the policy backend (or a
  small `storage/sql_utils.py`) and do not expose them as a public API.

### Exit criteria

- `rg "SafeTable|SafeColumn|QueryBuilder|codeintel.storage.sql|codeintel.config.datasets.sql"`
  returns nothing.
- `DuckDBPolicyBackend` owns DML generation and uses SQLGlot for it.

## Phase 5 — Eliminate raw SQL outside storage internals (reads + writes)

### Work

- Inventory `con.execute`/`con.executemany` outside allowlisted modules and classify:
  - Reads → migrate to `gateway.ibis.table(...).filter(...).select(...).execute()`
  - Writes/deletes → migrate to `gateway.policy.*` (or `gateway.ibis.delete` where appropriate)
  - DuckDB system queries (rare) → allow `gateway.ibis.con.raw_sql` behind a tiny helper in storage.
- Priority targets (highest leverage first):
  - Serving backends using raw SQL for existence/count queries
  - Graph adapters/resources using `con.execute` directly
  - Analytics plugins that still do direct mutations

### Exit criteria

- Grep-gate for `con.execute(` outside allowlist passes.
- Tests green.

## Phase 6 — Delete legacy/compat modules (after callsites are gone)

### Work

- Delete macro-era leftovers (once unused):
  - `src/codeintel/ingestion/infrastructure/macros.py`
  - `safe_macro_exists` in `src/codeintel/ingestion/infrastructure/db_queries.py` (and exports in
    `src/codeintel/ingestion/infrastructure/__init__.py`)
- Delete Pandera legacy module once Phase 2 is complete:
  - `src/codeintel/storage/pandera_schemas.py`
- Delete any remaining compatibility shims in ingestion storage if no longer needed:
  - `src/codeintel/ingestion/adapters/duckdb_storage.py` (and `IngestStorageService`) only after
    graphs/ingestion plugins call policy/ibis directly.

### Exit criteria

- Dead modules removed, imports updated, docs updated, full suite green.
