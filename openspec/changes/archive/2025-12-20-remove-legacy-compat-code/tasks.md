## 1. Schema Contracts and Row Bindings
- [x] 1.1 Replace DatasetContract row binding usage with GeneratedRowBinding, including
      provenance metadata (table_key, schema_hash).
- [x] 1.2 Remove `RowBinding` and update all imports/call sites to use
      `codeintel.core.schemas.row_models.GeneratedRowBinding`.
- [x] 1.3 Delete `build/hamilton/contracts/schemas/row_migration.py` and remove any
      remaining call sites.

## 2. Contract Resolution Providers
- [x] 2.1 Remove `full_declared_schema_provider()` from
      `src/codeintel/build/schemas/provider_declared.py`.
- [x] 2.2 Update build-layer contract enumeration to use source-only declared
      schema providers and keep full declared providers only in core utilities.

## 3. Build Execution (Native-Only)
- [x] 3.1 Inventory wrapper-based targets and implement native equivalents where missing.
- [x] 3.2 Remove wrapper allowlist and wrapper fallback logic from build planning
      (`impl_kind` should always resolve to native).
- [x] 3.3 Delete or archive wrapper templates that are no longer referenced.

## 4. Configuration and Option Aliases
- [x] 4.1 Remove legacy execution profile alias "default" and update any configuration
      references to use canonical profile names.
- [x] 4.2 Remove `ValidationResult` alias from options protocol and update any references
      to use `ValidationOutcome`.

## 5. Serving MCP Imports
- [x] 5.1 Delete `src/codeintel/serving/mcp/_compat.py`.
- [x] 5.2 Update all serving MCP modules to import FastMCP, Context, and EventStore
      directly from fastmcp packages.

## 6. Storage Boundary Cleanup
- [x] 6.1 Remove `src/codeintel/storage/exceptions.py` and update imports to use
      canonical error types (`codeintel.core.errors.storage`,
      `codeintel.storage.duckdb_types`, or `storage.gateway.protocol`).
- [x] 6.2 Remove DuckDB fallback exception stubs from
      `src/codeintel/storage/gateway/protocol.py` and require DuckDB at runtime.

## 7. Core Parsing Model Split
- [x] 7.1 Remove graph compatibility fields from `core.parsing.ParsedFunction`.
- [x] 7.2 Update graph adapters to populate graph-specific parsing models instead of
      core parsing models.

## 8. Tests and Documentation
- [x] 8.1 Add tests to ensure legacy import paths are absent and canonical interfaces
      are used.
- [x] 8.2 Update any docs or guides that reference legacy profiles, row migration,
      or compatibility shims.

## 9. Validation
- [x] 9.1 Run quality gates and tests after implementation:
      - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
      - `uv run pytest -q`
