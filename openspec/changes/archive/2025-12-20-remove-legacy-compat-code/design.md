## Context
Multiple legacy and compatibility layers remain in the codebase even after recent
architecture shifts (schema-generated row models, native Hamilton targets, and
serving/runtime boundary cleanups). These include RowBinding and row migration shims,
profile aliases, validation aliases, storage exception re-exports, DuckDB fallback stubs,
FastMCP import shims, and graph compatibility fields embedded in core parsing models.
These surfaces drift from the current design, hide real boundary violations, and
complicate testing and type guarantees.

## Goals / Non-Goals
- Goals:
  - Remove legacy and compatibility APIs that are no longer supported.
  - Enforce canonical interfaces for schema contracts, configuration, storage errors,
    and serving imports.
  - Ensure build execution uses native Hamilton targets only.
  - Keep core parsing models clean and graph-agnostic.
- Non-Goals:
  - Change dataset schemas, table layouts, or semantic view definitions.
  - Redesign the serving API or introduce new endpoints.
  - Alter query semantics beyond removing legacy aliases and wrappers.

## Decisions
- Decision: Standardize on schema-generated row bindings.
  - Remove `RowBinding` and row migration APIs.
  - Use `SchemaService.get_row_binding()` and `GeneratedRowBinding` everywhere.
  - Update dataset contract builders to surface provenance metadata (table_key,
    schema_hash) and drop legacy adapter wrappers.

- Decision: Build contract enumeration uses source-only providers.
  - Remove `full_declared_schema_provider()` from build layer.
  - Keep full declared provider only in `codeintel.core.schemas.declared` for
    storage bootstrap, not for build contract enumeration.

- Decision: Build execution is native-only.
  - Remove wrapper-based target implementations and allowlist warnings.
  - Ensure `impl_kind` is always "native" for build plan entries.

- Decision: Configuration surfaces are canonical.
  - Remove legacy "default" profile alias.
  - Remove `ValidationResult` alias and rely on `ValidationOutcome` only.

- Decision: Serving uses direct FastMCP imports.
  - Delete the local `_compat` shim and import `FastMCP`, `Context`, and `EventStore`
    directly from fastmcp packages.

- Decision: Storage error surfaces are canonical and DuckDB is required.
  - Remove `codeintel.storage.exceptions` compatibility re-exports.
  - Remove fallback DuckDB exception stubs from `storage.gateway.protocol` and
    require DuckDB at runtime for storage protocol usage.

- Decision: Core parsing models are graph-agnostic.
  - Remove graph compatibility fields from `core.parsing.ParsedFunction`.
  - Keep graph-specific fields in graph parsing ports and adapters.

## Risks / Trade-offs
- Breaking imports for downstream callers relying on legacy aliases and shims.
  - Mitigation: Provide a migration guide in docs and update internal call sites
    before removal.
- Removing DuckDB fallback stubs may break lightweight import scenarios.
  - Mitigation: Confirm DuckDB is a required runtime dependency and document that
    storage modules assume it is installed.
- Native-only targets could expose gaps if any remaining targets are wrapper-only.
  - Mitigation: inventory targets and add native implementations before removal.

## Migration Plan
1. Update schema contract and row binding code to use `GeneratedRowBinding` only.
2. Remove row migration APIs and legacy RowBinding types.
3. Remove build-layer full declared provider and update contract enumeration call sites.
4. Migrate all build targets to native implementations and remove wrapper plumbing.
5. Remove legacy profile alias and ValidationResult alias; update configs and tests.
6. Remove MCP compatibility shim and update serving imports.
7. Remove storage exception compatibility module and DuckDB fallback stubs.
8. Remove graph compatibility fields from core parsing models and update adapters.
9. Update docs, tests, and validation to reflect canonical surfaces.

## Open Questions
- Do any downstream integrations still require the legacy profile alias or row migration
  APIs, and if so should we publish a short-lived migration shim outside the core package?
- Is DuckDB officially required for all runtime environments, or do we need a minimal
  import-only mode that avoids importing `duckdb` in storage protocols?
