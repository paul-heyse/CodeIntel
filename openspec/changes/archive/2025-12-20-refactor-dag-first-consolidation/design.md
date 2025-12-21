## Context
The DAG-first architecture in `src/codeintel/build` depends on storage-only concerns in a few
places (export relation creation, audit logging, DuckDB connection types), and dataset contract
construction is duplicated between build (`build/schemas/contract_service.py`) and storage
(`storage/contracts/provider.py`). In addition, schema-only contract enumeration still triggers
Hamilton DAG initialization via `get_target_metadata_service()` in `build/run_context.py` and
`build/schemas/provider_declared.py`, violating the DAG-free contract-resolution requirement.
Serving and build also define parallel error payload models (core ProblemDetail vs serving
ProblemDetail/ErrorResponse) and parallel export format registries (build jsonl/parquet vs
serving ndjson/json/parquet/arrow).

## Goals / Non-Goals
- Goals:
  - Remove DuckDB types and connection access from build/serving modules by routing through
    storage-owned export services and duckdb-agnostic protocols.
  - Provide a single contract factory for DatasetContract derivation used by both build and
    storage, ensuring deterministic, shared defaults.
  - Guarantee schema-only contract enumeration and output inventory resolution are DAG-free.
  - Enforce explicit settings injection (Build/Serving/Hamilton) at runtime boundaries, with
    env resolution confined to CLI entrypoints.
  - Unify error payload representation around the core ProblemDetail model with serving
    adapters and consistent error-code mapping.
  - Centralize export format definitions with alias handling and consistent MIME/suffix
    mapping across build and serving.
- Non-Goals:
  - Redesign Hamilton DAGs, target specs, or analytic outputs.
  - Change dataset schemas or stored table structures.
  - Replace the serving error catalog codes or introduce new error semantics.

## Decisions
- Decision: Introduce a core DatasetContract factory and reuse it everywhere.
  - Create `codeintel.core.schemas.contract_factory` (name TBD) that builds `DatasetContract`
    from a `SchemaService`, a table key, and optional metadata (OutputContract details and
    composition/ownership fields). It owns:
    - view detection (`docs.v_`), base-table tagging, and owner package mapping
    - default JSON Schema IDs and export filenames via `contract_policy`
    - row binding resolution (via `SchemaService.get_row_binding`)
  - Build `SchemaContractService` and `ContractService` become thin wrappers that call the
    factory with schema-only inputs or with injected `TargetMetadataProvider` metadata.
  - Storage `storage.contracts.provider` delegates to the same factory using its schema
    provider + row binding registry.

- Decision: Provide DAG-free output inventory and declared schema providers.
  - Add a new `build/target_inventory.py` (or extend `target_catalog`) to compute
    `OutputInventory` directly from `load_target_specs()` without Hamilton driver
    initialization.
  - Update `build/run_context.py` and `build/schemas/provider_declared.py` to use the
    DAG-free inventory for `exclude_table_keys`, removing calls to
    `get_target_metadata_service()` in schema-only paths.
  - Keep `TargetMetadataProvider` for metadata enrichment only, with lazy injection via
    `ContractResolutionSettings` and `LazyTargetMetadataProvider`.

- Decision: Move export relation construction and audit logging into storage.
  - Introduce a storage-owned export service (e.g., `codeintel.storage.exports.service`)
    providing:
    - `build_export_relation(gateway, expr | table_key, limit, offset) -> ExportRelation`
    - `write_export_audit(gateway, record, audit_settings)`
    - `audit_enabled(audit_settings)`
  - Update build exports to call the service instead of `gateway.con` and to pass explicit
    audit settings (no direct DuckDB connection usage in build).
  - Update `StorageGateway`/`MinimalGateway` protocols to expose the export service through
    a duckdb-agnostic interface; move any duckdb imports in protocol modules under
    `TYPE_CHECKING` to avoid runtime imports outside storage.

- Decision: Consolidate settings into explicit runtime bundles.
  - Define canonical settings dataclasses under `codeintel.core.config.settings` (names TBD):
    - `BuildSettings` (engine_version, export_audit_*)
    - `ServingSettings` (current serving config)
    - `HamiltonExecutionSettings` (parallel backend, max workers)
    - Optional `ExportAuditSettings` for storage audit logging
  - Remove implicit environment lookups from library code. `from_env()` helpers remain only
    in CLI boundary modules. `create_serving_app()` and `build_runtime()` require explicit
    settings injection or accept a pre-built `ServingRuntime`.
  - Inject settings via `BuildRunContext` -> `BuildEnv` and optionally register them in
    `ConfigRegistry`/`ConfigProvider` for plugin access.

- Decision: Unify ProblemDetail payloads across build/serving.
  - Adopt `codeintel.core.errors.problem_details.ProblemDetail` as the canonical payload.
  - Provide a serving adapter to convert `ErrorResponse` (catalog mapping) into core
    `ProblemDetail` with explicit extensions (code, kind, retryable, hint, correlation_id).
  - Keep a Pydantic model for OpenAPI generation, but construct it from the core
    ProblemDetail to prevent drift.

- Decision: Create a shared export format registry with alias support.
  - Move/duplicate `serving/export/formats.py` into a shared module (e.g.,
    `codeintel.core.exports.formats`) and make build/serving import from it.
  - Treat `ndjson` as an alias of `jsonl` at the registry level; expose surface-specific
    defaults without changing build artifacts (build continues to emit `.jsonl`).
  - Add helper functions for MIME type, suffix resolution, and preview capability based on
    canonical format IDs.

## Risks / Trade-offs
- Risk: Changing protocol imports may require careful typing adjustments across build/storage.
  - Mitigation: Keep duckdb imports behind `TYPE_CHECKING` and provide alias types for runtime.
- Risk: Settings injection changes may require updates to CLI entrypoints and tests.
  - Mitigation: Provide compatibility wrappers (deprecated) to minimize breakage and stage
    removals.
- Risk: Error payload unification may impact OpenAPI schema outputs.
  - Mitigation: Keep a serving-specific Pydantic adapter, but back it with core ProblemDetail.
- Risk: Export format alias handling might change client expectations.
  - Mitigation: Preserve existing defaults (build `.jsonl`, serving `ndjson`) and document
    alias behavior.

## Migration Plan
1. Add core contract factory + DAG-free output inventory helpers.
2. Update build/storage contract providers to use the factory and new inventory provider.
3. Add storage export service and update build export modules to use it.
4. Introduce settings bundles and update runtime builders/entrypoints to inject settings.
5. Implement ProblemDetail adapters and switch serving HTTP/MCP responses to canonical model.
6. Move export format registry to shared module; update build/serving usage.
7. Add tests and documentation updates; remove deprecated paths after validation.

## Open Questions
- Should `StorageGateway` expose a dedicated `exports` accessor or should export helpers live
  as module functions bound to a `MinimalGateway`?
- Do we want to preserve `ServingSettings.from_env()` for interactive usage, or move all env
  resolution into CLI handlers only?
- For export formats, should serving keep `.ndjson` file suffix, or should we align suffixes
  and MIME types with build `.jsonl` outputs?
