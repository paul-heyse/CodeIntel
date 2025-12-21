## 1. Contract Factory + DAG-Free Enumeration
- [x] 1.1 Create `codeintel.core.schemas.contract_factory` with a deterministic
      `build_dataset_contract()` function that accepts:
      - `SchemaService` (for table schema + row binding)
      - table key + optional `OutputContract` metadata (owner, tags, filenames, etc.)
      - view detection + owner package mapping + composition lookup
- [x] 1.2 Refactor `src/codeintel/build/schemas/contract_service.py` to delegate to the
      factory for both schema-only and enriched modes; remove duplicate helpers
      (`is_view`, `_owner_package_from_prefix`, `_derive_contract_from_*`).
- [x] 1.3 Refactor `src/codeintel/storage/contracts/provider.py` to call the core factory
      and drop duplicate mapping/view logic.
- [x] 1.4 Add a DAG-free output inventory helper (e.g., `build/target_inventory.py`) that
      derives `OutputInventory` from `load_target_specs()` without building the Hamilton
      driver, and update:
      - `build/run_context.py` to use it for default `output_inventory`
      - `build/schemas/provider_declared.py` to exclude output table keys without DAG init
- [x] 1.5 Provide test seams for injected `TargetMetadataProvider` and output inventory
      in contract resolution, ensuring schema-only enumeration stays DAG-free.

## 2. Storage-Owned Export Surface (Boundary Enforcement)
- [x] 2.1 Add a storage export service module (e.g., `src/codeintel/storage/exports/service.py`)
      with functions:
      - `build_export_relation()` (wraps DuckDB relation as `ExportRelation`)
      - `write_export_audit()` (DB + optional log file)
      - `audit_enabled()` (settings gate)
- [x] 2.2 Update storage gateway protocols to expose the export service through a
      duckdb-agnostic interface; move duckdb imports under `TYPE_CHECKING` in
      `src/codeintel/storage/gateway/protocol.py` or split into duckdb-free protocol module.
- [x] 2.3 Update build export utilities to use the storage export service instead of
      `gateway.con` or `DuckDBConnection` (e.g., `src/codeintel/build/exports/common.py` and
      `src/codeintel/build/exports/engine.py`).

## 3. Settings Injection (No Hidden Env Reads)
- [x] 3.1 Define canonical settings dataclasses under `src/codeintel/core/config/settings.py`
      (BuildSettings, ServingSettings, HamiltonExecutionSettings, ExportAuditSettings).
- [x] 3.2 Update build settings usage:
      - Remove `get_build_settings()` calls from library code
      - Inject settings through `BuildRunContext` / `BuildEnv` and pass explicitly
        to export/audit helpers
- [x] 3.3 Update serving settings usage:
      - Move `ServingSettings.from_env()` calls to CLI boundary handlers
      - Require explicit settings in `create_serving_app()` / runtime builders
- [x] 3.4 Update Hamilton parallel config to accept injected settings
      (remove env reads from execution path outside CLI).

## 4. Error Payload Unification
- [x] 4.1 Introduce a serving adapter that converts `ErrorResponse` to
      `codeintel.core.errors.problem_details.ProblemDetail` with stable extensions
      (code, kind, retryable, hint, correlation_id).
- [x] 4.2 Replace `serving/http/errors.ProblemDetail` construction with the adapter
      (retain a Pydantic wrapper only for OpenAPI schema generation).
- [x] 4.3 Ensure `CodeIntelDomainError` and `exception_to_error_response()` paths
      yield consistent ProblemDetail payloads for HTTP and MCP.

## 5. Export Format Registry Unification
- [x] 5.1 Create a shared export format registry module (e.g.,
      `src/codeintel/core/exports/formats.py`) with alias handling for `jsonl`/`ndjson`.
- [x] 5.2 Update build exports to use the shared registry for format normalization,
      suffix, and MIME handling.
- [x] 5.3 Update serving export planners to use the shared registry and
      map serving defaults to canonical format IDs.

## 6. Validation + Docs
- [x] 6.1 Add tests for:
      - DAG-free contract enumeration (no Hamilton driver init)
      - Contract factory parity between build and storage
      - Export service boundary (no direct duckdb access in build)
      - Error payload parity between HTTP and MCP
      - Export format alias normalization
- [x] 6.2 Update any relevant docs or developer notes describing the new boundaries
      and settings injection patterns.
- [x] 6.3 Run quality and test gates after implementation:
      - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
      - `uv run pytest -q`
