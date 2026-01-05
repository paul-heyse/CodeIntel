# Core Utilities Independence Plan

## Context and decisions
- Core must be a dependency-free utilities layer that both build and storage use.
  The direction is: core -> build/storage/serving, never the reverse.
- Best-effort JSON parsing is acceptable for boundary handling.
- Export serialization stays export-only; internal pipelines should use Parquet/PyArrow.

## Goals
- Remove all core imports of codeintel.build, codeintel.storage, and codeintel.serving.
- Consolidate duplicate utilities (SQLGlot, filters, JSON helpers, path normalization).
- Establish a clear boundary between internal Arrow/Parquet data flow and export JSON.
- Improve maintainability and extensibility with strict module ownership and layering.

## Non-goals
- Rewriting all JSON columns in DuckDB immediately.
- Changing external API semantics beyond necessary relocations and re-exports.
- Re-architecting the build or serving pipeline beyond dependency direction.

## Current state summary (problem areas)
- Core implements storage-backed services:
  - codeintel.core.catalog.service
  - codeintel.core.datasets.registry
  - codeintel.core.datasets.manifest_index
  - codeintel.core.registry.service
- Core imports storage/build/serving types (even type-only) in several places.
- Duplicate utilities exist in core and storage:
  - sqlglot_tools, filter_compiler, duckdb_types, query_results
  - json helpers and json normalization logic
- Path normalization exists in multiple variants with subtle differences.

## Target architecture (layering)
```
core (pure utilities, protocols, primitives)
  ^
  |
storage (DuckDB, SQLGlot, IO, datasets, persistence)
  ^
  |
build / serving (domain-specific orchestration, registry, views)
```

Core owns:
- Protocols, primitives, validation, hashing, paths, Arrow metadata primitives.
- Pure logic with no storage/build/serving imports.

Storage owns:
- DuckDB-backed implementations, SQLGlot helpers, dataset registry, IO helpers.

Build/serving owns:
- Registry service, view registries, runtime bundles, orchestration logic.

## Workstreams and steps

### Workstream 0: Inventory and guardrails
1. Add a dependency rule check that fails on core importing build/storage/serving.
   - Implement a lightweight AST-based check in tools/ or a ruff rule.
   - Enforce in CI and local quality_report.
2. Add a simple grep-based preflight to the quality report output
   (temporary until the AST check exists).

### Workstream 1: Move storage-backed implementations out of core
1. Move function catalog service to storage.
   - Move codeintel.core.catalog.service -> codeintel.storage.catalog.service.
   - Keep core.protocol, core.function_span, core.span_index in core.
   - Add re-export in storage to preserve public API if needed.
2. Move dataset registry and manifest index to storage.
   - codeintel.core.datasets.registry -> codeintel.storage.datasets.registry.
   - codeintel.core.datasets.manifest_index -> codeintel.storage.datasets.manifest_index.
   - Keep contract primitives in core; storage imports them.
3. Move registry service to build or runtime layer.
   - codeintel.core.registry.service -> codeintel.build.registry.service
     (or codeintel.runtime.registry if build is not appropriate).
   - Update call sites and runtime bundling.
4. Remove type-only imports of storage/build/serving from core where possible.
   - Replace with core protocols or narrow local Protocols.

### Workstream 2: Consolidate SQLGlot and filter utilities
1. Choose core.sqlglot_tools as canonical.
2. Update storage.sqlglot_tools to re-export from core (thin wrapper).
3. Remove duplicated logic in storage.sqlglot_tools.
4. Choose core.queries.filter_compiler as canonical.
5. Update storage.queries.filter_compiler to re-export from core.
6. Ensure all internal call sites use the canonical path.

### Workstream 3: JSON helpers and boundary rules
1. Choose core.helpers.json as canonical for best-effort parsing.
2. Replace row_models JSON normalization with a shared helper from core.helpers.json.
3. Replace storage.helpers.json with re-exports of core.helpers.json.
4. Document JSON parsing as boundary-only (DuckDB JSON columns, ingestion)
   and forbid internal JSON objects in core-level logic.
5. Add a guideline doc or short ADR clarifying Arrow/Parquet as internal format.

### Workstream 4: Arrow metadata and schema utilities
1. Consolidate metadata encoding/decoding through core.columnar.schema_metadata.
2. Ensure arrow_gen, arrow_polars, arrow_ipc all use the shared metadata utilities.
3. Add a single "schema metadata keys" source of truth in core.schemas.arrow_gen.
4. Make export serialization explicitly limited to export artifacts.
   - Review codeintel.core.exports.serialization usage and constrain it.
5. Provide a small validation helper to confirm schema metadata is consistent.

### Workstream 5: Path normalization unification
1. Replace SpanIndex.normalize_path with codeintel.core.paths.normalize_path.
2. Ensure all catalog and file path logic uses the same normalization logic.
3. Add tests for edge cases (./, ../, Windows separators).

### Workstream 6: Correctness fixes in core utilities
1. Fix BaseValidationOptions.with_defaults to respect explicit False.
2. Fix deserialize_value Optional/Union handling in core.serialization.converters.
3. Add targeted unit tests for both behaviors.

### Workstream 7: Update imports and re-exports
1. Update all imports in build/storage/serving to new module locations.
2. Add temporary re-export modules where needed to reduce churn.
3. Deprecate old import paths in a single release cycle with clear warnings.

### Workstream 8: Tests and quality gates
1. Update or add tests for:
   - catalog service relocation
   - dataset registry and manifest index relocation
   - filter compiler canonical path
   - JSON helper behavior
   - path normalization
2. Run quality_report and targeted pytest subsets:
   - tools.quality_report
   - tests covering catalog, datasets, storage queries, and export pipeline

## Migration mapping (initial proposal)
| Old module path | New module path | Notes |
| --- | --- | --- |
| codeintel.core.catalog.service | codeintel.storage.catalog.service | core retains protocol/types |
| codeintel.core.datasets.registry | codeintel.storage.datasets.registry | storage-backed implementation |
| codeintel.core.datasets.manifest_index | codeintel.storage.datasets.manifest_index | serving can import storage |
| codeintel.core.registry.service | codeintel.build.registry.service | confirm final owner |
| codeintel.storage.sqlglot_tools | re-export core.sqlglot_tools | remove duplicate logic |
| codeintel.storage.queries.filter_compiler | re-export core.queries.filter_compiler | canonicalize |
| codeintel.storage.helpers.json | re-export core.helpers.json | best-effort parsing |
| codeintel.core.catalog.span_index.normalize_path | use core.paths.normalize_path | unify behavior |

## Acceptance criteria
- No imports from codeintel.storage/build/serving inside codeintel.core (enforced by check).
- Single canonical implementation for SQLGlot and filter compiler utilities.
- Single JSON parsing helper with explicit boundary-only usage.
- Catalog, dataset registry, and registry service owned by correct layer.
- Internal pipelines use Arrow/Parquet; JSON serialization only for export artifacts.
- All quality gates pass (ruff, pyright, pyrefly, targeted tests).

## Risks and mitigations
- Risk: Relocation breaks imports in downstream modules.
  - Mitigation: temporary re-exports and explicit deprecation warnings.
- Risk: Serving depends on storage and creates a cycle.
  - Mitigation: keep storage below serving; avoid serving -> build -> storage loops.
- Risk: JSON parsing behavior changes during consolidation.
  - Mitigation: snapshot tests for decode_json and normalize_duckdb_json_value.

## Rollout plan
1. Introduce dependency boundary check (no core->storage/build/serving).
2. Move storage-backed implementations and update imports.
3. Consolidate SQLGlot/filter utilities and JSON helpers.
4. Unify Arrow metadata handling and path normalization.
5. Fix correctness issues and add tests.
6. Remove deprecated re-exports after one cycle.

## Open decisions
- Final owner for RegistryService: build or runtime layer.
- Whether serving imports storage directly or via a thin facade.
- Timeline for reducing JSON columns in DuckDB (future scope).
