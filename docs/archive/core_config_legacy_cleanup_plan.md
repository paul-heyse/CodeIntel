# Core/Config Legacy Cleanup Plan (Design-Phase, Breaking Changes OK)

## Context and Decisions
- We are in a design phase with a single maintainer; breaking changes are acceptable.
- The goal is to remove unused or compatibility-only surfaces in `core`/`config`.
- We will prefer a smaller, explicit public API over unused aggregators and fallbacks.
- Any legacy compatibility paths must be removed or converted into explicit, enforced behavior.

## Goals
- Remove unused configuration builders/registries and align to a single config access path.
- Remove unused plugin registry abstractions and unused serialization/ports protocols.
- Remove legacy compatibility toggles (e.g., telemetry attribute back-compat).
- Eliminate cross-layer fallbacks (e.g., config datasets using build-owned schema service).
- Standardize manifest versioning (v2-only) and remove all deprecation metadata in core/config.

## Non-goals
- No new features unrelated to legacy/compatibility removal.
- No backward compatibility shims for external consumers.

## Success Criteria
- No production code depends on `ConfigBuilder`, `ConfigRegistry`, or `ConfigAccessor`.
- Core plugin registry (`core/plugins/registry`) is removed or fully adopted (we will remove).
- Unused serialization and ports protocol layers are removed.
- Legacy observability flags are removed from settings, loader, and telemetry wiring.
- `config/datasets/columns.py` does not import build-owned modules.
- Manifest versioning is v2-only and explicit.
- No deprecation or legacy metadata surfaces remain in core/config models or schemas.
- Quality gates pass and targeted tests are updated.

## Status Snapshot (Latest)
- W1-W8: Implemented (ConfigBuilder removal, registry/accessor removal, plugin registry removal,
  serialization/ports cleanup, aggregator alignment, observability flag removal, schema fallback removal).
- W9: Superseded by the new mandate to delete deprecation/legacy metadata end-to-end in core/config.
- W10: Implemented in code (v2-only compile/parse/schema updates); remaining work is fixture coverage,
  quality gates, and any lingering v1 references in tests/docs.

## Workstreams

### W0: Inventory and Impact Confirmation
**Objective**: Validate that the surfaces below are still unused by production code.

**Steps**
1. Confirm production usage is empty (or test-only) for each target module:
   - `rg -n "ConfigBuilder|ConfigRegistry|ConfigAccessor" src`
   - `rg -n "core\\.plugins\\.registry|BasePluginRegistry|PluginPlan" src`
   - `rg -n "SerializableBase|SerializableProtocol|BaseQueryResult|BaseBatchResult" src`
2. Confirm only tests import `codeintel.config` top-level:
   - `rg -n "from codeintel\\.config import|import codeintel\\.config" src tests`
3. Confirm schema service fallback in config columns:
   - `rg -n "get_schema_service|lazy_getattr\\(" src/codeintel/config/datasets/columns.py`
4. Confirm manifest version handling:
   - `rg -n "SchemaManifest|version == \\\"v1\\\"|is_v2" src`

**Outputs**
- A final edit/delete checklist with file paths and tests to update/remove.

**Acceptance**
- All references match the recommendation list and remain test-only or unused in `src/`.

---

### W1: Remove `ConfigBuilder` + Top-Level `codeintel.config` Aggregator
**Objective**: Delete the builder entrypoint and reduce `codeintel.config` to direct primitives.

**Steps**
1. Remove `src/codeintel/config/builder.py`.
2. Update `src/codeintel/config/__init__.py`:
   - Remove `ConfigBuilder` and `BuilderDependencies`.
   - Remove lazy import logic for the builder.
   - Update docstring examples to use `SnapshotInit` + `BuildLayoutOptions` directly.
3. Update tests that used `ConfigBuilder`:
   - `tests/test_tests_analytics_unit.py`
   - `tests/_helpers/configs/coverage_config.py`
   - `tests/_helpers/configs/graph_config.py`
4. Replace `ConfigBuilder.from_snapshot(...)` patterns with:
   - `snapshot_ref = SnapshotInit(...).to_snapshot_ref()`
   - `paths = BuildLayoutOptions().materialize(snapshot_ref.repo_root, ...)`
   - Use `RuntimeInputs` + `build_runtime_primitives(...)` when a runtime bundle is required.

**Acceptance**
- No `ConfigBuilder` imports remain in `src/` or `tests/`.
- `codeintel.config` exports only primitives and CLI boundary models.

---

### W2: Remove `ConfigRegistry` + `ConfigAccessor`
**Objective**: Keep a single concrete config implementation (`ConfigProvider`) for plugins.

**Steps**
1. Remove `src/codeintel/core/config/registry.py` and
   `src/codeintel/core/config/accessor.py`.
2. Update `src/codeintel/core/plugins/execution/context.py`:
   - Replace `configs: ConfigAccessor` with `configs: ConfigProvider`.
   - Remove `ConfigAccessor` type-only import.
3. Update `src/codeintel/core/config/__init__.py`:
   - Export only settings dataclasses (BuildSettings, ServingSettings, etc.).
4. Update `src/codeintel/core/__init__.py`:
   - Remove exports for `ConfigRegistry` and `ConfigAccessor`.
   - Update module docstring to reflect actual subpackages.
5. Remove config registry tests:
   - `tests/core/config/test_config_registry.py`
   - `tests/core/config/test_accessor.py`
   - Adjust `tests/core/conftest.py` fixtures that reference registry/accessor.

**Acceptance**
- No references to `ConfigRegistry` or `ConfigAccessor` remain.
- Plugin contexts rely on `ConfigProvider` only.

---

### W3: Remove Core Plugin Registry Infrastructure
**Objective**: Eliminate unused registry base classes and exports in core plugins.

**Steps**
1. Remove `src/codeintel/core/plugins/registry/base.py`,
   `src/codeintel/core/plugins/registry/sorting.py`,
   and `src/codeintel/core/plugins/registry/__init__.py`.
2. Update `src/codeintel/core/plugins/__init__.py`:
   - Stop exporting registry-related symbols.
   - Keep execution context and types exports only.
3. Update any docs referencing plugin registries in core.
4. Remove any registry-related tests if present.

**Acceptance**
- No registry classes remain under `core/plugins/registry`.
- No imports of registry types in `src/`.

---

### W4: Remove Serialization Base/Protocol Layer
**Objective**: Keep only the JSON conversion helpers that are actively used.

**Steps**
1. Remove `src/codeintel/core/serialization/base.py` and
   `src/codeintel/core/serialization/protocol.py`.
2. Update `src/codeintel/core/serialization/__init__.py` to export only:
   - `serialize_value`, `serialize_dataclass_to_dict`, and related converter helpers.
3. Update any documentation that referenced `SerializableBase` or
   `SerializableProtocol`.

**Acceptance**
- No references to `SerializableBase` or `SerializableProtocol` remain.
- CLI result serialization still uses `serialize_dataclass_to_dict`.

---

### W5: Remove Port Result Protocols
**Objective**: Use concrete result types (`QueryResult`, `BatchResult`) only.

**Steps**
1. Remove `src/codeintel/core/ports/results.py`.
2. Update `src/codeintel/core/ports/__init__.py` to stop exporting
   `BaseQueryResult` and `BaseBatchResult`.
3. Update `src/codeintel/ingestion/ports/storage.py` to refer to
   `QueryResult` and `BatchResult` directly.
4. Update any docs in `ingestion/ports` that mention the base protocols.

**Acceptance**
- No references to `BaseQueryResult` or `BaseBatchResult` remain.
- Ingestion port types rely on concrete result classes.

---

### W6: Clean Up Aggregator Modules and Documentation
**Objective**: Ensure top-level package docs/export lists match reality.

**Steps**
1. Align `src/codeintel/core/__init__.py` docstring with actual subpackages.
   - Remove mentions of `recipes` and `types` if they do not exist.
2. Align `src/codeintel/core/config/__init__.py` exports to settings only.
3. Align `src/codeintel/config/__init__.py`:
   - Export only primitives + CLI boundary models.
   - Remove lazy import logic for removed builder.

**Acceptance**
- No stale references to removed modules in top-level docs.
- `__all__` lists reflect actual public API.

---

### W7: Remove Legacy Observability Flag
**Objective**: Remove `duckdb_emit_legacy_db_attributes` and its env var hook.

**Steps**
1. Remove field from `src/codeintel/core/config/settings.py`.
2. Remove env var from `src/codeintel/core/runtime/loader.py`
   (`CODEINTEL_OTEL_DB_LEGACY_ATTRIBUTES`).
3. Remove flag usage in:
   - `src/codeintel/observability/otel.py`
   - `src/codeintel/observability/duckdb_tracing.py`
4. Update docs or tests referencing the env var.
   - Use `rg -n "LEGACY_ATTRIBUTES|duckdb_emit_legacy_db_attributes"`.

**Acceptance**
- No references to `duckdb_emit_legacy_db_attributes` remain.
- Observability behavior is fixed to the modern attribute set.

---

### W8: Remove Schema Service Fallback in Config Columns
**Objective**: Eliminate build-owned fallback from `config/datasets/columns.py`.

**Steps**
1. Update `src/codeintel/config/datasets/columns.py`:
   - Remove lazy import fallback to `codeintel.build.schemas.service`.
   - Rely solely on `core.schemas.service.get_schema_service()`.
2. Ensure `SchemaService` is always configured before columns are requested:
   - Set via `set_schema_service(...)` in runtime/bootstrap paths that
     currently rely on the fallback.
   - Use `rg -n "set_schema_service" src` to identify existing init points.

**Acceptance**
- No build-owned imports in `config/datasets/columns.py`.
- Runtime fails fast if schema service is not configured.

---

### W9: Enforce Deprecation Metadata Semantics
**Objective**: Remove deprecation metadata and legacy compatibility surfaces in core/config.

**Steps**
1. Delete dataset deprecation fields from core schemas:
   - Remove `deprecated`, `deprecation_message`, and `deprecation_warning()` from
     `src/codeintel/core/schemas/contract_primitives.py`.
   - Remove `deprecated`/`deprecation_message` from `src/codeintel/core/schemas/contract_factory.py`.
   - Remove deprecation serialization/deserialization from
     `src/codeintel/core/schemas/contract_serde.py`.
2. Remove deprecation validation in core:
   - Delete `_validate_deprecations(...)` and its call site in
     `src/codeintel/core/schemas/contract_validation.py`.
3. Remove deprecation tags and semantic hints in core Hamilton tagging:
   - Remove `TAG_DEPRECATED` and `TAG_REPLACED_BY` (and any encoding helpers) from
     `src/codeintel/core/hamilton/semantic_tags.py`.
4. Remove deprecation fields from config schemas:
   - Remove `deprecated` and `replaced_by` properties from the semantic registry schema
     generation used for serving artifacts.
   - Ensure the schema no longer enforces `replaced_by` when deprecated.
5. Remove remaining deprecation mentions in core/config docs and public exports:
   - Update `src/codeintel/config/__init__.py` and any docstrings that reference
     deprecated or legacy step configurations.
6. Update tests/fixtures tied to core/config deprecation metadata:
   - `tests/config/test_datasets_schema_builder.py`
   - `tests/config/test_datasets_contracts.py`
   - Any fixtures under `tests/_helpers/contracts.py` or other core/config helpers
     that set `deprecated`/`deprecation_message` fields.
7. Sweep and verify:
   - `rg -n "deprecat|replaced_by" src/codeintel/core src/codeintel/config` returns zero results.

**Acceptance**
- No `deprecated`, `replaced_by`, or `deprecation_message` fields exist in core/config models or schemas.
- No deprecation-related tags remain in core Hamilton metadata.
- Core/config tests and fixtures compile and run without deprecation fields.

**Follow-ups (Outside Core/Config)**
- After removing core/config deprecation metadata, update build/serving layers that
  referenced those fields (semantic registry compilation, serving responses, and export metadata).

---

### W10: Standardize Schema Manifests on v2
**Objective**: Make v2 the only supported manifest version.

**Steps**
1. Update `src/codeintel/core/manifests.py` docs to remove v1 language.
2. Update manifest parsing (CLI or serving) to reject v1 explicitly.
3. Update schema manifest generation to always emit v2.
4. Remove any v1-specific tests or fixtures.

**Acceptance**
- v1 manifests are rejected with a clear error.
- All generated manifests are v2.

---

## Execution Order (Recommended PRs)
1. W1 + W2 (config surface removal).
2. W3 + W4 + W5 (plugin/serialization/ports cleanup).
3. W6 (doc + aggregator alignment).
4. W7 + W8 (legacy toggle + schema fallback removal).
5. W9 + W10 (deprecation removal + manifest standardization).

## Validation Checklist
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted tests (by touched areas):
  - `tests/core/` for config/resource/serialization changes.
  - `tests/build/` for schema manifest and build pipeline changes.
  - `tests/config/` for contract/schema updates and fixtures.
  - `tests/serving/` only if schema manifest changes touch serving adapters.

## Notes for Implementation
- When removing modules, delete corresponding tests and update any fixtures.
- Keep changes isolated per workstream to limit regressions.
- Avoid any new legacy or compatibility shims; prefer hard errors.
