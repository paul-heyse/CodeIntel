# Change: Refactor asset catalog and retire compatibility shims

## Why
The codebase still exposes legacy and compatibility surfaces (build.assets, no-op CLI commands,
compatibility shims, and unused parameters) that conflict with the native-only design and
complicate maintenance. Consolidating on a single versioned asset catalog and removing
compatibility surfaces improves clarity, correctness, and extensibility.

## What Changes
- **BREAKING**: Make the versioned asset catalog canonical
  (build.asset_versions + build.run_asset_versions + build.asset_version_events +
  build.asset_lineage), remove build.assets, and drop AssetRecord/AssetTracking legacy CRUD and
  build.assets CLI output (no transitional view).
- **BREAKING**: Separate immutable asset version metadata from run-scoped events and mappings;
  update catalog queries and CLI output to derive version history via event records.
- **BREAKING**: Remove deprecated schema flags and outputs (`--only-native`, legacy diff output);
  structured diff is the only schema diff format.
- **BREAKING**: Remove compatibility shims/re-exports (parsing validation reporters, CLI taxonomy,
  BuildResult adapter) and update callers to use canonical core modules.
- **BREAKING**: Remove compatibility-only parameters/methods across analytics, graphs, ingestion,
  CLI completion, and observability surfaces.
- Remove external-library compatibility normalization (numpy scalar conversions) and rely on
  DuckDB/Ibis native scalar handling.

## Impact
- Affected specs: build-execution, storage-boundaries, schema-contracts, parsing-models,
  error-reporting, interface-hygiene (new).
- Affected code: src/codeintel/build/assets/emitter.py,
  src/codeintel/storage/tracking/asset_tracking.py,
  src/codeintel/config/datasets/declared_schemas.py,
  src/codeintel/cli/handlers/build.py,
  src/codeintel/cli/commands/build.py,
  src/codeintel/cli/handlers/build_schema.py,
  src/codeintel/cli/commands/build_schema.py,
  src/codeintel/storage/ibis_adapter.py,
  src/codeintel/build/analytics/utilities/datasets.py,
  src/codeintel/build/analytics/parsing/validation.py,
  src/codeintel/cli/errors/taxonomy.py,
  src/codeintel/cli/handlers/storage.py,
  src/codeintel/cli/handlers/history.py,
  src/codeintel/build/graphs/engine/views.py,
  src/codeintel/build/graphs/compute/metrics/coupling.py,
  src/codeintel/ingestion/adapters/build_tool_adapter.py,
  src/codeintel/cli/completions/fish_generator.py,
  src/codeintel/cli/completions/zsh_generator.py,
  src/codeintel/cli/observability/_telemetry.py.

## Validation
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- `uv run pytest -q` (pending user run)
