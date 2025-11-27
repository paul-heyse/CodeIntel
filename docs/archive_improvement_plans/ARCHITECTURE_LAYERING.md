# Architecture Layering

This repo follows a three-layer structure to keep dependencies flowing in one direction.

## Layers

- **Core**: Configuration primitives and schemas, storage gateways, dataset/schema contracts.
- **Domain**: Ingestion, analytics, graphs, other read/write logic built on top of storage.
- **Application**: CLI, serving surfaces, pipeline orchestration, configuration builders.

## Import rules

- Core may import only other core modules.
- Domain may import core and storage helpers, but not CLI or serving packages.
- Application may import any layer.

## Examples

- ✅ `codeintel.graphs.engine_factory` imports `codeintel.graphs.nx_backend`.
- ✅ `codeintel.analytics.context` imports `codeintel.storage.module_index.load_module_map`.
- ✅ `codeintel.ingestion.common` imports `codeintel.storage.sql_helpers.prepared_statements_dynamic`.
- ❌ `codeintel.graphs.engine_factory` importing `codeintel.cli.nx_backend` (application leak).

## Enforcement

The optional `tools/check_layering.py` script reports violations by scanning imports. Run it from
the repo root to confirm module boundaries before landing changes.
