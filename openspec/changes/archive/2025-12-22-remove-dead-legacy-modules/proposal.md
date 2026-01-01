# Change: Remove dead legacy modules and compatibility shims

## Status
Archived on 2025-12-22 after implementation. Validation for guardrails/pytest is deferred to
the parallel guardrails workstream (schema override checks).

## Why
Dead and compatibility modules remain in the public package surface, creating confusion,
unused maintenance burden, and drift from the canonical architecture. We need a single,
clear set of interfaces aligned with current specs.

## What Changes
- Remove unused CLI entrypoints and helper modules (completion installer, jobs runner,
  skip-arg helper, MCP module entrypoint).
- Remove deprecated manifest and row-serialization re-export shims.
- Remove unused ingestion helpers (BuildToolAdapter, ChangeTracker).
- Remove duplicate tool execution helpers and legacy result builder/alias helpers.
- Remove legacy schema export/lineage/migration utilities and static column lists.
- Remove unused storage helpers (ephemeral gateway, dataset catalog generator).
- Remove test-only architecture boundary helpers from the runtime package.
- **BREAKING** Remove public compatibility shims and legacy entrypoints that may be imported
  by external callers.

## Impact
- Affected specs: interface-hygiene, build-execution, schema-contracts, storage-boundaries.
- Affected code: `src/codeintel/cli`, `src/codeintel/build`, `src/codeintel/ingestion`,
  `src/codeintel/storage`, `src/codeintel/build/analytics`, `src/codeintel/_architecture`.
