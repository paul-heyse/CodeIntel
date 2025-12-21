## Context
The codebase has accumulated multiple legacy and compatibility paths: a legacy asset catalog
(build.assets) alongside versioned asset records, deprecated CLI commands, compatibility
re-exports, and unused parameters kept solely for legacy callers. These paths increase
maintenance cost and blur the native-only architecture.

## Goals / Non-Goals
- Goals:
  - Make versioned asset catalog data canonical and immutable.
  - Remove legacy catalog tables, compatibility shims, and no-op CLI commands.
  - Ensure schema diff and schema compilation are single-path, native-only behaviors.
  - Remove external-library compatibility normalization.
  - Tighten public interfaces by removing compatibility-only parameters.
- Non-Goals:
  - Change asset fingerprinting algorithms or version hash formats.
  - Introduce new export formats or new serving endpoints.
  - Alter the underlying DuckDB/Ibis storage engine choice.

## Decisions
- Asset catalog model:
  - Treat build.asset_versions as the canonical, immutable store of asset version metadata.
  - Move run-scoped fields (repo, commit, run_id, target, status, impl_kind, location,
    input_hash, options_hash) into build.asset_version_events so asset_versions remains
    content-addressed and stable across runs and commits.
  - Preserve build.run_asset_versions as the run-level resolution map.
  - Remove build.assets and AssetRecord/AssetTracking legacy CRUD. Use derived queries/views
    for "latest asset" lookups if needed.
  - Do not provide transitional build.assets views; consumers must use versioned tables.
- CLI asset listing:
  - Build assets CLI returns versioned catalog output only; drop the legacy list path and
    the versions toggle.
- Schema diff:
  - Structured schema diff with breaking-change detection is the only output; remove the
    legacy string diff path and the --detailed toggle.
- Compatibility shims:
  - Remove compatibility re-exports and adapters; callers import canonical core modules
    directly (validation reporters, error taxonomy, build results).
- External-library compatibility:
  - Remove numpy scalar normalization and rely on DuckDB/Ibis native scalar handling.
- Interface hygiene:
  - Remove compatibility-only parameters and methods across analytics, graphs, ingestion,
    CLI completions, and observability.

## Alternatives Considered
- Keep build.assets and write-through to both catalogs.
  - Rejected: perpetuates dual sources of truth and ongoing maintenance.
- Preserve compatibility-only parameters with "_unused" prefixes.
  - Rejected: still leaks legacy surface area; prefer explicit removal and migration.
- Keep legacy schema diff output behind a flag.
  - Rejected: forces parallel maintenance for no current consumers.

## Risks / Trade-offs
- Breaking changes for any scripts or tooling that read build.assets or legacy CLI outputs.
  Mitigation: document migrations, provide clear error messages, and update docs/examples.
- Asset version history queries now require joins between version and run mapping tables.
  Mitigation: add indexes and helper queries in storage layer; add focused tests.
- DuckDB/Ibis scalar handling depends on upstream behavior; mitigation: rely on current
  library versions and existing integration tests.

## Migration Plan
1. Update schemas for build.asset_versions, build.run_asset_versions, and
   build.asset_version_events. Add indexes needed for repo/commit and asset lookups.
2. Update asset catalog writers and readers to use the versioned model only.
3. Update build.assets CLI output to use versioned catalog queries; remove legacy flags.
4. Remove build.assets schema, AssetRecord/AssetTracking legacy CRUD, and related tests.
5. Remove deprecated CLI commands and compatibility shims; update imports and docs.
6. Remove compatibility-only parameters/methods; update call sites and tests.
7. Remove numpy scalar normalization, backed by new tests.

## Open Questions
- None.
