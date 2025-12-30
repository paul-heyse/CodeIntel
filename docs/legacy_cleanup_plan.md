# Legacy Cleanup Plan

## Purpose

Remove dead, legacy, and compatibility-only code paths that no longer align with
current architecture, and update any references that still depend on them.

## Scope (Targets Identified)

Completed removals:
- `src/codeintel/storage/manifests/` and `src/codeintel/storage/io/` (empty packages removed).
- `src/codeintel/storage/materialization.py` (unused helper removed).
- `src/codeintel/build/serving/manifest.py` (compat shim removed).
- `src/codeintel/storage/serving/snapshot_service.py` (moved into build serving prep).
- Arrow dataset configuration stubs (already removed from settings/loader).

Remaining targets:
- Manifest-driven registry sync path:
  - `src/codeintel/storage/metadata/sync.py:sync_table_schema_registry_from_latest_manifest`
    is only re-exported, has no call sites.
- Legacy serving manifest schema:
  - `src/codeintel/config/schemas/serving/snapshot_manifest.json` should remain aligned with
    `codeintel.core.manifests.ServingSnapshotManifest` (verify fields).

## Cleanup Strategy

1) Remove dead package shells
   - Completed.

2) Remove unused view helper
   - Completed.

3) Align serving snapshot schema with actual manifest payload
   - Verify `src/codeintel/config/schemas/serving/snapshot_manifest.json` matches
     `codeintel.core.manifests.ServingSnapshotManifest` (no legacy fields).

4) Remove serving manifest compatibility shim
   - Completed.

5) Remove manifest-driven registry sync path
   - Remove `sync_table_schema_registry_from_latest_manifest` from
     `src/codeintel/storage/metadata/sync.py` and its re-exports in
     `src/codeintel/storage/metadata/__init__.py`.
   - Confirm no callers exist (currently none).

6) Retire Arrow dataset configuration stub
   - Completed.

## Execution Plan (Step-by-Step)

1. Confirm zero live references
   - `rg -n "storage.manifests|storage.io|create_view_from_relation|build.serving.manifest" src tests`
   - `rg -n "ArrowDatasetSettings" src`

2. Apply removals
   - Delete empty dirs: `src/codeintel/storage/manifests/`, `src/codeintel/storage/io/`.
   - Delete files: `src/codeintel/storage/materialization.py`,
     `src/codeintel/build/serving/manifest.py`.
   - Remove manifest-driven sync in `src/codeintel/storage/metadata/sync.py` and
     its exports in `src/codeintel/storage/metadata/__init__.py`.
   - Remove Arrow dataset settings and loader wiring.

3. Update schemas/imports
   - Update `src/codeintel/config/schemas/serving/snapshot_manifest.json`.
   - Update test imports (if any remain) to use `codeintel.core.manifests`.

4. Validate
   - `uv run ruff check`
   - `uv run pyright`
   - `uv run pyrefly check`
   - Run any targeted tests that previously referenced removed paths.

## Acceptance Criteria

- No references to removed modules remain.
- Serving snapshot schema matches the manifest dataclass.
- No unused package shells or helper files remain.
- Ruff, Pyright, and Pyrefly pass without new errors.
