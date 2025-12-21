## 1. Asset catalog refactor
- [x] 1.1 Update schema definitions for build.asset_versions/build.run_asset_versions/build.asset_version_events and remove build.assets.
- [x] 1.2 Remove AssetRecord/AssetTracking legacy CRUD and add versioned catalog queries.
- [x] 1.3 Update asset catalog emitters to persist versioned records only.
- [x] 1.4 Update build.assets CLI output to use versioned catalog only.

## 2. Schema diff and native-only flags
- [x] 2.1 Remove --only-native from schema compile/diff commands and request structs.
- [x] 2.2 Remove legacy schema diff output; structured diff only.

## 3. Compatibility shim removals
- [x] 3.1 Remove storage.generate_macros command/handler/result types and update docs/tests.
- [x] 3.2 Remove parsing validation re-export shim; update imports.
- [x] 3.3 Remove CLI taxonomy re-export; update callers to core taxonomy.
- [x] 3.4 Remove BuildResult adapter and legacy result interface; update build handlers/tests.

## 4. Interface hygiene cleanup
- [x] 4.1 Remove history_timeseries compatibility parameters and update call sites.
- [x] 4.2 Remove graph view compatibility fields/args and adjust consumers.
- [x] 4.3 Remove coupling metric compatibility parameter and update call sites.
- [x] 4.4 Remove ingestion build tool adapter unused args and update interface.
- [x] 4.5 Remove CLI completion depth params; update generators and docs.
- [x] 4.6 Remove telemetry compatibility methods; update observability interfaces.

## 5. External compatibility normalization
- [x] 5.1 Confirm DuckDB/Ibis handling of numpy scalars with current library behavior.
- [x] 5.2 Remove numpy scalar normalization helpers and update callers.

## 6. Tests and docs
- [x] 6.1 Update tests for asset catalog, schema diff, and interface changes.
- [x] 6.2 Update docs/examples referencing removed commands or flags.
