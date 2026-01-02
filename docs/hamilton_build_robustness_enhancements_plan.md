# Hamilton Build Robustness Enhancements Plan

## Goals

- Make the Hamilton build DAG safer and more deterministic at runtime.
- Detect and explain schema drift and misconfiguration early.
- Improve reproducibility for inference and build outputs.
- Emit a unified, build-scoped JSONL log artifact for diagnostics.

## Scope summary

This plan covers six improvements:

1. DAG preflight audit and layering enforcement
2. Deterministic inference plan manifest
3. Schema drift gates at materialization
4. Typed BuildConfig and execution settings validation
5. Runtime/settings fingerprint in dataset metadata
6. High-resolution diagnostics + unified build log artifact

Status overview (current):
- Workstream 1: implemented
- Workstream 2: implemented
- Workstream 3: not implemented
- Workstream 4: implemented
- Workstream 5: not implemented
- Workstream 6: implemented

## Success criteria

- Build fails fast with a clear error when tag contracts or layering are violated.
- Schema inference emits a reproducible plan artifact tied to a snapshot id.
- Schema drift is logged and optionally enforced at materialization time.
- Build configuration rejects unknown keys and invalid values with explicit errors.
- Dataset metadata includes a stable fingerprint of runtime and settings.
- A unified build log JSONL artifact is emitted alongside each dataset snapshot.

---

## Workstream 1: DAG preflight audit and layering enforcement

### Intent

Ensure every build run is validated for correct tags, saver wiring, and build-only
imports before execution.

### Design

- Add a preflight validator that runs after DAG construction and before execution.
- Validate:
  - Required tag presence on table nodes (table_key, target, data_node).
  - Saver and contract pairing (dataset nodes must have a saver and a contract).
  - No build modules import storage or serving modules.
  - No duplicate table_key across outputs.
- Provide a concise failure report that lists offending nodes and tags.

### Files to change

- src/codeintel/build/hamilton/dag_catalog.py
  - Add a preflight validation pass and report structure.
- src/codeintel/build/hamilton/tagging.py
  - Centralize required tag sets and the allowlist rules.
- src/codeintel/build/hamilton/executor.py
  - Invoke preflight before execution and surface errors.

### Checklist

- [x] Preflight validation runs before any target execution.
- [x] Tag requirements enforced for all table outputs.
- [x] Layering violations (build importing storage/serving) are detected.
- [x] Failures include node name, table_key, and missing tags.

---

## Workstream 2: Deterministic inference plan manifest

### Intent

Record a reproducible schema inference plan that can be replayed or audited.

### Design

- Emit a manifest for inference runs containing:
  - snapshot (repo, commit, repo_root)
  - target list and table_keys
  - qparams and loader overrides
  - seed dataset metadata
  - inference settings (polars flags, streaming, etc.)
- Persist as a JSON artifact and optionally as a small dataset table.

### Files to change

- src/codeintel/build/schemas/inference_service.py
  - Build and emit the inference plan manifest. (done)
- src/codeintel/build/assets/emitter.py
  - Register run-scoped inference plan artifact in asset tracking. (done)
- src/codeintel/core/manifests.py
  - Add msgspec.Struct for inference manifest (if needed). (done)

### Checklist

- [x] Inference plan manifest contains snapshot + target list + settings.
- [x] Manifest stored under `build/schema/inference_plan_<run_id>.json`.
- [x] Run-scoped artifact registration recorded in asset tracking.

---

## Workstream 3: Schema drift gates at materialization

### Intent

Detect and optionally enforce schema drift at the dataset boundary.

### Design

- Compare observed schema vs contract schema during dataset materialization.
- Support validation modes:
  - off: log only
  - warn: emit drift events but continue
  - strict: raise and fail the target
- Produce a structured drift record with column diffs.

### Files to change

- src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
  - Add drift comparison and optional enforcement.
- src/codeintel/build/hamilton/native/patterns/savers.py
  - Allow configuration of drift enforcement mode.
- src/codeintel/core/manifests.py
  - Add a drift event manifest type (optional).

### Checklist

- [ ] Drift detection is computed at materialization.
- [ ] Enforcement respects configured mode.
- [ ] Drift events include detailed column diffs.

---

## Workstream 4: Typed BuildConfig and execution settings validation

### Intent

Fail fast on mis-typed or unknown config keys and make configuration fully explicit.

### Design

- Introduce msgspec.Struct-based config models with strict decoding.
- Reject unknown keys and invalid types at load time.
- Provide an error report that points to the config path and offending keys.

### Files to change

- src/codeintel/build/config.py
  - Add typed config parsing and strict validation.
- src/codeintel/build/hamilton/execution_options.py
  - Validate execution settings, include in error report.
- src/codeintel/core/config/settings.py
  - Ensure runtime settings are aligned with config validation.

### Checklist

- [x] Config decoding is strict and rejects unknown keys.
- [x] Validation errors are logged and surfaced to CLI.
- [x] Execution settings are validated consistently.

---

## Workstream 5: Runtime/settings fingerprint in dataset metadata

### Intent

Provide full lineage for dataset outputs so scans can detect changes in runtime settings.

### Design

- Compute a stable fingerprint of build settings and runtime options.
- Persist in dataset metadata (parquet metadata + manifest extras).
- Include snapshot and build id components for traceability.

### Files to change

- src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
  - Compute and store fingerprint in metadata.
- src/codeintel/core/manifests.py
  - Include fingerprint fields in dataset manifest (if needed).
- src/codeintel/build/hamilton/run_records.py
  - Capture fingerprint in run metadata.

### Checklist

- [ ] Fingerprint is stable and deterministic.
- [ ] Fingerprint is stored in parquet metadata and manifest.
- [ ] Fingerprint is logged at write time.

---

## Workstream 6: High-resolution diagnostics + unified build log artifact

### Intent

Make inference and runtime errors fully diagnosable with node-level context, and
emit a single consolidated log artifact per build run stored alongside the
Parquet dataset snapshot.

### Design

- Diagnostics logging
  - Emit explicit node error events with node_name, target, table_key, exception type, and message.
  - Emit inference job lifecycle events (start/ok/fail) with qparam and loader override counts.
  - Emit a top-level runtime failure event for uncaught execution errors.
  - Wire HamiltonTracker tags to include repo/commit/run_id for UI traceability.
- Unified build log artifact
  - Collect structured events emitted during a run into a single JSONL file.
  - Write the log file into the dataset root for the snapshot (same folder as the Parquet
    dataset snapshot, e.g. `<dataset_root>/<snapshot_id>/build_logs/build_<run_id>.jsonl`).
  - Ensure the log artifact is written at end-of-run (success or failure).

### Files to change

- src/codeintel/build/hamilton/build_log.py
  - Buffer build-scoped events and compute JSONL path. (done)
- src/codeintel/build/hamilton/hooks/telemetry_hook.py
  - Emit node-level error events with consistent fields. (done)
- src/codeintel/build/schemas/inference_service.py
  - Emit inference job start/ok/fail events with table_key + qparam metadata. (done)
- src/codeintel/build/hamilton/executor.py
  - Emit runtime failure events and finalize the unified log. (done)
- src/codeintel/build/hamilton/run_writer.py
  - Write consolidated JSONL log artifact under dataset_root snapshot folder. (done)

### Build log events (implemented)

- Event: build.node.error
  - fields: run_id, repo, commit, node_name, target, table_key, exception_type, error
- Event: build.inference.job.start
  - fields: run_id, repo, commit, table_key, target, qparams_count, loader_overrides_count
- Event: build.inference.job.ok
  - fields: run_id, repo, commit, table_key, target, duration_ms
- Event: build.inference.job.fail
  - fields: run_id, repo, commit, table_key, target, exception_type, error
- Event: build.runtime.error
  - fields: run_id, repo, commit, exception_type, error

### Checklist

- [x] Node error events include node_name + target + table_key.
- [x] Inference job lifecycle events are emitted.
- [x] A single JSONL log artifact is written per build run under the dataset root snapshot.

---

## Rollout plan

### Phase 0: Audit and scaffolding

- Add logging events and no-op validators.
- Ensure build runs are unaffected by default settings.

### Phase 1: Preflight and config validation

- Turn on DAG preflight checks.
- Enforce strict BuildConfig decoding.

### Phase 2: Schema drift gating

- Enable drift detection in warn mode by default.
- Add strict mode for CI or targeted runs.

### Phase 3: Inference plan manifest + metadata fingerprint

- Emit inference plan manifest for all inference sessions.
- Write settings fingerprint into dataset metadata.

---

## Validation checklist

- [ ] `uv run codeintel build run --all` succeeds with preflight enabled.
- [ ] Invalid config key produces a structured validation error.
- [ ] Schema drift emits a structured event and is visible in logs.
- [ ] Inference plan manifest is present and contains settings, targets, and qparams.
- [ ] Dataset metadata includes a stable settings fingerprint.
- [ ] Consolidated build log JSONL artifact exists alongside the snapshot Parquet datasets.
