# Hamilton Build Robustness Enhancements Plan

## Goals

- Make the Hamilton build DAG safer and more deterministic at runtime.
- Detect and explain schema drift and misconfiguration early.
- Improve reproducibility for inference and build outputs.
- Expand structured logging so audits and failures are diagnosable from logs alone.

## Scope summary

This plan covers five improvements:

1. DAG preflight audit and layering enforcement
2. Deterministic inference plan manifest
3. Schema drift gates at materialization
4. Typed BuildConfig and execution settings validation
5. Runtime/settings fingerprint in dataset metadata

It also adds a structured logging expansion that cuts across each area.

## Success criteria

- Build fails fast with a clear error when tag contracts or layering are violated.
- Schema inference emits a reproducible plan artifact tied to a snapshot id.
- Schema drift is logged and optionally enforced at materialization time.
- Build configuration rejects unknown keys and invalid values with explicit errors.
- Dataset metadata includes a stable fingerprint of runtime and settings.
- Logs provide enough data to diagnose failures without re-running builds.

---

## Workstream 1: DAG preflight audit and layering enforcement

### Intent

Ensure every build run is validated for correct tags, saver wiring, and build-only imports before execution.

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

### Structured logging

- Event: build.dag.preflight.start
  - fields: run_id, repo, commit, target_count, table_count
- Event: build.dag.preflight.fail
  - fields: run_id, repo, commit, failures (list of structured entries)
- Event: build.dag.preflight.ok
  - fields: run_id, repo, commit, duration_ms

### Checklist

- [ ] Preflight validation runs before any target execution.
- [ ] Tag requirements enforced for all table outputs.
- [ ] Layering violations (build importing storage/serving) are detected.
- [ ] Failures include node name, table_key, and missing tags.

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
  - Build and emit the inference plan manifest.
- src/codeintel/build/hamilton/run_records.py
  - Register artifact output and include in run record metadata.
- src/codeintel/core/manifests.py
  - Add msgspec.Struct for inference manifest (if needed).

### Structured logging

- Event: build.inference.plan.emit
  - fields: run_id, repo, commit, table_keys_count, qparams_count
- Event: build.inference.plan.fail
  - fields: run_id, repo, commit, error

### Checklist

- [ ] Inference plan manifest contains snapshot + target list + settings.
- [ ] Manifest stored alongside other build artifacts.
- [ ] Logging emitted for success and failure.

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

### Structured logging

- Event: build.schema.drift.detected
  - fields: run_id, repo, commit, table_key, added_cols, removed_cols, type_changes
- Event: build.schema.drift.blocked
  - fields: run_id, repo, commit, table_key, reason

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

### Structured logging

- Event: build.config.validation.fail
  - fields: run_id, config_path, invalid_keys, error
- Event: build.config.validation.ok
  - fields: run_id, config_path

### Checklist

- [ ] Config decoding is strict and rejects unknown keys.
- [ ] Validation errors are logged and surfaced to CLI.
- [ ] Execution settings are validated consistently.

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

### Structured logging

- Event: build.dataset.metadata.write
  - fields: run_id, table_key, settings_fingerprint, schema_hash

### Checklist

- [ ] Fingerprint is stable and deterministic.
- [ ] Fingerprint is stored in parquet metadata and manifest.
- [ ] Fingerprint is logged at write time.

---

## Structured logging expansion (shared schema)

### Common fields

Use these fields consistently across all build validation and drift events:

- event
- run_id
- repo
- commit
- target (when relevant)
- table_key (when relevant)
- data_node (when relevant)
- mode (validation mode or inference mode)
- duration_ms
- error (string)
- details (structured payload)

### Logging principles

- Use structlog with JSONRenderer and orjson for stable payloads.
- Avoid logging large arrays directly; summarize and provide counts.
- Include a stable error code when possible.

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

