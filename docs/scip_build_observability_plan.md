SCIP Build Observability + Performance Plan

Overview
This plan implements best-in-class observability and performance improvements for the SCIP build
pipeline. It addresses long or opaque builds by adding structured telemetry, progress logging, and
performance-oriented changes such as batching and full-rebuild heuristics. The goal is to make the
build fast, transparent, and diagnosable, while preserving correctness and incremental behavior.

Goals
- Make SCIP build progress observable in real time (logs + structured telemetry).
- Identify why a build is slow via per-phase timing and tool output summaries.
- Prevent pathological slow paths (N-per-module tool invocations) through batching or full rebuild.
- Reduce unnecessary file hashing and repeated parsing where possible.
- Maintain deterministic, correct output with clear skip/rebuild decisions.

Non-Goals
- Change SCIP semantics, symbol identity, or data model beyond performance/telemetry needs.
- Introduce new external dependencies beyond the existing toolchain.
- Redesign the entire ingestion architecture outside the SCIP pathway.

Success Criteria
- Baseline SCIP full build completes within expected bounds (e.g., <= 30s on the current repo).
- Incremental runs avoid per-module scip-python invocations for large deltas.
- Telemetry captures phase timings and tool run summaries for every SCIP build.
- When a build is slow, logs and telemetry indicate which phase dominates.

Scope Summary
A. Observability foundation
B. Incremental plan summary + decision transparency
C. Performance upgrades (batching, thresholds, hashing reuse)
D. Manifest/module-state alignment and drift handling
E. Focused tests for correctness and telemetry

Design Principles
- Measure before changing behavior; emit telemetry for every phase.
- Prefer a single scip-python invocation over many (batching and full rebuild heuristics).
- Avoid re-reading files when state is already in the database.
- Keep configurations explicit and discoverable in logs.

Phase 0 - Baseline Capture and Safety Rails
1) Baseline metrics collection
   - Record current full-build time and shard counts on the repo.
   - Capture scip-python version and ToolsConfig parameters used.
   - Output baseline record to a JSON file in build/scip/ for comparison.

2) Define telemetry schemas and storage targets
   - Confirm where run telemetry is stored (DuckDB table + JSON artifacts).
   - Define a schema for SCIP run telemetry (see Phase 1).

Phase 1 - Telemetry Foundation and Progress Reporting
1) Add SCIP run telemetry dataclass and storage
   - New dataclass: ScipRunTelemetry
     - repo, commit, run_id, mode (full/incremental), options_hash, tool_version
     - counts: total_modules, changed_modules, deleted_modules, batch_count
     - durations: plan_ms, hash_ms, tool_ms, parse_ms, merge_ms, write_ms, total_ms
     - status: success/failed/skipped, error_summary
   - Persist to DuckDB table build.scip_runs (new schema) and to a JSON artifact
     in build/scip/runs/ for human inspection.
   - Hook into TargetRunRecord materialization or build tracking to persist the record.

2) ToolRunner progress and output summaries
   - Extend ToolRunOptions with optional telemetry hooks:
     - progress_interval_s (heartbeat)
     - log_prefix (identify long-running tool phases)
   - In ToolRunner.run_async:
     - Emit a heartbeat log entry every N seconds with elapsed time.
     - Record stdout/stderr tail (e.g., last 2 KB) for debugging.
   - Extend tool call logs with:
     - output_path size at end of run
     - elapsed time
     - version if available

3) SCIP run plan summary logging
   - Add a ScipIncrementalPlan log entry containing:
     - options mismatch reason (if any)
     - counts: changed, deleted, total
     - batching decision and target batch size
     - full rebuild decision and threshold used
   - Log once per run and persist in ScipRunTelemetry.

Phase 2 - Performance Improvements (Batching + Heuristics)
1) Introduce batching for incremental shard indexing
   - New helper: partition_plans(plans, batch_size, max_batch_bytes)
   - Run scip-python once per batch with multiple --target-only args.
   - Parse the batch index once and map documents to module records.
   - Write one shard file per batch; update module manifest records to point to
     the batch shard. Ensure downstream merge dedupes unique shard paths.

2) Full rebuild heuristics
   - Add thresholds (ToolsConfig or BuildSettings):
     - scip_full_rebuild_threshold_count
     - scip_full_rebuild_threshold_ratio
   - If changed_modules exceed thresholds, run a single full rebuild.
   - Log the decision and store in telemetry.

3) Hash reuse and plan acceleration
   - Prefer content hashes from core.file_state or core.scip_module_state
     instead of reading files in _build_shard_plans.
   - Use file_state table when available; fall back to digest only for missing rows.
   - Record a hash_source field ("file_state" | "computed") in telemetry.

4) Avoid unnecessary parsing
   - Skip base index parse when full rebuild is forced or output_scip missing.
   - Deduplicate shard path parsing during merge (load each shard once).

Phase 3 - Manifest and Module-State Alignment
1) Treat scip_module_state as canonical
   - After indexing, always upsert per-module state with options_hash, tool_version,
     and shard_path.
   - Regenerate manifest from module_state when manifest is missing or inconsistent.

2) Harden manifest validity checks
   - Validate shard_path existence and checksum if available.
   - Log and repair when a shard is missing or inconsistent.

Phase 4 - Developer Experience and CLI Surfacing
1) Enhanced logs and artifacts
   - Emit a concise progress line per phase (plan/hash/tool/parse/merge/write).
   - Write a run report to build/scip/runs/<timestamp>.json.

2) Optional verbosity flags
   - Support a CODEINTEL_SCIP_TRACE or build flag to toggle verbose progress logs.

Phase 5 - Testing Plan (Pytest)
Unit tests
- test_scip_incremental_plan_summary
  - Ensures plan summary logs correct counts and decisions.
- test_scip_batch_partitioning
  - Validates batch sizing and stable ordering of shards.
- test_scip_full_rebuild_thresholds
  - Confirms threshold logic selects full rebuild when expected.
- test_scip_hash_reuse_file_state
  - Verifies hashes are read from core.file_state when available.
- test_scip_manifest_regeneration
  - Ensures manifest rebuilds from module_state correctly.

Integration tests
- test_scip_tool_runner_heartbeat
  - Uses a fake tool runner that sleeps and asserts heartbeat logs.
- test_scip_batch_shard_mapping
  - Indexes a small set of modules and validates per-module shard mapping.
- test_scip_run_telemetry_persisted
  - Asserts build.scip_runs row is written with correct phase durations.

Test tooling notes
- Use a fake/scaffold ToolRunner to avoid invoking scip-python in unit tests.
- Keep a small test repo fixture for integration-level scip indexing
  (or reuse existing ingestion test fixtures).

Rollout and Validation
- Land telemetry first to capture baseline measurements.
- Enable batching and thresholds behind config defaults, then tighten thresholds.
- Compare run reports before and after to validate performance improvements.

Risks and Mitigations
- Risk: Batch shards complicate merge logic.
  Mitigation: Deduplicate shard paths and add tests for batch mapping.
- Risk: Full rebuild thresholds too aggressive.
  Mitigation: Start with conservative thresholds, track telemetry for tuning.

Open Questions
- Preferred location for build.scip_runs schema and whether it should be optional.
- Whether scip-python emits progress stdout that should be streamed by default.
