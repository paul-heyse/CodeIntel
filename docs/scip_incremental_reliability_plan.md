SCIP Incremental Reliability + Tool Capability Plan

Overview
This plan implements a best-in-class reliability and extensibility upgrade for SCIP incremental
indexing and tool execution. It addresses recent test failures by making rebuild decisions
explicit and policy-driven, hardening hash reuse and batching determinism, and adding capability
probes to avoid brittle tool failures (e.g., pytest JSON report support).

Goals
- Make incremental rebuild decisions correct, deterministic, and auditable across repo sizes.
- Centralize hash reuse policy and telemetry so file-state reuse is reliable and explainable.
- Make batching deterministic and observable with stable shard mapping.
- Harden tool execution paths with capability detection and failure classification.
- Improve maintainability via cohesive policy/resolver/service abstractions with tests.

Non-Goals
- Change SCIP semantics, symbol identity, or ingestion table schemas beyond telemetry fields.
- Replace the ToolRunner infrastructure or introduce new external dependencies.

Design Principles
- Explicit policy objects over ad-hoc conditionals.
- Determinism in decisions and outputs (sorted plans, stable batch IDs).
- Telemetry mirrors the decision logic and captures the full rationale.
- Capability detection before execution for optional tool features.
- Reusable services for hashing and policy to reduce duplication.

Scope Summary
A. Incremental rebuild policy object and decision record
B. Hash source resolver service (file_state/module_state/disk)
C. Deterministic batching and batch metadata
D. Telemetry schema upgrade for decisions and hash provenance
E. Tool capability probe + failure classification
F. Focused test coverage for new behaviors

Architecture Changes

A) Incremental rebuild policy (ScipIncrementalPolicy)
- Introduce a policy object that encapsulates all rebuild decision rules.
- Inputs: total_modules, changed_count, changed_ratio, options_hash, manifest presence,
  force_full_rebuild, output_scip existence.
- Default policy fields (proposed):
  - full_rebuild_threshold_count: int (existing)
  - full_rebuild_threshold_ratio: float (existing)
  - full_rebuild_ratio_min_modules: int (new, e.g., 200)
  - full_rebuild_ratio_min_changed: int (new, e.g., 25)
- Decision output: ScipIncrementalDecision (see below).
- This policy becomes the only place that determines full rebuild vs incremental.

B) Decision object (ScipIncrementalDecision)
- Dataclass fields:
  - mode: "full" | "incremental"
  - reason: "force_full_rebuild" | "options_mismatch" | "threshold_count" |
    "threshold_ratio" | "parse_failed_full_rebuild" | "incremental_failed_full_rebuild" |
    "incremental"
  - total_modules, changed_count, changed_ratio
  - thresholds (count, ratio)
  - ratio_gate_min_modules, ratio_gate_min_changed
  - ratio_gate_applied: bool
- Ensure telemetry and log summary use this object directly to prevent drift.

C) Hash source resolver (FileDigestResolver)
- New service that resolves FileDigest values using:
  1) core.file_state (or precomputed file_state_by_path)
  2) core.scip_module_state (optional extension)
  3) On-disk hashing fallback
- Expose hash provenance summary:
  - hash_source: "file_state" | "module_state" | "computed" | "mixed"
  - hash_reused: int, hash_computed: int
- Output used by _build_shard_plans and telemetry.

D) Deterministic batching and metadata
- Sort plans by scip_rel_path before partitioning.
- Assign batch IDs deterministically based on (scip_rel_path, content_hash) tuples.
- Add BatchPlan struct (batch_id, size_bytes, rel_paths).
- Ensure manifest references stable batch shard paths.

E) Telemetry schema upgrades
- Extend ScipRunTelemetry fields to include:
  - decision_reason
  - ratio_gate_applied
  - ratio_gate_min_modules
  - ratio_gate_min_changed
  - hash_source_breakdown (optional, e.g., "file_state:10,computed:2")
- Update build.scip_runs schema and JSON report payload accordingly.

F) Tool capability probe + failure classification
- Add ToolCapabilityProbe (shared utility) that checks:
  - pytest JSON report capability (via `pytest --help` output).
  - Optional: coverage JSON availability.
- PytestPlugin behavior:
  - If json-report flags unsupported, return ToolStatus.NOT_FOUND or a new
    ToolStatus.MISSING_DEPENDENCY and parsed TestReport.empty().
  - Do not raise ToolExecutionError for capability failures.
- ToolService.run_pytest_report:
  - Treat capability failures as non-fatal and return False (skipped).
- Optional: Add stderr parsing to classify "unrecognized arguments" as capability failures.

Configuration Changes
- Update ScipIngestOptions:
  - full_rebuild_ratio_min_modules: int
  - full_rebuild_ratio_min_changed: int
- Default values:
  - full_rebuild_ratio_min_modules = 200
  - full_rebuild_ratio_min_changed = 25
  - ratio thresholds apply only if both gates are satisfied.
- Document in options schema and target plan docs.

API Changes
- New dataclasses in `codeintel.ingestion.scip.incremental`:
  - ScipIncrementalPolicy
  - ScipIncrementalDecision
  - FileDigestResolver (or in a new module under ingestion/scip)
- Update ScipIncrementalConfig to accept policy or policy inputs.
- Update telemetry serialization to include new fields.

Implementation Steps

Phase 1: Policy + Decision Object
1) Add ScipIncrementalDecision and ScipIncrementalPolicy.
2) Refactor decision logic to return ScipIncrementalDecision.
3) Update plan log and telemetry to consume the decision object.
4) Add ratio gate checks (min modules/changed).

Phase 2: Hash Resolver Service
1) Implement FileDigestResolver with source ordering and provenance tracking.
2) Wire into _build_shard_plans to reuse hashes consistently.
3) Update telemetry hash_source/hash_reused/hash_computed via resolver summary.

Phase 3: Deterministic batching
1) Sort plans by scip_rel_path before partitioning.
2) Create BatchPlan abstraction for logging + telemetry.
3) Update batch shard naming to use deterministic batch ID.
4) Add log line summarizing batch count, bytes, and batch IDs.

Phase 4: Telemetry schema update
1) Extend ScipRunTelemetry fields and JSON payload.
2) Update build.scip_runs schema and BuildTracking persistence.
3) Update any report readers or telemetry consumers if applicable.

Phase 5: Tool capability probe
1) Add ToolCapabilityProbe utility (e.g., under ingestion/engine/tools).
2) Implement pytest json-report capability check.
3) Update PytestPlugin to skip or degrade gracefully on missing capability.
4) Update ToolService.run_pytest_report to treat capability missing as non-fatal.

Testing Plan

Unit Tests
- test_scip_incremental_policy_ratio_gate:
  - Verifies ratio thresholds are gated by min modules + min changed.
- test_scip_decision_object_consistency:
  - Ensures telemetry/logs mirror decision object.
- test_hash_resolver_file_state_preferred:
  - Verifies file_state reuse is detected and provenance tracked.
- test_hash_resolver_fallback_to_disk:
  - Ensures missing file_state triggers hashing.
- test_batching_is_deterministic:
  - Same inputs produce same batch IDs and shard paths.
- test_pytest_capability_probe_missing:
  - Simulate missing json-report; expect ToolStatus.NOT_FOUND or MISSING_DEPENDENCY.

Integration Tests
- test_incremental_small_repo_no_full_rebuild:
  - A small repo with 100% changes should still be incremental when gates apply.
- test_tool_service_run_coverage_report_with_data:
  - Should skip pytest JSON if capability is missing, not error.
- test_scip_run_telemetry_includes_decision_fields:
  - Validates new telemetry fields are persisted.

Rollout Plan
1) Land policy + decision object with default gating values.
2) Land hash resolver and batching determinism changes.
3) Update telemetry schema and backfill if needed.
4) Add capability probe and adjust tool behavior.
5) Run full quality report and ingestion tests.

Risks and Mitigations
- Risk: Default gates hide legitimate full rebuild opportunities.
  - Mitigation: Log gate application in telemetry and expose config overrides.
- Risk: Telemetry schema changes require migrations.
  - Mitigation: Use nullable columns and versioned JSON fields.
- Risk: Capability probes add latency.
  - Mitigation: Cache probe results per run or per tool invocation.

Open Questions
- Should ScipIncrementalPolicy live with ScipIngestOptions or as a separate config block?
- Should tool capability probe be global (shared) or per plugin?
- Do we want to persist capability probe results in build metadata for auditing?
