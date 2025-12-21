## Context
The schema system currently has parallel sources of truth (declared schemas + contract/Pandera
registry vs DAG-derived schema index), and tool execution is split between build providers and
ingestion ToolService. This creates drift risk, duplicated configuration, and uneven provenance.
Incremental change detection and build hashing are also separated, which can lead to inconsistent
skip/rebuild decisions across ingestion and build targets.

## Goals / Non-Goals
- Goals:
  - Make Hamilton DAG derivation the canonical schema source for DAG outputs.
  - Allow schema inference in production with explicit provenance tracking.
  - Restrict declared schemas to source-only inputs and explicit overrides.
  - Provide a single tool execution surface via ToolService/ToolRunner in BuildEnv.
  - Route all ingestion through Hamilton native targets, including SCIP table ingestion.
  - Align incremental change detection with build input hashes and fingerprint policy.
  - Align analytics resource loading with BuildEnv providers for consistent DI.
  - Unify execution result types across ingestion and build targets.
  - Converge registries on Hamilton tag/TargetSpec metadata.
  - Centralize row serialization on schema registry row models.
- Non-Goals:
  - Implement the changes in this proposal (design only).
  - Change the storage engine or replace Pandera as a validation layer.

## Decisions
- Decision: SchemaIndex + UnifiedSchemaProvider are canonical for DAG outputs.
  - SCHEMA_REGISTRY becomes a projection built from the unified provider plus constraint
    overlays, not a parallel contract/Pandera registry.
  - Production inference is enabled by default for inferable outputs and is recorded with
    derivation provenance (explicit override vs inferred).
  - Inference failures are hard errors when no non-DAG alternative exists.
- Decision: Declared schemas are source-only.
  - Declared schema providers are reserved for non-DAG inputs and explicit overrides.
  - Schema-only enumeration remains DAG-free via a declared-only provider.
  - Backups or seeds MUST have DAG-derived update tooling to maintain DAG coupling.
- Decision: Pandera schemas and row bindings are derived from TableSchema.
  - Constraint overlays (Pandera checks) are layered on top of the DAG-first TableSchema.
  - Row bindings include schema_hash and derivation metadata.
  - Constraint enforcement order is Hamilton checks first, then Pandera, then Pydantic only
    when required.
- Decision: ToolService/ToolRunner is the canonical tool execution subsystem.
  - BuildEnv Providers expose ToolService and ToolRunner.
  - SubprocessToolRunner and Real* tool helpers are removed or replaced with ToolService-backed
    adapters.
- Decision: Ingestion runs only via Hamilton native targets.
  - SCIP ingestion becomes a Hamilton target that writes core.scip_* tables and artifacts.
  - Non-DAG ingestion entrypoints are removed from public surfaces.
- Decision: Schema manifest compilation records derivation provenance and inference status.
- Decision: Incremental change detection is derived from build input hash/fingerprint data.
  - File-state hashes are surfaced as target options or inputs to the hashing pipeline.
  - Skip/rebuild decisions use a single hash/fingerprint authority across ingestion and build.
  - Change-detection deltas are persisted alongside build manifests for auditability.
- Decision: Analytics resources align with BuildEnv providers.
  - Providers expose ResourceRegistry for analytics use without long-term compatibility shims.
- Decision: Execution results are unified under a shared result protocol.
  - Ingestion steps and executor targets share a single result model with skip semantics.
- Decision: Registry surfaces converge on Hamilton metadata.
  - TargetSpec + Hamilton tags become the single registry surface for discovery.
- Decision: Row serialization uses schema registry row bindings.
  - Row ordering and serialization are derived from schema registry row models.

## Alternatives Considered
- Keep dual schema registries with reconciliation checks. Rejected due to ongoing drift risk.
- Make SubprocessToolRunner the canonical executor. Rejected because ToolService already
  provides parsing, plugin metadata, and consistent tool semantics.
- Keep non-DAG ingestion steps for “quick runs.” Rejected to avoid duplicate execution paths.

## Risks / Trade-offs
- Inference in production can mask schema regressions.
  - Mitigation: record provenance, enforce schema diff gates, and require explicit overrides
    for non-inferable outputs.
- Migration touches many call sites and may be disruptive.
  - Mitigation: staged rollout, compatibility shims behind internal modules, and targeted tests.
- DAG inference may add latency at schema resolution time.
  - Mitigation: cached SchemaIndex results, batch inference for manifests, and lazy evaluation.
- Unified change detection may alter rebuild frequency.
  - Mitigation: expose diagnostics and diff reports for before/after behavior.
- Registry consolidation may reduce plugin flexibility.
  - Mitigation: retain extension points via TargetSpec and tag overlays.

## Migration Plan
1. Introduce DAG-first schema provider and provenance metadata without removing existing APIs.
2. Update SCHEMA_REGISTRY and row bindings to project from the unified provider.
3. Restrict declared schemas to source-only and add validations for DAG outputs.
4. Consolidate tool execution on ToolService/ToolRunner via BuildEnv providers.
5. Unify change detection with build hashing and align skip decisions.
6. Align analytics resource registry with BuildEnv providers.
7. Unify execution result models and update templates/targets.
8. Implement Hamilton-native SCIP ingestion and remove non-DAG ingestion entrypoints.
9. Remove legacy tool helpers and update documentation/tests.
10. Converge registry discovery on Hamilton tags/TargetSpec metadata.
11. Centralize row serialization on schema registry row models.

## Open Questions
## Open Questions (Resolved)
- Inference failures are hard errors when no viable non-DAG alternative exists.
- Default posture is DAG-produced datasets; backups/seeds require DAG-derived refresh tooling.
- Constraints use Hamilton checks first, then Pandera, then Pydantic only when required.
- Change-detection deltas are stored alongside build manifests for auditability.
- No long-term compatibility facade for analytics ResourceRegistry; short-lived shims only.
