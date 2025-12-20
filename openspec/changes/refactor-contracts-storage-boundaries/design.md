## Context
The current architecture mixes storage-specific types (DuckDB) into build/export code,
creates divergent contract-to-schema mappings across layers, and eagerly initializes the
Hamilton DAG during schema enumeration. Tests now require monkeypatching and property-based
roundtrips exceed deadlines. This change proposes a structural realignment focused on
layering, deterministic contract policies, safe query behavior, and explicit dependency
injection.

## Goals / Non-Goals
- Goals:
  - Enforce storage boundary: DuckDB usage only within storage modules.
  - Establish a single, canonical contract policy for schema IDs and exportability.
  - Guarantee safe query helpers never raise for invalid input.
  - Avoid Hamilton DAG initialization for schema-only contract enumeration.
  - Remove monkeypatch reliance by providing injectable settings/providers.
- Non-Goals:
  - Redesign the Hamilton DAG itself or change analytic outputs.
  - Change external storage backends beyond DuckDB isolation.
  - Replace the existing schema registry system end-to-end.

## Decisions
- Decision: Define storage-owned protocols for relation/record batch handling and keep
  duckdb types confined to storage modules.
  - Alternatives considered: Move all export code into storage (rejected: over-couples
    build logic and storage concerns). The protocol approach preserves boundaries while
    allowing shared export behavior.
- Decision: Introduce a shared contract policy module for schema ID derivation and
  exportability rules, consumed by both build and storage providers.
  - Alternatives considered: Duplicate policy with lint checks (rejected: still allows
    drift and inconsistent behavior).
- Decision: Provide strict and safe table-key validation APIs; safe_* helpers return
  None/False on invalid input and never raise.
  - Alternatives considered: Allow ValueError for invalid keys (rejected: violates safe
    API contract and breaks SQL-injection safety expectations).
- Decision: Split contract enumeration into schema-only and metadata-enriched paths.
  Schema-only enumeration MUST NOT trigger DAG initialization; metadata enrichment is
  lazy and only when requested.
  - Alternatives considered: Cache full TargetMetadataService globally (rejected: still
    forces initial DAG load and complicates tests).
- Decision: Replace monkeypatch with explicit dependency injection via settings objects
  (e.g., BuildSettings) and injectable providers (e.g., ContractProvider).
  - Alternatives considered: Keep monkeypatch as test-only (rejected: hides production
    coupling and prevents deterministic behavior).

## Risks / Trade-offs
- Risk: Introducing new protocols may require refactoring multiple call sites.
  - Mitigation: Provide adapters in storage and a transitional compatibility layer.
- Risk: Schema ID policy change affects downstream consumers expecting prior mappings.
  - Mitigation: Version outputs or provide migration tooling and release notes.
- Risk: Lazy DAG initialization could hide runtime errors until later.
  - Mitigation: Add explicit validation paths and targeted tests for metadata resolution.

## Migration Plan
1. Add protocol interfaces and adapters in storage; update build/export to use protocols.
2. Implement contract policy module and update build/storage providers to use it.
3. Add table-key validation API; update safe_* helpers and tests.
4. Split contract services into schema-only and metadata-enriched layers; refactor
   enumerators to use schema-only by default.
5. Introduce settings/provides injection; update tests to use explicit providers.
6. Update documentation and release notes for breaking changes.

## Open Questions
- Should we introduce a formal contract policy version to support future changes without
  altering schema versions?
- Do we need a migration flag to temporarily allow legacy schema ID derivation during
  rollout?
