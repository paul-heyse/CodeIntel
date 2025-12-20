## Context
The system currently derives schema, contract, validation, export, and error behavior from multiple
sources spread across build, storage, serving, and core. This makes provenance unclear, introduces
schema/hash drift, and complicates extension work. We are in a design-phase window where breaking
changes are acceptable and a clean, deterministic boundary is preferred.

## Goals / Non-Goals
- Goals:
  - Establish a single source of truth anchored in the Hamilton global DAG.
  - Make runtime layers build-import-free via artifact-backed providers.
  - Centralize JSON Schema generation, validation, exports, and error envelopes.
  - Enforce deterministic manifests/catalogs with stable hashing and provenance.
- Non-Goals:
  - Rewriting analytics algorithms or changing semantic query behavior.
  - Performance tuning beyond removing duplicative paths.
  - Introducing new external services or infrastructure.

## Decisions
- Decision: Adopt an artifact-first boundary with SchemaManifest + DatasetCatalog + SemanticRegistry
  outputs as the runtime source of truth.
  - Rationale: Enforces determinism, keeps runtime minimal, and unifies schemas/contracts/exports.
- Decision: Canonicalize export format naming to ndjson; treat jsonl as deprecated.
  - Rationale: Aligns naming across tools and eliminates format drift.
- Decision: Standardize error envelopes on RFC 9457 ProblemDetail with a single catalog.
  - Rationale: Consistent error semantics for CLI, HTTP, and MCP.
- Decision: Consolidate all write paths through a single writer facade.
  - Rationale: Ensures consistent validation, hashing, and contract enforcement.

## Alternatives considered
- Keep storage/serving using declared schema registries.
  - Rejected: Continues fragmentation and import-time hazards.
- Maintain jsonl as canonical and map to ndjson only at HTTP.
  - Rejected: Keeps format drift and duplicated mapping logic.
- Partial consolidation (schemas only).
  - Rejected: Contracts/validation/exports would still drift and block extensibility.

## Risks / Trade-offs
- Large breaking changes could invalidate downstream scripts and tests.
- Missing or mismatched artifacts could break runtime loads without clear fallbacks.
- Strict determinism may surface hidden sources of nondeterminism in DAG ordering.

## Migration Plan
1. Emit artifacts alongside existing providers and add compatibility shims.
2. Switch runtime layers to artifact-backed providers with feature flags.
3. Remove jsonl as canonical; retain a temporary alias at input boundaries.
4. Remove legacy providers and finalize strict artifact-only loading.

## Open Questions
- None (design choices are intentional for best-in-class consolidation).
