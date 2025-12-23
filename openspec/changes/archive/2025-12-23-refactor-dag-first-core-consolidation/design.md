## Context
The codebase has converged on a Hamilton-first architecture, but several subsystems
still maintain parallel registries, serializers, and execution contexts. This creates
duplicate logic, weakens boundaries, and makes it harder to evolve the DAG as a single
source of truth.

## Goals / Non-Goals
- Goals:
  - Make Hamilton DAG execution the only orchestration path for compute workflows.
  - Consolidate registries, contracts, and serialization into canonical services.
  - Strengthen storage boundaries with a single facade API for non-storage callers.
  - Provide a unified ExecutionContext that carries snapshot, settings, paths, and
    run metadata across build, CLI, and serving.
  - Unify observability and error taxonomies across transports.
- Non-Goals:
  - Introducing new external dependencies.
  - Rewriting the Hamilton target graph or changing dataset semantics.
  - Changing user-facing CLI flags beyond what is required for new APIs.

## Decisions
- Decision: Introduce ExecutionContext as the canonical injected runtime object.
  - Why: Centralizes snapshot identity, settings, and runtime primitives for all
    Hamilton targets and transport entrypoints.
- Decision: Create RegistryService to own dataset, semantic, and export discovery.
  - Why: Ensures a single source of truth for metadata that drives build and serving.
- Decision: Create ContractService to compile Pandera/JSON Schema and row serializers.
  - Why: Eliminates divergent schema generation and validation paths.
- Decision: Introduce StorageFacade for non-storage modules.
  - Why: Keeps storage boundaries strong while simplifying caller APIs.
- Decision: Consolidate export serialization around core export formats and shared
  coercion utilities.
  - Why: Guarantees consistent encoding behavior across build/serving/MCP.
- Decision: Introduce a DB span emitter abstraction.
  - Why: Centralizes span attribute composition and enables future DB adapters.
- Decision: Consolidate error taxonomy and mapping through the core catalog.
  - Why: Prevents drift between CLI/HTTP/MCP error surfaces.

## Risks / Trade-offs
- Consolidation may require wide refactors across build, serving, and storage modules.
- Registry and contract service changes may touch many tests and integration points.
- ExecutionContext plumbing could introduce temporary complexity during migration.

## Migration Plan
1. Introduce ExecutionContext and adopt it in build/CLI/serving entrypoints.
2. Implement RegistryService and update discovery and catalog callers.
3. Implement ContractService and migrate schema/serialization consumers.
4. Add StorageFacade and migrate non-storage modules away from gateways/repositories.
5. Consolidate export serialization and observability span emission.
6. Remove legacy registries, serializers, and helper utilities.

## Open Questions
- None. Legacy registries/helpers were removed, tags aligned to the registry, and
  no transitional feature flags were required.
