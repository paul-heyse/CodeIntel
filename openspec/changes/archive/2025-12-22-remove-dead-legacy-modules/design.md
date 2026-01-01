## Status
Archived on 2025-12-22 after implementation. Guardrails/pytest validation is deferred to
the parallel guardrails workstream (schema override checks).

## Context
The codebase still ships unreferenced modules, compatibility shims, and legacy helpers that
are not part of the canonical architecture. Removing them reduces maintenance burden and
clarifies the supported interfaces across build, ingestion, storage, and serving.

## Goals / Non-Goals
- Goals:
  - Remove unused or compatibility-only modules from public packages.
  - Align public interfaces with canonical build, schema, and storage surfaces.
  - Reduce confusion by eliminating legacy CLI entrypoints and helpers.
- Non-Goals:
  - Redesign ingestion, build, or storage pipelines.
  - Remove actively used analytics computation modules (for example,
    codeintel.build.analytics.compute.functions.typedness).
  - Add new features or CLI commands.

## Decisions
- Decision: Delete unused entrypoints and compatibility shims rather than retaining them as
  deprecated stubs to keep the public surface minimal and unambiguous.
- Decision: Treat CLI surfaces as the single registry of supported entrypoints and remove
  standalone helper modules that are not registered.
- Decision: Keep canonical schema registry, tool execution, and semantic compilation APIs,
  and remove legacy or alternate helpers that are not wired into the build pipeline.
- Alternatives considered:
  - Retain modules with deprecation warnings. Rejected to avoid long-term ambiguity and
    maintenance of unused code paths.
  - Keep modules private but shipped. Rejected because it still expands the distribution
    surface and encourages accidental use.

## Risks / Trade-offs
- Risk: External users may import legacy modules directly. Mitigation: document removals,
  provide migration notes, and keep canonical import paths stable.
- Risk: Hidden dynamic imports might rely on removed modules. Mitigation: search for
  importlib usage, run quality checks and test suites before release.

## Migration Plan
1. Remove legacy modules and update any internal references to canonical paths.
2. Update CLI docs/help to reflect the supported command surface.
3. Update tests to use canonical APIs and remove references to deleted modules.
4. Run quality gates and regression tests.
5. Publish release notes for breaking removals.

## Open Questions
None. Legacy surfaces were removed as planned; validation follows the guardrails workstream.
