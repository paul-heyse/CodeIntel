## Context
MCP tools are currently implemented as thin FastMCP adapters but carry noqa suppressions for
large signatures, local complexity, and async/sync mismatches in health routes. Tests also
reach into private FastMCP state, parse tool payloads with Any, and hardcode security-sensitive
values. We need to remove suppressions and align the serving layer with stricter interface and
concurrency standards without altering Hamilton-derived, dynamic output schemas.

## Goals / Non-Goals
- Goals:
  - Remove all remaining MCP-related noqa suppressions by addressing their root causes.
  - Keep MCP transport adapters thin and easy to extend (new tools, caching, export formats).
  - Ensure health/ready routes are async-safe and do not block the event loop.
  - Provide a public prompt registry API for tests and tooling.
  - Consolidate MCP test utilities and security fixtures.
- Non-Goals:
  - Change semantic output payload shapes beyond existing models and hashes.
  - Alter Hamilton DAG compilation or schema artifact generation.
  - Introduce new runtime dependencies or external services.

## Decisions
- Decision: Use request-envelope inputs for semantic_query, semantic_explain, and semantic_export.
  - Rationale: Replacing multi-parameter tool signatures removes PLR0913 while preserving
    strict validation via MCP-specific request models that normalize to SemanticQueryRequest
    and SemanticExportRequest.
  - Notes: semantic_explain reuses the query request envelope; semantic_export adds
    export_format and pagination normalization.
- Decision: Replace inline tool workflows with dedicated handlers/workflows.
  - Rationale: Smaller units remove PLR0914, improve testability, and allow incremental extension
    (e.g., caching, alternate export targets) without bloating tool functions.
- Decision: Add readiness signaling in ServingDBManager and use cached snapshot metadata for
  health/ready routes.
  - Rationale: Avoid blocking the event loop on sync calls while keeping responses consistent and
    inexpensive. Routes await readiness, then return cached summary.
- Decision: Introduce a public prompt registry API in codeintel.serving.mcp.prompts.
  - Rationale: Tests should not access FastMCP private state (SLF001); the registry provides a
    stable introspection surface for prompts.
- Decision: Centralize MCP test helpers and security fixtures under tests/_helpers/.
  - Rationale: Shared helpers remove ANN401 and S104/S106 suppressions while documenting intent.

## Risks / Trade-offs
- Breaking change: MCP tool input schema changes to a single request envelope.
  - Mitigation: Update tests, docs/examples, and announce in serving_meta or release notes.
- Behavioral risk: readiness wait could mask startup issues if not bounded.
  - Mitigation: use timeouts and return 503 when readiness is not achieved.

## Migration Plan
1. Add readiness signaling and cached summary; update health/ready routes. (Done)
2. Implement request envelopes and refactor MCP tool handlers into workflows. (Done)
3. Add prompt registry API and update tests to use it. (Done)
4. Add MCP test helper + security fixtures; update tests accordingly. (Done)
5. Update docs/examples; run quality gates and tests. (Done)

## Implementation Status
All decisions have been implemented. MCP tool input schemas use request envelopes, readiness
signals gate health/ready endpoints, prompt registration is publicly introspectable, and tests
use shared MCP payload + security fixtures. No compatibility layer is retained.
