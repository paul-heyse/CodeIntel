# Change: Refactor MCP serving interfaces to remove noqa suppressions

## Why
MCP serving code and tests currently rely on noqa suppressions for large tool signatures,
async/sync mismatches in health routes, private prompt access, untyped payload parsing,
and hardcoded security fixtures. Removing these suppressions requires structural changes
that also improve concurrency safety, maintainability, and extensibility.

## What Changes
- Introduce request envelopes for semantic_query, semantic_explain, and semantic_export
  using the semantic request models (BREAKING: MCP tool input schema changes).
- Refactor MCP tool implementations into focused use-case handlers/workflows with
  explicit dependencies.
- Add async-safe health/ready routes backed by readiness signaling and cached snapshot
  metadata.
- Provide a public prompt registry/introspection API for MCP prompt registration.
- Centralize MCP test payload extraction helpers and security-oriented test fixtures.

## Impact
- Affected specs: serving-interfaces
- Affected code:
  - src/codeintel/serving/mcp/app.py
  - src/codeintel/serving/mcp/tools/*.py
  - src/codeintel/serving/mcp/prompts.py
  - src/codeintel/serving/db/manager.py
  - tests/serving/**
  - tests/_helpers/**
