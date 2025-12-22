# Change: Refactor MCP serving interfaces to remove noqa suppressions

## Why
MCP serving code and tests currently rely on noqa suppressions for large tool signatures,
async/sync mismatches in health routes, private prompt access, untyped payload parsing,
and hardcoded security fixtures. Removing these suppressions requires structural changes
that also improve concurrency safety, maintainability, and extensibility.

## What Changes
- Introduce request envelopes for semantic_query, semantic_explain, and semantic_export
  using MCP-specific request models that normalize into semantic request models
  (BREAKING: MCP tool input schema changes).
- Refactor MCP tool implementations into focused use-case handlers/workflows with
  explicit dependencies.
- Add async-safe health/ready routes backed by readiness signaling and cached snapshot
  metadata.
- Provide a public prompt registry/introspection API for MCP prompt registration.
- Centralize MCP test payload extraction helpers and security-oriented test fixtures.

## Status Update
All work is complete. MCP tool schemas now require request envelopes with normalized validation,
tool workflows and metrics are centralized, health/ready routes await readiness with cached
metadata, prompt registry introspection is public, tests use shared MCP payload/security helpers,
and docs/examples reflect the updated tool inputs. Quality gates and pytest are green.

## Impact
- Affected specs: serving-interfaces
- Affected code:
  - src/codeintel/serving/mcp/app.py
  - src/codeintel/serving/mcp/tools/*.py
  - src/codeintel/serving/mcp/prompts.py
  - src/codeintel/serving/db/manager.py
  - tests/serving/**
  - tests/_helpers/**
