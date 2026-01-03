# CLI Handler Canonicalization and Schema-First Consolidation Plan

> Status: Proposed
> Author: Codex
> Date: 2026-01-02
> Scope: src/codeintel/cli, src/codeintel/build/schemas, src/codeintel/storage, src/codeintel/serving

---

## Executive summary

This plan consolidates CLI functionality around a single, handler and registry based
operation layer. The CLI becomes a thin adapter that only parses flags and delegates
to registered handlers. Build schemas are the canonical source of dataset contracts,
and storage is a consumer for validation and persistence. Runtime and config resolution
is centralized in core/config and exposed through a narrow CLI adapter.

The result is a smaller CLI surface area, fewer duplicate implementations, consistent
operation IDs, and a clearer separation of concerns between build, storage, serving,
and CLI adapters.

---

## Context and decisions

- Handler and registry path is canonical for operations and business logic.
- Build schemas are canonical for dataset contracts; storage is a downstream consumer.
- CLI should be a thin adapter for runtime and config resolution.

---

## Goals

- Single implementation per operation, registered in the operation registry.
- CLI commands contain no business logic beyond flag parsing and delegation.
- Dataset list, describe, constraints, and flow all reflect build schema canonical data.
- Runtime and config resolution live in core/config; CLI adapts them without duplicating logic.
- Storage gateway lifecycle is centralized in one service with consistent validation behavior.
- DTOs are standardized and serialized consistently via @result_type.

## Non-goals

- Re-architecting Hamilton DAG composition or inference pipeline.
- Changing external behavior beyond necessary aliasing and deprecation.
- Removing storage validation operations that rely on DuckDB state.

---

## Current state summary (primary consolidation targets)

- Duplicate graph operations and DTOs in handlers vs commands:
  - src/codeintel/cli/handlers/graphs.py
  - src/codeintel/cli/commands/graphs.py
- Duplicate jobs logic and DTOs:
  - src/codeintel/cli/handlers/jobs.py
  - src/codeintel/cli/commands/jobs.py
- Duplicate health execution logic:
  - src/codeintel/cli/handlers/health.py
  - src/codeintel/cli/commands/health.py
- Dataset operations split across dataset and datasets groups and multiple contract sources:
  - src/codeintel/cli/handlers/ops.py
  - src/codeintel/cli/handlers/datasets.py
  - src/codeintel/cli/handlers/storage.py
- Gateway lifecycle duplicated between StorageService and handler utilities:
  - src/codeintel/cli/services/storage.py
  - src/codeintel/cli/handlers/_utilities.py
  - src/codeintel/cli/handlers/storage.py
- Runtime/config resolution duplicated across CLI abstractions:
  - src/codeintel/cli/commands/_common.py
  - src/codeintel/cli/resolution/runtime.py
  - src/codeintel/cli/resolution/params.py
  - src/codeintel/config/models.py
  - src/codeintel/cli/project/_project.py

---

## Target architecture (high level)

```
CLI commands (thin adapter)
  -> registry OperationSpec
     -> handler function (canonical)
        -> build/storage/serving services
           -> core runtime/config
```

Key points:
- Handlers define the operation behavior and are registered in the registry.
- CLI commands only gather parameters and invoke the handler via @cli_command or execute_operation.
- Build schemas are the canonical dataset contract layer for list/describe/constraints.
- Storage is a consumer for validation and persistence, not the schema source of truth.
- Runtime/config resolution flows through core/config and is only adapted by CLI.

---

## Workstreams and steps

### Workstream 0: Inventory and guardrails

1. Add a registry inventory utility to detect duplicate operation IDs and conflicting groups.
   - New tool in tools/ or a lightweight module in src/codeintel/cli/introspection/.
   - Output list of duplicates and alias candidates.
2. Add a preflight check that fails CI when duplicate operation IDs are registered.
3. Document canonical operation ID and group naming rules (graph vs graphs, dataset vs datasets).

### Workstream 1: Operation ID canonicalization and thin CLI commands

1. Choose canonical operation IDs (recommend: "graph.*", "jobs.*", "health.*", "dataset.*").
2. Add alias support in the registry for legacy IDs (example: "graphs.targets.list").
   - Provide a small alias map with deprecation notes.
3. Update command modules to be thin wrappers:
   - Replace Command[T] implementations with handler based @cli_command usage.
   - Remove manual register_operation calls where they duplicate @cli_command.
4. Update help/introspection to render canonical IDs and annotate aliases.

Deliverables:
- Registry alias mechanism.
- Single operation ID per behavior.
- CLI commands that delegate to handlers only.

### Workstream 2: Jobs consolidation (handler canonical)

1. Move JobInfo, JobOutput, and related DTOs into a single canonical result module:
   - Prefer src/codeintel/cli/core/result_types.py or a new src/codeintel/cli/results/jobs.py.
2. Update src/codeintel/cli/handlers/jobs.py to use canonical DTOs and @result_type.
3. Update src/codeintel/cli/commands/jobs.py to call handler via @cli_command.
4. Ensure background job execution uses execute_operation on the canonical handler:
   - Update src/codeintel/cli/jobs/_jobs.py to rely on registry and canonical DTOs.
5. Add tests for jobs list/status/output serialization with CLI renderer.

### Workstream 3: Graph operations consolidation (handler canonical)

1. Move GraphTargetInfo, GraphPlanStage, GraphPlanResult into canonical result types.
2. Remove duplicate graph logic in src/codeintel/cli/commands/graphs.py:
   - Keep CLI command definitions only, mapped to handlers in src/codeintel/cli/handlers/graphs.py.
3. Standardize operation IDs to graph.targets.list and graph.targets.plan.
4. Add alias support for graphs.targets.* and update any references.

### Workstream 4: Health consolidation

1. Keep health_check_handler as canonical implementation.
2. Update src/codeintel/cli/commands/health.py to call handler via @cli_command.
3. Remove duplicate logic from commands and ensure registry uses handler spec.
4. Add a single health check test to validate output shape.

### Workstream 5: Dataset consolidation with build schema canonical source

1. Introduce a build schema facade for dataset operations:
   - New module: src/codeintel/build/schemas/dataset_service.py
   - Functions:
     - list_datasets()
     - describe_dataset(table_key)
     - constraints_summary(table_key)
     - flow(table_key, catalog)
     - inferability_inventory(driver, catalog)
2. Update CLI dataset handlers to use the new service:
   - src/codeintel/cli/handlers/ops.py
   - src/codeintel/cli/handlers/datasets.py
3. Ensure CLI dataset list/describe output always uses build schema contracts.
4. Keep storage validation as a distinct operation:
   - dataset.verify stays in storage context and compares against build schema contract.
5. Align dataset and datasets groups:
   - Keep both command groups but share handler implementations and DTOs.
   - Add aliasing to deprecate duplicated commands over time.

### Workstream 6: Runtime/config resolution centralization (CLI thin adapter)

1. Define RuntimeParams as the single canonical input for runtime resolution:
   - src/codeintel/cli/resolution/params.py becomes the CLI boundary only.
2. Ensure runtime resolution flows through core/config:
   - Use codeintel.config.models and core.runtime.loader primitives.
3. Update RuntimeService to accept RuntimeParams directly and delegate to core/config.
4. Reduce duplicate parsing in CLI:
   - Refactor src/codeintel/cli/commands/_common.py to map CLI flags to RuntimeParams.
   - Remove redundant path conversions in decorators where possible.
5. Add tests for runtime resolution with config file, env, and CLI overrides.

### Workstream 7: Storage gateway lifecycle and validation unification

1. Extend StorageService with explicit validation mode support:
   - Add gateway_scope(validation_mode=...) or a specialized method.
2. Update handlers to use StorageService only:
   - src/codeintel/cli/handlers/_utilities.py (deprecate open_handler_gateway, runtime_gateway)
   - src/codeintel/cli/handlers/storage.py (use StorageService from CommandContext)
3. Standardize validation summary path handling in one place.
4. Add a single integration test for read only and write gateway usage.

### Workstream 8: Result type and rendering normalization

1. Move all handler defined DTOs into canonical result types with @result_type.
   - Replace manual to_dict implementations.
2. Update handlers to return canonical DTOs only.
3. Ensure renderer handles all DTOs consistently (no manual dict conversions).
4. Add tests for JSON rendering of nested DTOs.

### Workstream 9: Documentation and migration notes

1. Update CLI docs and help output to reflect canonical operation IDs.
2. Add a migration section for deprecated aliases and command names.
3. Link this plan in existing CLI architecture docs for alignment.

---

## Migration mapping (initial proposal)

| Legacy operation ID | Canonical operation ID | Notes |
| --- | --- | --- |
| graphs.targets.list | graph.targets.list | Alias with deprecation notice |
| graphs.targets.plan | graph.targets.plan | Alias with deprecation notice |
| datasets.list | dataset.list | Shared handler output, keep both temporarily |
| datasets.diff | dataset.diff | Optional alias if kept |
| datasets.snapshot | dataset.snapshot | Optional alias if kept |

---

## Acceptance criteria

- No duplicate business logic across handlers and commands for jobs, graphs, or health.
- Operation registry lists one canonical ID per behavior with explicit aliases only.
- Dataset list/describe/constraints/flow use build schema contracts exclusively.
- CLI commands are thin adapters with no domain logic.
- Runtime/config resolution uses core/config primitives consistently.
- Storage gateway lifecycle is centralized and consistent across handlers.
- Ruff, pyright, and pyrefly pass for all modified files.
- Targeted tests pass for jobs, graphs, dataset handlers, and runtime resolution.

---

## Risks and mitigations

- Risk: Backward compatibility breaks for operation IDs.
  - Mitigation: Provide registry aliasing with deprecation warnings.
- Risk: Dataset output changes when switching to build schemas.
  - Mitigation: Snapshot tests and output comparison before removal of storage contracts.
- Risk: Runtime resolution changes across CLI commands.
  - Mitigation: End to end CLI tests with config and env overrides.

---

## Rollout plan

1. Add registry alias support and duplicate detection tooling.
2. Consolidate jobs, graphs, and health into handler canonical path.
3. Introduce build schema dataset service and migrate dataset operations.
4. Centralize runtime/config resolution and storage gateway lifecycle.
5. Normalize DTOs and update renderers and tests.
6. Remove deprecated duplicates after a full deprecation cycle.
