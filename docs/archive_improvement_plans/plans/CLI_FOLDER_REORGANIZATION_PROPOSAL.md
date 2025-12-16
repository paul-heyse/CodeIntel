# CLI Folder Reorganization Proposal

## Executive Summary

This document proposes a best-in-class folder structure for `src/codeintel/cli/`. The current structure evolved through multiple phases of design and consolidation work, resulting in an ad-hoc organization with files scattered between root level and subfolders, type duplications, and mixed concerns within files. This proposal analyzes the current state from first principles and presents a comprehensive reorganization that addresses structural issues at both the folder and file level.

---

## Current State Analysis

### File Inventory by Location

#### Root Level (38 files, ~13,400 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 50 | Package exports |
| `cli_completions.py` | 362 | Shell completion logic |
| `cli_errors.py` | 441 | RFC 9457 Problem Details support |
| `cli_render.py` | 560 | Unified output rendering |
| `cli_types.py` | 126 | Canonical CLI type definitions |
| `cli_validation.py` | 667 | Input validation framework |
| `command_context.py` | 361 | Unified command context manager |
| `cyclopts_app.py` | 152 | Root Cyclopts application |
| `cyclopts_build.py` | 251 | Build command wiring |
| `cyclopts_common.py` | 303 | Shared Cyclopts primitives |
| `cyclopts_completions.py` | 108 | Completion CLI commands |
| `cyclopts_config.py` | 459 | Config introspection commands |
| `cyclopts_datasets.py` | 348 | Dataset management commands |
| `cyclopts_docs.py` | 222 | Docs export commands |
| `cyclopts_graphs.py` | 137 | Graph commands |
| `cyclopts_health.py` | 58 | Health check commands |
| `cyclopts_help.py` | 288 | Help rendering hardening |
| `cyclopts_help_commands.py` | 106 | Operation discovery commands |
| `cyclopts_history.py` | 133 | History commands |
| `cyclopts_ide.py` | 90 | IDE helper commands |
| `cyclopts_jobs.py` | 213 | Background job commands |
| `cyclopts_ops.py` | 1343 | Op/dataset/serve commands |
| `cyclopts_plugins.py` | 276 | Plugin management commands |
| `cyclopts_storage.py` | 237 | Storage commands |
| `cyclopts_subsystem.py` | 342 | Subsystem exploration commands |
| `dry_run.py` | 157 | Dry-run execution planning |
| `error_taxonomy.py` | 931 | Error taxonomy & factories |
| `errors.py` | 28 | Small error utilities |
| `health.py` | 413 | Health check implementation |
| `help_system.py` | 245 | Help system utilities |
| `introspection.py` | 279 | CLI introspection |
| `job_runner.py` | 74 | Job runner entry point |
| `jobs.py` | 518 | Job management core |
| `observability.py` | 443 | Observability middleware |
| `op_params.py` | 751 | Dynamic parameter introspection |
| `operation_registry.py` | 151 | Operation registry |
| `output.py` | 204 | Output handling utilities |
| `pipelines.py` | 390 | Pipeline execution |
| `project.py` | 483 | Project detection |
| `resilience.py` | 1275 | Retry/circuit breaker patterns |
| `result_types.py` | 1002 | Handler result type definitions |
| `results.py` | 224 | Result abstractions |
| `shell.py` | 501 | Shell utilities |
| `telemetry.py` | 588 | OpenTelemetry integration |

#### Existing Subfolders

| Folder | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `handlers/` | 15 | ~4,800 | Domain-specific handler implementations |
| `execution/` | 8 | ~2,900 | Execution infrastructure (executor, middleware, progress) |
| `config/` | 7 | ~2,200 | Configuration management |
| `resolution/` | 6 | ~1,000 | Runtime resolution |
| `plugins/` | 7 | ~2,100 | Plugin system |
| `rendering/` | 5 | ~900 | Output rendering service |
| `completions/` | 6 | ~700 | Shell completion generators |
| `operations/` | 10 | ~900 | Operation registrations (side-effect imports) |
| `commands/` | 1 | 75 | Just `__init__.py` (underutilized) |
| `options/` | 2 | ~250 | Common CLI options |

---

## Critical Issues Identified

### 1. Type Duplications (Must Eliminate)

| Type | Current Locations | Issue |
|------|-------------------|-------|
| `OutputFormat` | `cli_types.py` + `rendering/types.py` | Duplicate enum |
| `BackendFlags` | `cli_types.py` + `resolution/params.py` | Duplicate dataclass |
| `Shell` enum | `cli_completions.py` + `completions/__init__.py` | Duplicate enum |
| `ColumnSpec/TableSpec` | `cli_render.py` + `rendering/table.py` | Duplicate dataclasses |
| `DocsValidationError` | `cli_errors.py` + `errors.py` | Duplicate exception |

### 2. Files With Mixed Concerns

| File | Issue |
|------|-------|
| `cli_types.py` | Contains types that belong in domain-specific locations |
| `cli_render.py` | Duplicates types from `rendering/` and mixes concerns |
| `telemetry.py` | Mixes provider, config, middleware, and metrics |

### 3. Files That Need Splitting (>1000 lines)

| File | Lines | Action |
|------|-------|--------|
| `cyclopts_ops.py` | 1343 | Split into 3 files (op, dataset, serve) |
| `resilience.py` | 1275 | Split into 4 files (retry, circuit_breaker, middleware, exceptions) |

### 4. Misplaced Files

| File | Current | Should Be |
|------|---------|-----------|
| `command_context.py` | Root | `commands/` (it's command orchestration) |
| `cli_types.py` | Root | Distribute to domain folders, then delete |
| `cli_render.py` | Root | Merge into `rendering/` |

---

## Design Principles

### File Size Guidelines

| Zone | Lines | Action |
|------|-------|--------|
| Red (too small) | <100 | Merge with related file |
| Yellow (borderline small) | 100-250 | Consider merging if natural fit exists |
| Green (ideal) | 250-750 | No action needed |
| Yellow (borderline large) | 750-1000 | Monitor, but acceptable for cohesive modules |
| Red (too large) | >1000 | Split into focused modules |

### Core Principles

1. **Single Source of Truth** - No type duplications
2. **Clear Conceptual Boundaries** - Each folder has one responsibility
3. **Domain-Appropriate Placement** - Types live where they're used
4. **Cohesive Files** - Each file has one clear purpose
5. **Consistent Paradigm** - Functional organization throughout

---

## Recommended Architecture

### Folder Structure Overview

```
cli/
├── commands/           # Cyclopts command definitions + context
├── handlers/           # Handler implementations (KEEP AS-IS)
├── execution/          # Execution infrastructure (KEEP AS-IS)
├── config/             # Configuration (KEEP AS-IS)
├── resolution/         # Runtime resolution (minor additions)
├── plugins/            # Plugin system (KEEP AS-IS)
├── core/               # Fundamental abstractions (results, options, I/O)
├── errors/             # Error taxonomy and handling
├── observability/      # Telemetry and tracing (split from current)
├── resilience/         # Retry and circuit breaker (split from current)
├── introspection/      # CLI/operation introspection
├── completions/        # Shell completions (consolidate duplicates)
├── rendering/          # Output rendering (merge cli_render.py)
├── project/            # Project detection and pipelines (split current)
├── jobs/               # Background job management (split current)
└── shell/              # Interactive shell (split current)
```

---

### Detailed File Organization by Folder

#### `commands/` - Command Definitions and Context

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports all apps | ~100 | New |
| `context.py` | `command_context()` manager | ~360 | `command_context.py` |
| `_common.py` | `RuntimeCLI`, `OutputFormatCLI`, helpers | ~300 | `cyclopts_common.py` |
| `_help.py` | Help rendering hardening | ~290 | `cyclopts_help.py` |
| `app.py` | Root app assembly, `main()` | ~150 | `cyclopts_app.py` |
| `build.py` | Build commands | ~250 | `cyclopts_build.py` |
| `completions.py` | Completion commands | ~110 | `cyclopts_completions.py` |
| `config.py` | Config commands | ~460 | `cyclopts_config.py` |
| `datasets.py` | Dataset commands | ~350 | `cyclopts_datasets.py` |
| `docs.py` | Docs commands | ~220 | `cyclopts_docs.py` |
| `graphs.py` | Graph commands | ~140 | `cyclopts_graphs.py` |
| `health.py` | Health commands | ~60 | `cyclopts_health.py` |
| `help_commands.py` | Operation discovery | ~110 | `cyclopts_help_commands.py` |
| `history.py` | History commands | ~130 | `cyclopts_history.py` |
| `ide.py` | IDE commands | ~90 | `cyclopts_ide.py` |
| `jobs.py` | Job commands | ~210 | `cyclopts_jobs.py` |
| `ops.py` | Op commands | ~450 | Split from `cyclopts_ops.py` |
| `dataset_ops.py` | Dataset ops commands | ~450 | Split from `cyclopts_ops.py` |
| `serve.py` | Serve commands | ~450 | Split from `cyclopts_ops.py` |
| `plugins.py` | Plugin commands | ~280 | `cyclopts_plugins.py` |
| `storage.py` | Storage commands | ~240 | `cyclopts_storage.py` |
| `subsystem.py` | Subsystem commands | ~340 | `cyclopts_subsystem.py` |

**Key change:** `command_context.py` moves here as `context.py` since it's command orchestration, not a core abstraction.

---

#### `core/` - Fundamental Abstractions

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~50 | New |
| `results.py` | `CliResult[T]` generic wrapper | ~220 | `results.py` |
| `result_types.py` | Handler result dataclasses | ~1000 | `result_types.py` |
| `options.py` | Common option definitions | ~230 | `options/common.py` |
| `output.py` | `OutputEnvelope`, stdin/stdout helpers | ~200 | `output.py` |

**Note:** `cli_types.py` is **eliminated** - its types are distributed to domain folders:
- `OutputFormat` → `rendering/types.py` (canonical)
- `BackendFlags`, `RuntimeOptions` → `resolution/params.py` (canonical)
- `RepoSelection`, `PathSelection` → `resolution/types.py`

---

#### `errors/` - Error Infrastructure

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Unified exports | ~60 | New |
| `exceptions.py` | `CliError`, `ValidationError`, exit codes | ~150 | `cli_errors.py` + `errors.py` |
| `taxonomy.py` | `ErrorCategory`, `ErrorCode`, all constants | ~930 | `error_taxonomy.py` |
| `problem_detail.py` | `ProblemDetail`, conversion helpers | ~200 | Split from `cli_errors.py` |
| `handlers.py` | `handle_cli_error()`, `run_handler()` | ~150 | Split from `cli_errors.py` |

---

#### `rendering/` - Output Rendering

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~80 | Existing |
| `types.py` | `OutputFormat`, `RenderContext`, `RenderMode` (canonical) | ~180 | Existing + merge |
| `table.py` | `ColumnSpec`, `TableSpec` (canonical) | ~90 | Existing (dedupe) |
| `renderers.py` | `RichRenderer`, `PlainRenderer`, `get_renderer()` | ~350 | From `cli_render.py` |
| `service.py` | `UnifiedRenderer` orchestration | ~460 | Existing |
| `cli_result.py` | `render_cli_result()` | ~150 | From `cli_render.py` |
| `specs.py` | Table specs for specific result types | ~100 | Existing |

**Key change:** `cli_render.py` content is split and merged here; duplicate type definitions eliminated.

---

#### `resolution/` - Runtime Resolution

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~60 | Existing |
| `types.py` | `ResolvedRuntime`, `RepoSelection`, `PathSelection` | ~150 | Existing + from `cli_types.py` |
| `params.py` | `BackendFlags`, `RuntimeParams`, `RuntimeOptions` (canonical) | ~400 | Existing + from `cli_types.py` |
| `runtime.py` | `RuntimeResolver`, `resolve_runtime()` | ~310 | Existing |
| `gateway.py` | `GatewayManager` | ~110 | Existing |
| `errors.py` | `ResolutionError` | ~60 | Existing |

**Key change:** Types from `cli_types.py` move here as canonical location.

---

#### `observability/` - Telemetry and Observability

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~50 | New |
| `config.py` | `TelemetryConfig`, `ObservabilityConfig` | ~80 | Split from `telemetry.py` |
| `provider.py` | `TelemetryProvider`, span wrappers | ~300 | Split from `telemetry.py` |
| `metrics.py` | `OperationMetrics`, metric collection | ~150 | Split from `telemetry.py` |
| `middleware.py` | `ObservabilityMiddleware`, `TracingMiddleware` | ~400 | `observability.py` + from `telemetry.py` |
| `logging.py` | `StructuredLogFormatter`, logging config | ~100 | Split from `observability.py` |

**Key change:** `telemetry.py` (588 lines) split into focused modules.

---

#### `resilience/` - Resilience Patterns

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports + `ResilienceConfig` | ~80 | New |
| `exceptions.py` | `RetryableError`, `CircuitOpenError` | ~60 | Split from `resilience.py` |
| `retry.py` | `RetryPolicy`, `RetryContext`, decorators | ~400 | Split from `resilience.py` |
| `circuit_breaker.py` | `CircuitBreaker`, `CircuitState`, registry | ~400 | Split from `resilience.py` |
| `middleware.py` | `ResilienceMiddleware` | ~350 | Split from `resilience.py` |

---

#### `introspection/` - CLI/Operation Introspection

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~50 | New |
| `registry.py` | `OperationRegistry`, `register_operation()` | ~150 | `operation_registry.py` |
| `params.py` | `CliParamSpec`, parameter classification | ~750 | `op_params.py` |
| `discovery.py` | `OperationInfo`, `list_operations()` | ~280 | `introspection.py` |
| `help.py` | `HelpRenderer` | ~250 | `help_system.py` |
| `validation.py` | `Validator` classes, `ValidationSchema` | ~670 | `cli_validation.py` |

---

#### `completions/` - Shell Completions

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | `Shell` enum (canonical), `generate_completion()` | ~120 | Existing + merge |
| `model.py` | `CompletionModel`, `CompletionSpec` | ~290 | Existing |
| `install.py` | Installation, detection, path helpers | ~250 | From `cli_completions.py` |
| `bash.py` | Bash generator | ~110 | `bash_generator.py` |
| `fish.py` | Fish generator | ~120 | `fish_generator.py` |
| `zsh.py` | Zsh generator | ~150 | `zsh_generator.py` |
| `powershell.py` | PowerShell generator | ~100 | `powershell_generator.py` |

**Key change:** `Shell` enum duplication eliminated; `cli_completions.py` content merged.

---

#### `project/` - Project and Pipeline Management

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~50 | New |
| `config.py` | `ProjectConfig`, section configs | ~200 | Split from `project.py` |
| `detection.py` | `find_project_root()`, git detection | ~200 | Split from `project.py` |
| `runtime.py` | `ProjectRuntime`, `build_project_runtime()` | ~100 | Split from `project.py` |
| `pipelines.py` | `PipelineConfig`, batch execution | ~390 | `pipelines.py` |
| `dry_run.py` | Dry run planning and rendering | ~160 | `dry_run.py` |

---

#### `jobs/` - Background Job Management

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~50 | New |
| `models.py` | `JobStatus`, `JobInfo` | ~100 | Split from `jobs.py` |
| `store.py` | `JobStore` | ~150 | Split from `jobs.py` |
| `manager.py` | `JobManager`, `get_job_manager()` | ~270 | Split from `jobs.py` |
| `runner.py` | Entry point `main()` | ~80 | `job_runner.py` |

---

#### `shell/` - Interactive Shell

| File | Content | Lines (Est.) | Source |
|------|---------|--------------|--------|
| `__init__.py` | Exports | ~30 | New |
| `session.py` | `ShellSession` | ~150 | Split from `shell.py` |
| `completer.py` | `ShellCompleter` | ~150 | Split from `shell.py` |
| `interactive.py` | `InteractiveShell`, `start_shell()` | ~200 | Split from `shell.py` |

---

#### Folders Kept As-Is

| Folder | Reason |
|--------|--------|
| `handlers/` | Well organized, clear domain structure |
| `execution/` | Well organized, cohesive responsibility |
| `config/` | Well organized, appropriate file sizes |
| `plugins/` | Well organized, clear module boundaries |

---

## Type Canonical Locations (Single Source of Truth)

After reorganization, each type has exactly one definition:

| Type | Canonical Location |
|------|-------------------|
| `OutputFormat` | `rendering/types.py` |
| `RenderContext` | `rendering/types.py` |
| `ColumnSpec`, `TableSpec` | `rendering/table.py` |
| `BackendFlags` | `resolution/params.py` |
| `RuntimeParams` | `resolution/params.py` |
| `RuntimeOptions` | `resolution/params.py` |
| `ResolvedRuntime` | `resolution/types.py` |
| `RepoSelection`, `PathSelection` | `resolution/types.py` |
| `Shell` | `completions/__init__.py` |
| `CliResult[T]` | `core/results.py` |
| `ProblemDetail` | `errors/problem_detail.py` |
| `CliError`, `ValidationError` | `errors/exceptions.py` |
| `ErrorCategory`, `ErrorCode` | `errors/taxonomy.py` |

---

## Files to Delete After Migration

| File | Reason |
|------|--------|
| `cli_types.py` | Contents distributed to domain folders |
| `cli_render.py` | Contents merged into `rendering/` |
| `cli_completions.py` | Contents merged into `completions/` |
| `errors.py` | Contents merged into `errors/exceptions.py` |

---

## Implementation Phases

### Phase 1: Eliminate Type Duplications (Critical)

1. Choose canonical location for each duplicated type
2. Update all imports to use canonical location
3. Delete duplicate definitions
4. Run `ruff check --fix` and verify with `pyright`

**Types to consolidate:**
- `OutputFormat` → keep in `rendering/types.py`
- `BackendFlags` → keep in `resolution/params.py`
- `Shell` → keep in `completions/__init__.py`
- `ColumnSpec/TableSpec` → keep in `rendering/table.py`
- `DocsValidationError` → keep in `errors/exceptions.py`

### Phase 2: Create New Folder Structure

1. Create folders: `core/`, `errors/`, `observability/`, `resilience/`, `introspection/`, `project/`, `jobs/`, `shell/`
2. Create `__init__.py` files with appropriate exports

### Phase 3: Split Oversized Files

1. `cyclopts_ops.py` (1343) → `commands/ops.py`, `commands/dataset_ops.py`, `commands/serve.py`
2. `resilience.py` (1275) → `resilience/retry.py`, `resilience/circuit_breaker.py`, `resilience/middleware.py`, `resilience/exceptions.py`
3. `telemetry.py` (588) → `observability/provider.py`, `observability/metrics.py`, `observability/config.py`
4. `project.py` (483) → `project/config.py`, `project/detection.py`, `project/runtime.py`
5. `jobs.py` (518) → `jobs/models.py`, `jobs/store.py`, `jobs/manager.py`
6. `shell.py` (501) → `shell/session.py`, `shell/completer.py`, `shell/interactive.py`

### Phase 4: Move and Merge Files

1. Move cyclopts files to `commands/` (rename to drop `cyclopts_` prefix)
2. Move `command_context.py` to `commands/context.py`
3. Merge `cli_render.py` into `rendering/`
4. Merge `cli_errors.py` into `errors/`
5. Move remaining files per the plan

### Phase 5: Update All Imports

1. Run `ruff check --fix` for automatic fixes
2. Manually fix complex import patterns
3. Verify with `pyright` and `pyrefly`

### Phase 6: Deprecation Shims

Add re-exports from old locations with deprecation warnings:

```python
# Old location: codeintel/cli/cyclopts_build.py
from __future__ import annotations

import warnings

warnings.warn(
    "codeintel.cli.cyclopts_build is deprecated, "
    "use codeintel.cli.commands.build instead",
    DeprecationWarning,
    stacklevel=2,
)

from codeintel.cli.commands.build import *  # noqa: F401, F403
```

### Phase 7: Cleanup

1. Remove deprecation shims after one release cycle
2. Delete empty/stub files
3. Update documentation

---

## Import Path Changes

### Before and After Examples

| Before | After |
|--------|-------|
| `from codeintel.cli.cyclopts_build import build_app` | `from codeintel.cli.commands import build_app` |
| `from codeintel.cli.cli_errors import ProblemDetail` | `from codeintel.cli.errors import ProblemDetail` |
| `from codeintel.cli.cli_types import OutputFormat` | `from codeintel.cli.rendering import OutputFormat` |
| `from codeintel.cli.cli_types import BackendFlags` | `from codeintel.cli.resolution import BackendFlags` |
| `from codeintel.cli.command_context import command_context` | `from codeintel.cli.commands import command_context` |
| `from codeintel.cli.resilience import RetryPolicy` | `from codeintel.cli.resilience import RetryPolicy` |
| `from codeintel.cli.telemetry import TelemetryProvider` | `from codeintel.cli.observability import TelemetryProvider` |
| `from codeintel.cli.jobs import JobManager` | `from codeintel.cli.jobs import JobManager` |

### Primary Import Points

After reorganization, these are the main entry points:

- `codeintel.cli.commands` - All command apps and context
- `codeintel.cli.handlers` - All handlers and handler results
- `codeintel.cli.core` - Results, options, I/O utilities
- `codeintel.cli.errors` - All error handling
- `codeintel.cli.rendering` - Output formatting and types
- `codeintel.cli.execution` - Execution infrastructure
- `codeintel.cli.config` - Configuration
- `codeintel.cli.resolution` - Runtime resolution and backend flags
- `codeintel.cli.observability` - Telemetry and tracing
- `codeintel.cli.resilience` - Retry and circuit breaker
- `codeintel.cli.introspection` - Operation discovery and validation
- `codeintel.cli.completions` - Shell completion generation
- `codeintel.cli.project` - Project detection and pipelines
- `codeintel.cli.jobs` - Background job management
- `codeintel.cli.shell` - Interactive shell

---

## Folder Descriptions

| Folder | Responsibility |
|--------|---------------|
| `commands/` | Cyclopts command definitions, wiring, and command context |
| `handlers/` | Business logic for each command domain |
| `execution/` | Operation execution infrastructure |
| `config/` | Configuration loading, validation, schema |
| `resolution/` | Runtime resolution (project, paths, gateway, backend flags) |
| `plugins/` | Plugin discovery, loading, sandboxing |
| `core/` | Fundamental abstractions (results, options, I/O) |
| `errors/` | Error taxonomy, problem details, exception handling |
| `observability/` | Telemetry, metrics, tracing, structured logging |
| `resilience/` | Retry policies, circuit breaker, resilience middleware |
| `introspection/` | Operation registry, parameter introspection, validation, help |
| `completions/` | Shell completion model and generators |
| `rendering/` | Output formatting, renderers, table specs |
| `project/` | Project detection, configuration, pipelines, dry run |
| `jobs/` | Background job store, manager, runner |
| `shell/` | Interactive shell session and completion |

---

## Summary of Changes from Current State

### Files Eliminated (Contents Distributed)
- `cli_types.py` → types move to `rendering/` and `resolution/`
- `cli_render.py` → merge into `rendering/`
- `cli_completions.py` → merge into `completions/`
- `errors.py` → merge into `errors/`

### Files Split
- `cyclopts_ops.py` (1343) → 3 files
- `resilience.py` (1275) → 4 files
- `telemetry.py` (588) → 4 files
- `project.py` (483) → 3 files
- `jobs.py` (518) → 3 files
- `shell.py` (501) → 3 files
- `cli_errors.py` (441) → 3 files

### Files Moved (Renamed)
- 17 `cyclopts_*.py` → `commands/*.py`
- `command_context.py` → `commands/context.py`
- Various files to new folders per plan

### Duplications Eliminated
- 5 type duplications consolidated to single sources of truth

---

## Next Steps

1. Review and approve this proposal
2. Create implementation tickets for each phase
3. Begin with Phase 1 (type duplication elimination) - lowest risk, high value
4. Run full test suite after each phase
5. Update documentation and CHANGELOG
6. Remove deprecation shims after one release cycle
