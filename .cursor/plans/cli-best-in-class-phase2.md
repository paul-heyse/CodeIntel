# CLI Best-in-Class Implementation Plan (Phase 2)

## Executive Summary

This plan details seven phases of enhancements to elevate the CodeIntel CLI to best-in-class status. Building on the foundation established in Phase 1 (shell completion, structured errors, CliResult protocol, pipeable I/O infrastructure, config introspection, command aliases, and simplified help rendering), this plan completes the vision by:

1. **Unifying type definitions** to eliminate duplication and confusion
2. **Wiring stdin support** to complete the pipeable composition story
3. **Migrating handlers** to the structured CliResult pattern
4. **Adding dry-run mode** for safe operation previewing
5. **Implementing middleware** for consistent observability
6. **Adding progress reporting** for long-running operations
7. **Creating a validation layer** for robust input handling

---

## Design Principles

### Best-in-Class Features
- **Composability**: Commands can be piped together (`cmd1 --json | cmd2 --from-stdin`)
- **Introspection**: Users can inspect configuration, dry-run operations, and understand system state
- **Progressive Disclosure**: Simple defaults with power available via flags and aliases
- **Structured Output**: Machine-readable JSON output with RFC 9457 error semantics

### Hardness (Robustness)
- **Fail Fast**: Validate inputs before expensive operations
- **Graceful Degradation**: Informative errors when dependencies are unavailable
- **Defensive Defaults**: Safe behaviors unless explicitly overridden
- **Timeout Awareness**: Long operations report progress and can be interrupted cleanly

### Extensibility
- **Middleware Pattern**: Cross-cutting concerns (logging, metrics, tracing) are pluggable
- **Validation Rules**: Custom validators can be registered per-operation
- **Handler Protocol**: New handlers follow a consistent pattern that enables composition

### Maintainability
- **Single Source of Truth**: One canonical definition for each type
- **Separation of Concerns**: Parsing, validation, execution, and rendering are distinct layers
- **Testability**: All components can be tested in isolation without subprocess spawning

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                              │
│                        cyclopts_app.py                               │
├─────────────────────────────────────────────────────────────────────┤
│                      Parsing Layer (Cyclopts)                        │
│  cyclopts_ops.py | cyclopts_build.py | cyclopts_config.py | ...     │
├─────────────────────────────────────────────────────────────────────┤
│                      Validation Layer (NEW)                          │
│                        cli_validation.py                             │
├─────────────────────────────────────────────────────────────────────┤
│                      Middleware Layer (NEW)                          │
│                        cli_middleware.py                             │
├─────────────────────────────────────────────────────────────────────┤
│                       Handler Layer                                  │
│  ops_handlers.py | build_handlers.py | docs_handlers.py | ...       │
├─────────────────────────────────────────────────────────────────────┤
│                       Output Layer                                   │
│              results.py | output.py | cli_errors.py                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Unify Type Definitions

### Goal
Eliminate duplicate type definitions across CLI modules to establish a single source of truth.

### Current State
- `OutputFormat` defined in both `cli_errors.py` and `common_handlers.py`
- Multiple `RuntimeCliOptions` variations in `common_handlers.py`, `datasets_handlers.py`, `subsystem_handlers.py`
- Inconsistent imports across modules
- Storage exceptions (`StorageError`, `StorageConnectionError`, etc.) defined in storage layer but not re-exported for CLI convenience

### Implementation

#### 1.1 Create Canonical Types Module

Create `src/codeintel/cli/cli_types.py`:

```python
"""Canonical CLI type definitions.

This module is the single source of truth for all CLI-related types.
Other modules should import from here rather than defining their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class OutputFormat(Enum):
    """Output rendering format for CLI commands."""

    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class BackendFlags:
    """Backend preferences provided via CLI."""

    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RuntimeOptions:
    """Unified runtime discovery and backend options.

    This is the canonical runtime options structure used across all CLI modules.
    """

    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)


@dataclass(frozen=True)
class RepoSelection:
    """Repository identification inputs."""

    repo: str | None
    commit: str | None


@dataclass(frozen=True)
class PathSelection:
    """Repository path inputs for storage and builds."""

    repo_root: Path | None
    db_path: Path | None
    build_dir: Path | None
    document_output_dir: Path | None = None


# Type alias for help level
HelpLevel = Literal["brief", "full"]


__all__ = [
    "BackendFlags",
    "HelpLevel",
    "OutputFormat",
    "PathSelection",
    "RepoSelection",
    "RuntimeOptions",
]
```

#### 1.2 Update cli_errors.py

Remove `OutputFormat` definition, import from `cli_types.py`:

```python
# Before
class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"

# After
from codeintel.cli.cli_types import OutputFormat
```

Update `__all__` to re-export for backward compatibility:

```python
from codeintel.cli.cli_types import OutputFormat

__all__ = [
    # ... existing exports ...
    "OutputFormat",  # Re-exported from cli_types
]
```

#### 1.3 Update common_handlers.py

Remove duplicate `OutputFormat` and `RuntimeCliOptions`:

```python
# Before
class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"

@dataclass(frozen=True)
class RuntimeCliOptions:
    ...

# After
from codeintel.cli.cli_types import OutputFormat, RuntimeOptions

# Alias for backward compatibility during transition
RuntimeCliOptions = RuntimeOptions
```

#### 1.4 Update All Importing Modules

Update imports in these files:
- `ops_handlers.py`
- `build_handlers.py`
- `docs_handlers.py`
- `datasets_handlers.py`
- `subsystem_handlers.py`
- `graphs_handlers.py`
- `history_handlers.py`
- `ide_handlers.py`

For each file, change:
```python
# Before
from codeintel.cli.common_handlers import OutputFormat

# After
from codeintel.cli.cli_types import OutputFormat
```

#### 1.5 Re-export Storage Exceptions

Add storage exception re-exports to `cli_errors.py` for convenient access from handlers:

```python
# In cli_errors.py - add to imports section
from codeintel.storage.exceptions import (
    QueryError as StorageQueryError,
    SchemaError as StorageSchemaError,
    StorageConnectionError,
    StorageError,
)

# Update __all__ to include storage exceptions
__all__ = [
    # ... existing exports ...
    "StorageConnectionError",
    "StorageError",
    "StorageQueryError",
    "StorageSchemaError",
]
```

This provides handlers with a single import location for all error types:

```python
# Handlers can now do:
from codeintel.cli.cli_errors import (
    ProblemDetail,
    StorageConnectionError,
    ValidationError,
)

# Instead of importing from multiple locations
```

The storage exceptions integrate with RFC 9457 Problem Details:

```python
# In handler error handling:
except StorageConnectionError as exc:
    return CliResult.fail(ProblemDetail(
        type=ErrorType.STORAGE,
        title="Storage connection failed",
        detail=str(exc),
    ))
```

#### 1.6 Verification

Run quality checks:
```bash
uv run ruff check --fix src/codeintel/cli/
uv run pyright src/codeintel/cli/
uv run pytest tests/cli/ -v
```

### Deliverables
- [ ] `src/codeintel/cli/cli_types.py` created
- [ ] `cli_errors.py` updated to import from `cli_types`
- [ ] `cli_errors.py` updated to re-export storage exceptions
- [ ] `common_handlers.py` updated to import from `cli_types`
- [ ] All handler modules updated
- [ ] All tests pass

---

## Phase 2: Wire stdin Support into Dynamic Operations

### Goal
Enable pipeable composition by connecting the `output.py` stdin utilities to dynamic operation execution.

### Current State
- `output.py` provides `read_stdin_records()`, `iter_stdin_records()`, `merge_stdin_with_args()`
- Dynamic operations in `cyclopts_ops.py` don't accept stdin input
- Users cannot pipe JSON output from one operation to another

### Implementation

#### 2.1 Add --from-stdin Parameter to Operation Config

Update `_make_operation_params_dataclass()` in `cyclopts_ops.py`:

```python
def _make_operation_params_dataclass(metadata: OperationCliMetadata) -> type:
    """Create a dataclass for the operation's CLI parameters."""
    fields: list[tuple[str, type, Any]] = []

    # Add runtime field
    fields.append(runtime_field())

    # Add output format field
    fields.append((
        "output",
        Annotated[OutputFormatCLI | None, Parameter(name="*")],
        None,
    ))

    # Add skip_prereqs field
    fields.append((
        "skip_prereqs",
        Annotated[bool, Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite operations.",
            negative=(),
        )],
        False,
    ))

    # NEW: Add from_stdin field
    fields.append((
        "from_stdin",
        Annotated[bool, Parameter(
            name="--from-stdin",
            help="Read input records from stdin (JSON or JSONL).",
            negative=(),
        )],
        False,
    ))

    # Add operation-specific parameters
    for spec in metadata.params:
        fields.append(build_param_field_for_spec(spec))

    return make_dataclass(
        f"{metadata.cli_name.replace('-', '_').title()}Params",
        fields,
        frozen=True,
    )
```

#### 2.2 Update Dynamic Operation Execution

Modify the `dynamic_op` closure in `_register_dynamic_operation()`:

```python
from codeintel.cli.output import iter_stdin_records, merge_stdin_with_args

def _register_dynamic_operation(metadata: OperationCliMetadata) -> None:
    """Register a dynamic subcommand for an operation."""
    command_name = metadata.cli_name
    if command_name in _REGISTERED_OP_COMMANDS:
        return

    params_cls = _make_operation_params_dataclass(metadata)
    cfg_annotation = Annotated[params_cls, Parameter(name="*")]

    def dynamic_op(cfg: OperationCliArgs | None = None) -> None:
        if cfg is None:
            message = "Operation parameters are required."
            raise ValidationError(message)

        typed_cfg = cfg
        runtime_cli = typed_cfg.runtime
        runtime = _runtime_from_cli(runtime_cli)
        verbose = bool(get_verbose(runtime_cli))

        # Handle stdin input for pipeable composition
        if getattr(typed_cfg, "from_stdin", False):
            _execute_from_stdin(metadata, typed_cfg, runtime, verbose)
        else:
            params = _build_params_dict(typed_cfg, metadata.params)
            _invoke_operation_with_prereqs(
                metadata.operation.id,
                params,
                runtime,
                skip_prereqs=typed_cfg.skip_prereqs,
                verbose=verbose,
            )

    dynamic_op.__annotations__["cfg"] = cfg_annotation

    aliases = _get_aliases_for_operation(command_name)
    op_app.command(
        name=command_name,
        alias=aliases if aliases else None,
        help=metadata.operation.summary or metadata.operation.id,
    )(dynamic_op)
    _REGISTERED_OP_COMMANDS.add(command_name)


def _execute_from_stdin(
    metadata: OperationCliMetadata,
    cfg: OperationCliArgs,
    runtime: ProjectRuntime,
    verbose: bool,
) -> None:
    """Execute operation for each record from stdin.

    Parameters
    ----------
    metadata
        Operation CLI metadata.
    cfg
        CLI configuration with explicit arguments.
    runtime
        Project runtime context.
    verbose
        Whether to emit verbose output.
    """
    cli_args = _build_params_dict(cfg, metadata.params)
    results: list[dict[str, object]] = []

    for stdin_record in iter_stdin_records():
        # CLI args override stdin values
        merged_params = merge_stdin_with_args(stdin_record, cli_args)

        try:
            result = _invoke_operation_for_result(
                metadata.operation.id,
                merged_params,
                runtime,
                skip_prereqs=getattr(cfg, "skip_prereqs", False),
                verbose=verbose,
            )
            results.append({"input": stdin_record, "result": result, "success": True})
        except Exception as exc:  # noqa: BLE001
            results.append({
                "input": stdin_record,
                "error": str(exc),
                "success": False,
            })

    # Output all results as JSON array
    output_format = get_output_format(getattr(cfg, "output", None))
    envelope = OutputEnvelope(
        data=results,
        metadata={"operation": metadata.operation.id, "count": len(results)},
    )
    envelope.write(output_format, sys.stdout)
```

#### 2.3 Add Helper for Result Capture

Add `_invoke_operation_for_result()` that captures the result instead of printing:

```python
def _invoke_operation_for_result(
    op_id: str,
    params: dict[str, Any],
    runtime: ProjectRuntime,
    *,
    skip_prereqs: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Invoke an operation and return the result as a dictionary.

    Returns
    -------
    dict[str, Any]
        The operation result.

    Raises
    ------
    ValidationError
        When the operation fails.
    """
    operation = get_operation(op_id)
    if operation is None:
        message = f"Unknown operation: {op_id}"
        raise ValidationError(message)

    if not skip_prereqs:
        run_operation_prereqs(op_id, params, runtime.gateway)

    stack = build_service_stack(runtime.gateway)
    backend_method = stack.get_method(operation.backend_method)
    if backend_method is None:
        message = f"Backend method not found: {operation.backend_method}"
        raise ValidationError(message)

    result = backend_method(**params)

    # Convert result to dict if needed
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return result.__dict__
    return {"value": result}
```

#### 2.4 Update Imports

Add required imports to `cyclopts_ops.py`:

```python
from codeintel.cli.output import (
    OutputEnvelope,
    iter_stdin_records,
    merge_stdin_with_args,
)
```

#### 2.5 Add Tests

Create `tests/cli/test_stdin_composition.py`:

```python
"""Tests for stdin-based operation composition."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest

from codeintel.cli.cyclopts_ops import op_app, register_dynamic_operations


@pytest.fixture(autouse=True)
def _setup_operations() -> None:
    """Ensure operations are registered."""
    register_dynamic_operations()


def test_from_stdin_single_record() -> None:
    """Verify --from-stdin processes a single JSON record."""
    stdin_data = json.dumps({"goid": "test.func1"})

    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("codeintel.cli.cyclopts_ops._invoke_operation_for_result") as mock:
            mock.return_value = {"summary": "Test function"}
            op_app(["function-summary", "--from-stdin"])
            mock.assert_called_once()


def test_from_stdin_jsonl() -> None:
    """Verify --from-stdin processes JSONL input."""
    stdin_data = '{"goid": "func1"}\n{"goid": "func2"}\n'

    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("codeintel.cli.cyclopts_ops._invoke_operation_for_result") as mock:
            mock.return_value = {"summary": "Test"}
            op_app(["function-summary", "--from-stdin"])
            assert mock.call_count == 2


def test_from_stdin_cli_args_override() -> None:
    """Verify CLI arguments override stdin values."""
    stdin_data = json.dumps({"goid": "stdin.func", "verbose": False})

    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("codeintel.cli.cyclopts_ops._invoke_operation_for_result") as mock:
            mock.return_value = {}
            # CLI --goid should override stdin goid
            op_app(["function-summary", "--from-stdin", "--goid", "cli.func"])
            call_args = mock.call_args
            assert call_args[1]["goid"] == "cli.func"
```

### Deliverables
- [ ] `from_stdin` field added to operation params dataclass
- [ ] `_execute_from_stdin()` implemented
- [ ] `_invoke_operation_for_result()` implemented
- [ ] Imports updated
- [ ] Tests added and passing

---

## Phase 3: Migrate Handlers to CliResult Pattern

### Goal
Migrate all CLI handlers to return `CliResult[T]` instead of printing directly, enabling composition and testing.

### Current State
- `CliResult` protocol exists in `results.py`
- `run_structured_handler()` exists in `cli_errors.py`
- All handlers return `None` and print to stdout directly

### Implementation Strategy

Handlers will be migrated in groups by module. Each migration follows this pattern:

1. Define a result dataclass for the handler's output
2. Change return type from `None` to `CliResult[ResultType]`
3. Replace `print()`/`stdout.write()` with `return CliResult.ok(data)`
4. Update the Cyclopts command to use `run_structured_handler()`

#### 3.1 Create Result Types Module

Create `src/codeintel/cli/result_types.py`:

```python
"""Result type definitions for CLI handlers.

Each handler that returns structured data should have a corresponding
result type defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationListResult:
    """Result from op list command."""

    operations: list[dict[str, str]]
    count: int


@dataclass(frozen=True)
class OperationCallResult:
    """Result from op call command."""

    operation_id: str
    result: dict[str, Any]


@dataclass(frozen=True)
class DatasetListResult:
    """Result from dataset list command."""

    datasets: list[dict[str, str]]
    count: int


@dataclass(frozen=True)
class DatasetDescribeResult:
    """Result from dataset describe command."""

    table_key: str
    columns: list[dict[str, str]]
    row_count: int | None


@dataclass(frozen=True)
class BuildStatusResult:
    """Result from build status command."""

    targets: list[dict[str, Any]]
    stale_count: int
    fresh_count: int


@dataclass(frozen=True)
class BuildRunResult:
    """Result from build run command."""

    executed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_seconds: float


@dataclass(frozen=True)
class SubsystemListResult:
    """Result from subsystem list command."""

    subsystems: list[dict[str, Any]]
    count: int


@dataclass(frozen=True)
class ConfigShowResult:
    """Result from config show command."""

    config: dict[str, Any]
    sources: dict[str, list[str]]


__all__ = [
    "BuildRunResult",
    "BuildStatusResult",
    "ConfigShowResult",
    "DatasetDescribeResult",
    "DatasetListResult",
    "OperationCallResult",
    "OperationListResult",
    "SubsystemListResult",
]
```

#### 3.2 Migrate ops_handlers.py

**Before:**
```python
def op_list_handler(
    *,
    category: str | None,
    output_format: OutputFormat,
) -> None:
    """List available serving operations."""
    stdout = sys.stdout
    operations = list(iter_operations())
    if category:
        operations = [op for op in operations if op.category == category]

    if output_format is OutputFormat.JSON:
        output = [
            {
                "id": op.id,
                "category": op.category,
                "summary": op.summary,
                "http_path": op.http_path,
                "tool_name": op.tool_name,
            }
            for op in operations
        ]
        stdout.write(json.dumps(output, indent=2))
        stdout.write("\n")
    else:
        stdout.write(f"Available operations ({len(operations)}):\n")
        for op in sorted(operations, key=lambda o: o.id):
            stdout.write(f"  {op.id:<35} {op.summary}\n")
```

**After:**
```python
from codeintel.cli.results import CliResult
from codeintel.cli.result_types import OperationListResult


def op_list_handler(
    *,
    category: str | None,
) -> CliResult[OperationListResult]:
    """List available serving operations.

    Returns
    -------
    CliResult[OperationListResult]
        List of operations matching the filter.
    """
    operations = list(iter_operations())
    if category:
        operations = [op for op in operations if op.category == category]

    operation_dicts = [
        {
            "id": op.id,
            "category": op.category,
            "summary": op.summary,
            "http_path": op.http_path,
            "tool_name": op.tool_name,
        }
        for op in sorted(operations, key=lambda o: o.id)
    ]

    return CliResult.ok(
        OperationListResult(operations=operation_dicts, count=len(operations))
    )
```

**Update Cyclopts command:**
```python
@op_app.command(name="list")
@dataclass
class OpListCommand:
    """List available serving operations."""

    cfg: Annotated[OpListCli | None, Parameter(name="*")] = None

    def __call__(self) -> None:
        """Execute the command."""
        category = self.cfg.category if self.cfg else None
        output_format = get_output_format(self.cfg.output if self.cfg else None)

        run_structured_handler(
            op_list_handler,
            category=category,
            output_format=output_format,
        )
```

#### 3.3 Add Custom Text Renderer Support

Update `run_structured_handler()` to accept a text renderer:

```python
def run_structured_handler[ResultT](
    handler: Callable[..., CliResult[ResultT]],
    *args: object,
    output_format: OutputFormat = OutputFormat.TEXT,
    text_renderer: TextRenderer | None = None,
    **kwargs: object,
) -> None:
    """Execute a handler that returns CliResult with structured output.

    Parameters
    ----------
    handler
        Handler function returning CliResult.
    *args
        Positional arguments for the handler.
    output_format
        Output format (TEXT or JSON).
    text_renderer
        Optional custom text renderer for non-JSON output.
    **kwargs
        Keyword arguments for the handler.
    """
    try:
        result: CliResult[ResultT] = handler(*args, **kwargs)
        result.render(output_format, sys.stdout, text_renderer=text_renderer)
        if not result.success:
            raise SystemExit(CLI_EXIT_VALIDATION)
    except ValidationError as exc:
        # ... error handling ...
```

#### 3.4 Migration Order

Migrate handlers in this order (simplest to most complex):

1. **ops_handlers.py** (6 handlers)
   - `op_list_handler` - Simple list
   - `op_call_handler` - Operation invocation
   - `dataset_list_handler` - Dataset listing
   - `dataset_describe_handler` - Schema description
   - `dataset_verify_handler` - Verification
   - `serve_http_handler` / `serve_mcp_handler` - Server start (special case)

2. **build_handlers.py** (3 handlers)
   - `build_status_handler`
   - `build_run_handler`
   - `build_history_handler`

3. **subsystem_handlers.py** (6 handlers)
   - `subsystem_list_handler`
   - `subsystem_show_handler`
   - `subsystem_profiles_handler`
   - `subsystem_coverage_handler`
   - `subsystem_module_memberships_handler`

4. **graphs_handlers.py** (1 handler)
   - `graph_plugins_handler`

5. **docs_handlers.py** (1 handler)
   - `docs_export_handler`

6. **datasets_handlers.py** (9 handlers)
   - All dataset management handlers

7. **history_handlers.py** (1 handler)
   - `history_timeseries_handler`

8. **ide_handlers.py** (1 handler)
   - `ide_hints_handler`

#### 3.5 Backward Compatibility

During migration, maintain backward compatibility by:

1. Keep old signature available with deprecation warning
2. Add `@deprecated` decorator pointing to new pattern
3. Update all internal callers
4. Remove deprecated versions after one release cycle

### Deliverables
- [ ] `result_types.py` created with all result dataclasses
- [ ] All handlers in `ops_handlers.py` migrated
- [ ] All handlers in `build_handlers.py` migrated
- [ ] All handlers in `subsystem_handlers.py` migrated
- [ ] All handlers in `graphs_handlers.py` migrated
- [ ] All handlers in `docs_handlers.py` migrated
- [ ] All handlers in `datasets_handlers.py` migrated
- [ ] All handlers in `history_handlers.py` migrated
- [ ] All handlers in `ide_handlers.py` migrated
- [ ] All tests updated and passing

---

## Phase 4: Dry-Run Mode

### Goal
Add `--dry-run` flag that shows what operations would execute without actually running them.

### Implementation

#### 4.1 Create Dry-Run Result Types

Add to `result_types.py`:

```python
@dataclass(frozen=True)
class DryRunStep:
    """A single step in a dry-run plan."""

    operation_id: str
    description: str
    params: dict[str, Any]
    is_prereq: bool = False


@dataclass(frozen=True)
class DryRunResult:
    """Result from dry-run execution."""

    target_operation: str
    steps: list[DryRunStep]
    estimated_duration: str | None = None
    warnings: list[str] = field(default_factory=list)
```

#### 4.2 Add Dry-Run Planning Function

Create `src/codeintel/cli/dry_run.py`:

```python
"""Dry-run execution planning for CLI operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.cli.result_types import DryRunResult, DryRunStep
from codeintel.serving.auto_pipeline import plan_operation_prereqs
from codeintel.serving.operations.catalog import get_operation


def plan_dry_run(
    op_id: str,
    params: dict[str, Any],
    *,
    skip_prereqs: bool = False,
) -> DryRunResult:
    """Plan a dry-run execution of an operation.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Operation parameters.
    skip_prereqs
        Whether prerequisites would be skipped.

    Returns
    -------
    DryRunResult
        Execution plan without actual execution.
    """
    operation = get_operation(op_id)
    if operation is None:
        return DryRunResult(
            target_operation=op_id,
            steps=[],
            warnings=[f"Unknown operation: {op_id}"],
        )

    steps: list[DryRunStep] = []
    warnings: list[str] = []

    # Plan prerequisites
    if not skip_prereqs:
        try:
            prereqs = plan_operation_prereqs(op_id, params)
            for prereq in prereqs:
                steps.append(DryRunStep(
                    operation_id=prereq.id,
                    description=prereq.summary or prereq.id,
                    params=prereq.computed_params or {},
                    is_prereq=True,
                ))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not plan prerequisites: {exc}")

    # Add target operation
    steps.append(DryRunStep(
        operation_id=op_id,
        description=operation.summary or op_id,
        params=params,
        is_prereq=False,
    ))

    # Estimate duration (if we have historical data)
    estimated = _estimate_duration(steps)

    return DryRunResult(
        target_operation=op_id,
        steps=steps,
        estimated_duration=estimated,
        warnings=warnings,
    )


def _estimate_duration(steps: list[DryRunStep]) -> str | None:
    """Estimate total duration based on historical data.

    Returns
    -------
    str | None
        Human-readable duration estimate or None if unknown.
    """
    # TODO: Integrate with run tracking for historical duration data
    return None
```

#### 4.3 Add --dry-run to Dynamic Operations

Update `_make_operation_params_dataclass()`:

```python
# Add dry_run field
fields.append((
    "dry_run",
    Annotated[bool, Parameter(
        name="--dry-run",
        help="Show execution plan without running.",
        negative=(),
    )],
    False,
))
```

Update `dynamic_op` closure:

```python
def dynamic_op(cfg: OperationCliArgs | None = None) -> None:
    if cfg is None:
        message = "Operation parameters are required."
        raise ValidationError(message)

    typed_cfg = cfg

    # Handle dry-run mode
    if getattr(typed_cfg, "dry_run", False):
        params = _build_params_dict(typed_cfg, metadata.params)
        plan = plan_dry_run(
            metadata.operation.id,
            params,
            skip_prereqs=typed_cfg.skip_prereqs,
        )
        output_format = get_output_format(getattr(typed_cfg, "output", None))
        _render_dry_run(plan, output_format)
        return

    # Handle stdin input
    if getattr(typed_cfg, "from_stdin", False):
        # ... existing stdin handling ...

    # Normal execution
    # ... existing execution ...
```

#### 4.4 Add Dry-Run Renderer

```python
def _render_dry_run(plan: DryRunResult, output_format: OutputFormat) -> None:
    """Render dry-run plan to stdout.

    Parameters
    ----------
    plan
        Execution plan.
    output_format
        Output format.
    """
    if output_format == OutputFormat.JSON:
        from dataclasses import asdict
        print(json.dumps(asdict(plan), indent=2))
        return

    print(f"Dry-run plan for: {plan.target_operation}")
    print("-" * 50)

    for i, step in enumerate(plan.steps, 1):
        prefix = "[prereq]" if step.is_prereq else "[target]"
        print(f"{i}. {prefix} {step.operation_id}")
        print(f"   {step.description}")
        if step.params:
            print(f"   Params: {step.params}")

    if plan.estimated_duration:
        print(f"\nEstimated duration: {plan.estimated_duration}")

    if plan.warnings:
        print("\nWarnings:")
        for warning in plan.warnings:
            print(f"  - {warning}")
```

### Deliverables
- [ ] `DryRunStep` and `DryRunResult` types created
- [ ] `dry_run.py` module created with `plan_dry_run()`
- [ ] `--dry-run` flag added to dynamic operations
- [ ] Dry-run rendering implemented
- [ ] Tests added

---

## Phase 5: Operation Middleware for Observability

### Goal
Implement a middleware pattern for consistent logging, metrics, and tracing across all operations.

### Implementation

#### 5.1 Define Middleware Protocol

Create `src/codeintel/cli/cli_middleware.py`:

```python
"""Middleware pattern for CLI operation execution.

Middleware components intercept operation execution to provide
cross-cutting concerns like logging, metrics, and tracing.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

LOG = logging.getLogger(__name__)


class OperationMiddleware(ABC):
    """Base class for operation execution middleware."""

    @abstractmethod
    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute before operation invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context data to pass to after_invoke.
        """
        ...

    @abstractmethod
    def after_invoke(
        self,
        op_id: str,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Execute after successful operation invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        ...

    @abstractmethod
    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Execute on operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        ...


class LoggingMiddleware(OperationMiddleware):
    """Log operation execution details."""

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Log operation start."""
        LOG.info("Starting operation", extra={"op_id": op_id, "params": params})
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Log operation completion."""
        duration = time.monotonic() - context["start_time"]
        LOG.info(
            "Operation completed",
            extra={"op_id": op_id, "duration_seconds": duration},
        )

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log operation error."""
        duration = time.monotonic() - context.get("start_time", 0)
        LOG.error(
            "Operation failed",
            extra={
                "op_id": op_id,
                "duration_seconds": duration,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )


class MetricsMiddleware(OperationMiddleware):
    """Collect operation metrics."""

    def __init__(self) -> None:
        self._operation_count: dict[str, int] = {}
        self._operation_errors: dict[str, int] = {}
        self._operation_durations: dict[str, list[float]] = {}

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Record operation start."""
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Record operation success."""
        duration = time.monotonic() - context["start_time"]
        self._operation_count[op_id] = self._operation_count.get(op_id, 0) + 1
        if op_id not in self._operation_durations:
            self._operation_durations[op_id] = []
        self._operation_durations[op_id].append(duration)

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record operation error."""
        self._operation_errors[op_id] = self._operation_errors.get(op_id, 0) + 1

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns
        -------
        dict[str, Any]
            Metrics summary.
        """
        return {
            "operation_counts": self._operation_count,
            "operation_errors": self._operation_errors,
            "operation_durations": {
                op_id: {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations) if durations else 0,
                }
                for op_id, durations in self._operation_durations.items()
            },
        }


@dataclass
class MiddlewareStack:
    """Stack of middleware to execute around operations."""

    middleware: list[OperationMiddleware] = field(default_factory=list)

    def add(self, mw: OperationMiddleware) -> None:
        """Add middleware to the stack."""
        self.middleware.append(mw)

    @contextmanager
    def wrap(self, op_id: str, params: dict[str, Any]) -> Iterator[None]:
        """Wrap operation execution with middleware.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Yields
        ------
        None
            Control to the wrapped operation.
        """
        contexts: list[dict[str, Any]] = []

        # Before hooks
        for mw in self.middleware:
            ctx = mw.before_invoke(op_id, params)
            contexts.append(ctx)

        try:
            yield
        except Exception as exc:
            # Error hooks (reverse order)
            for mw, ctx in zip(reversed(self.middleware), reversed(contexts)):
                try:
                    mw.on_error(op_id, exc, ctx)
                except Exception:  # noqa: BLE001, S110
                    pass  # Don't let middleware errors mask original error
            raise
        else:
            # After hooks (reverse order)
            for mw, ctx in zip(reversed(self.middleware), reversed(contexts)):
                mw.after_invoke(op_id, None, ctx)


# Global middleware stack
_MIDDLEWARE_STACK = MiddlewareStack()


def get_middleware_stack() -> MiddlewareStack:
    """Get the global middleware stack.

    Returns
    -------
    MiddlewareStack
        Global middleware stack instance.
    """
    return _MIDDLEWARE_STACK


def configure_default_middleware() -> None:
    """Configure default middleware (logging)."""
    stack = get_middleware_stack()
    stack.add(LoggingMiddleware())


__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareStack",
    "OperationMiddleware",
    "configure_default_middleware",
    "get_middleware_stack",
]
```

#### 5.2 Integrate Middleware into Operation Execution

Update `_invoke_operation_with_prereqs()` in `cyclopts_ops.py`:

```python
from codeintel.cli.cli_middleware import get_middleware_stack


def _invoke_operation_with_prereqs(
    op_id: str,
    params: dict[str, Any],
    runtime: ProjectRuntime,
    *,
    skip_prereqs: bool = False,
    verbose: bool = False,
) -> None:
    """Invoke operation with middleware and prerequisites."""
    middleware = get_middleware_stack()

    with middleware.wrap(op_id, params):
        if not skip_prereqs:
            run_operation_prereqs(op_id, params, runtime.gateway)

        invoke_operation(op_id, params, runtime)
```

#### 5.3 Configure Middleware at CLI Startup

Update `cyclopts_app.py`:

```python
from codeintel.cli.cli_middleware import configure_default_middleware

# Configure middleware before app creation
configure_default_middleware()

app: App = build_patched_app(make_root_app)
```

### Deliverables
- [ ] `cli_middleware.py` created
- [ ] `OperationMiddleware` protocol defined
- [ ] `LoggingMiddleware` implemented
- [ ] `MetricsMiddleware` implemented
- [ ] `MiddlewareStack` implemented
- [ ] Operation execution integrated with middleware
- [ ] Default middleware configured at startup
- [ ] Tests added

---

## Phase 6: Progress Reporting

### Goal
Add progress bars and status indicators for long-running operations.

### Implementation

#### 6.1 Create Progress Module

Create `src/codeintel/cli/cli_progress.py`:

```python
"""Progress reporting for long-running CLI operations.

Uses Rich progress bars for visual feedback during batch operations
and long-running builds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")


def create_progress_bar(
    *,
    console: Console | None = None,
    transient: bool = False,
) -> Progress:
    """Create a standard progress bar.

    Parameters
    ----------
    console
        Rich console instance.
    transient
        Whether to remove progress bar after completion.

    Returns
    -------
    Progress
        Configured progress bar.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )


@contextmanager
def progress_context(
    description: str = "Processing...",
    total: int | None = None,
    *,
    console: Console | None = None,
) -> Iterator[Progress]:
    """Context manager for progress bar.

    Parameters
    ----------
    description
        Initial task description.
    total
        Total number of items (None for indeterminate).
    console
        Rich console instance.

    Yields
    ------
    Progress
        Progress bar instance.
    """
    progress = create_progress_bar(console=console)
    with progress:
        progress.add_task(description, total=total)
        yield progress


def iterate_with_progress(
    items: list[T],
    description: str = "Processing...",
    *,
    console: Console | None = None,
) -> Iterator[T]:
    """Iterate over items with progress bar.

    Parameters
    ----------
    items
        Items to iterate.
    description
        Task description.
    console
        Rich console instance.

    Yields
    ------
    T
        Each item from the input list.
    """
    with create_progress_bar(console=console, transient=True) as progress:
        task = progress.add_task(description, total=len(items))
        for item in items:
            yield item
            progress.advance(task)


class OperationProgressTracker:
    """Track progress of multi-step operations."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._progress: Progress | None = None
        self._tasks: dict[str, TaskID] = {}

    def start(self, steps: list[str]) -> None:
        """Start progress tracking with known steps.

        Parameters
        ----------
        steps
            List of step names.
        """
        self._progress = create_progress_bar(console=self._console)
        self._progress.start()

        for step in steps:
            task_id = self._progress.add_task(step, total=1, visible=False)
            self._tasks[step] = task_id

    def begin_step(self, step: str) -> None:
        """Mark a step as in progress.

        Parameters
        ----------
        step
            Step name.
        """
        if self._progress and step in self._tasks:
            self._progress.update(self._tasks[step], visible=True)

    def complete_step(self, step: str) -> None:
        """Mark a step as complete.

        Parameters
        ----------
        step
            Step name.
        """
        if self._progress and step in self._tasks:
            self._progress.update(self._tasks[step], completed=1)

    def fail_step(self, step: str, error: str) -> None:
        """Mark a step as failed.

        Parameters
        ----------
        step
            Step name.
        error
            Error message.
        """
        if self._progress and step in self._tasks:
            self._progress.update(
                self._tasks[step],
                description=f"[red]{step}: {error}[/red]",
            )

    def finish(self) -> None:
        """Stop progress tracking."""
        if self._progress:
            self._progress.stop()
            self._progress = None


__all__ = [
    "OperationProgressTracker",
    "create_progress_bar",
    "iterate_with_progress",
    "progress_context",
]
```

#### 6.2 Integrate Progress into Build Commands

Update `build_handlers.py`:

```python
from codeintel.cli.cli_progress import iterate_with_progress, OperationProgressTracker


def build_run_handler(
    options: BuildRunOptions,
    ctx_opts: BuildRunContext,
) -> CliResult[BuildRunResult]:
    """Run build targets with progress reporting.

    Returns
    -------
    CliResult[BuildRunResult]
        Build execution results.
    """
    targets = resolve_build_targets(options)

    executed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    start_time = time.monotonic()

    # Use progress bar for batch execution
    for target in iterate_with_progress(targets, "Building targets..."):
        try:
            if target_is_fresh(target):
                skipped.append(target)
            else:
                execute_target(target)
                executed.append(target)
        except BuildError as exc:
            failed.append(target)

    duration = time.monotonic() - start_time

    return CliResult.ok(BuildRunResult(
        executed=executed,
        skipped=skipped,
        failed=failed,
        duration_seconds=duration,
    ))
```

#### 6.3 Add Progress to Stdin Processing

Update `_execute_from_stdin()`:

```python
def _execute_from_stdin(
    metadata: OperationCliMetadata,
    cfg: OperationCliArgs,
    runtime: ProjectRuntime,
    verbose: bool,
) -> None:
    """Execute operation for each record from stdin with progress."""
    cli_args = _build_params_dict(cfg, metadata.params)

    # Read all records first to get count
    records = list(iter_stdin_records())
    results: list[dict[str, object]] = []

    # Process with progress bar
    for stdin_record in iterate_with_progress(
        records,
        f"Processing {metadata.operation.id}...",
    ):
        merged_params = merge_stdin_with_args(stdin_record, cli_args)
        try:
            result = _invoke_operation_for_result(
                metadata.operation.id,
                merged_params,
                runtime,
                skip_prereqs=getattr(cfg, "skip_prereqs", False),
                verbose=verbose,
            )
            results.append({"input": stdin_record, "result": result, "success": True})
        except Exception as exc:  # noqa: BLE001
            results.append({
                "input": stdin_record,
                "error": str(exc),
                "success": False,
            })

    # Output results
    output_format = get_output_format(getattr(cfg, "output", None))
    envelope = OutputEnvelope(
        data=results,
        metadata={"operation": metadata.operation.id, "count": len(results)},
    )
    envelope.write(output_format, sys.stdout)
```

### Deliverables
- [ ] `cli_progress.py` created
- [ ] `create_progress_bar()` implemented
- [ ] `iterate_with_progress()` implemented
- [ ] `OperationProgressTracker` implemented
- [ ] Build commands integrated with progress
- [ ] Stdin processing integrated with progress
- [ ] Tests added

---

## Phase 7: Input Validation Layer

### Goal
Create a centralized validation layer that runs before handler execution.

### Implementation

#### 7.1 Define Validation Protocol

Create `src/codeintel/cli/cli_validation.py`:

```python
"""Input validation layer for CLI commands.

Provides a framework for validating inputs before handler execution,
with built-in validators for common patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.cli_errors import ValidationError


class Validator(ABC):
    """Base class for input validators."""

    @abstractmethod
    def validate(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters.

        Parameters
        ----------
        params
            Parameters to validate.

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid).
        """
        ...


class RequiredFieldValidator(Validator):
    """Validate that required fields are present."""

    def __init__(self, *fields: str) -> None:
        self.fields = fields

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Check for required fields."""
        errors: list[str] = []
        for field_name in self.fields:
            if field_name not in params or params[field_name] is None:
                errors.append(f"Required field missing: {field_name}")
        return errors


class PathExistsValidator(Validator):
    """Validate that path fields point to existing files/directories."""

    def __init__(self, *fields: str, must_be_file: bool = False, must_be_dir: bool = False) -> None:
        self.fields = fields
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Check path existence."""
        errors: list[str] = []
        for field_name in self.fields:
            value = params.get(field_name)
            if value is None:
                continue

            path = Path(value)
            if not path.exists():
                errors.append(f"Path does not exist: {field_name}={value}")
            elif self.must_be_file and not path.is_file():
                errors.append(f"Path is not a file: {field_name}={value}")
            elif self.must_be_dir and not path.is_dir():
                errors.append(f"Path is not a directory: {field_name}={value}")
        return errors


class ChoiceValidator(Validator):
    """Validate that field values are in allowed set."""

    def __init__(self, field: str, choices: set[str]) -> None:
        self.field = field
        self.choices = choices

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Check value is in choices."""
        value = params.get(self.field)
        if value is None:
            return []
        if value not in self.choices:
            return [f"Invalid value for {self.field}: {value}. Must be one of: {sorted(self.choices)}"]
        return []


class RangeValidator(Validator):
    """Validate numeric fields are within range."""

    def __init__(
        self,
        field: str,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        self.field = field
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Check value is in range."""
        value = params.get(self.field)
        if value is None:
            return []

        try:
            num_value = float(value)
        except (TypeError, ValueError):
            return [f"Invalid numeric value for {self.field}: {value}"]

        if self.min_value is not None and num_value < self.min_value:
            return [f"{self.field} must be >= {self.min_value}, got {value}"]
        if self.max_value is not None and num_value > self.max_value:
            return [f"{self.field} must be <= {self.max_value}, got {value}"]
        return []


class GoidFormatValidator(Validator):
    """Validate GOID format."""

    def __init__(self, field: str = "goid") -> None:
        self.field = field

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Check GOID format."""
        value = params.get(self.field)
        if value is None:
            return []

        # GOIDs should match pattern: module.path:function_name
        if not isinstance(value, str):
            return [f"Invalid GOID type for {self.field}: expected string"]
        if ":" not in value and "." not in value:
            return [f"Invalid GOID format for {self.field}: {value}"]
        return []


@dataclass
class ValidationChain:
    """Chain of validators to run in sequence."""

    validators: list[Validator] = field(default_factory=list)

    def add(self, validator: Validator) -> "ValidationChain":
        """Add a validator to the chain.

        Returns
        -------
        ValidationChain
            Self for chaining.
        """
        self.validators.append(validator)
        return self

    def validate(self, params: dict[str, Any]) -> list[str]:
        """Run all validators.

        Parameters
        ----------
        params
            Parameters to validate.

        Returns
        -------
        list[str]
            All validation errors.
        """
        errors: list[str] = []
        for validator in self.validators:
            errors.extend(validator.validate(params))
        return errors

    def validate_or_raise(self, params: dict[str, Any]) -> None:
        """Run all validators and raise if any fail.

        Parameters
        ----------
        params
            Parameters to validate.

        Raises
        ------
        ValidationError
            If any validation fails.
        """
        errors = self.validate(params)
        if errors:
            message = "Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValidationError(message)


# Pre-built validation chains for common operations
GOID_OPERATION_VALIDATORS = ValidationChain([
    RequiredFieldValidator("goid"),
    GoidFormatValidator("goid"),
])

FILE_OPERATION_VALIDATORS = ValidationChain([
    RequiredFieldValidator("file_path"),
    PathExistsValidator("file_path", must_be_file=True),
])

RUNTIME_VALIDATORS = ValidationChain([
    PathExistsValidator("db_path", must_be_file=True),
    PathExistsValidator("repo_root", must_be_dir=True),
])


__all__ = [
    "ChoiceValidator",
    "FILE_OPERATION_VALIDATORS",
    "GOID_OPERATION_VALIDATORS",
    "GoidFormatValidator",
    "PathExistsValidator",
    "RUNTIME_VALIDATORS",
    "RangeValidator",
    "RequiredFieldValidator",
    "ValidationChain",
    "Validator",
]
```

#### 7.2 Register Validators Per Operation

Add validation registry:

```python
# In cli_validation.py

from typing import Callable

_OPERATION_VALIDATORS: dict[str, ValidationChain] = {}


def register_validators(op_id: str, chain: ValidationChain) -> None:
    """Register validators for an operation.

    Parameters
    ----------
    op_id
        Operation identifier.
    chain
        Validation chain.
    """
    _OPERATION_VALIDATORS[op_id] = chain


def get_validators(op_id: str) -> ValidationChain | None:
    """Get validators for an operation.

    Parameters
    ----------
    op_id
        Operation identifier.

    Returns
    -------
    ValidationChain | None
        Registered validators or None.
    """
    return _OPERATION_VALIDATORS.get(op_id)


def validate_operation_params(op_id: str, params: dict[str, Any]) -> None:
    """Validate parameters for an operation.

    Parameters
    ----------
    op_id
        Operation identifier.
    params
        Parameters to validate.

    Raises
    ------
    ValidationError
        If validation fails.
    """
    chain = get_validators(op_id)
    if chain:
        chain.validate_or_raise(params)
```

#### 7.3 Integrate Validation into Operation Execution

Update `_invoke_operation_with_prereqs()`:

```python
from codeintel.cli.cli_validation import validate_operation_params


def _invoke_operation_with_prereqs(
    op_id: str,
    params: dict[str, Any],
    runtime: ProjectRuntime,
    *,
    skip_prereqs: bool = False,
    verbose: bool = False,
) -> None:
    """Invoke operation with validation, middleware, and prerequisites."""
    middleware = get_middleware_stack()

    # Validate inputs first
    validate_operation_params(op_id, params)

    with middleware.wrap(op_id, params):
        if not skip_prereqs:
            run_operation_prereqs(op_id, params, runtime.gateway)

        invoke_operation(op_id, params, runtime)
```

#### 7.4 Register Default Validators

Create `src/codeintel/cli/validators_registry.py`:

```python
"""Register validators for built-in operations."""

from __future__ import annotations

from codeintel.cli.cli_validation import (
    GOID_OPERATION_VALIDATORS,
    FILE_OPERATION_VALIDATORS,
    RangeValidator,
    RequiredFieldValidator,
    ValidationChain,
    register_validators,
)


def configure_default_validators() -> None:
    """Register validators for all built-in operations."""

    # Function operations
    register_validators("function.summary", GOID_OPERATION_VALIDATORS)
    register_validators("profiles.function", GOID_OPERATION_VALIDATORS)
    register_validators("architecture.function", GOID_OPERATION_VALIDATORS)
    register_validators("functions.tests", GOID_OPERATION_VALIDATORS)

    # File operations
    register_validators("file.summary", FILE_OPERATION_VALIDATORS)
    register_validators("profiles.file", FILE_OPERATION_VALIDATORS)

    # Module operations
    register_validators("profiles.module", ValidationChain([
        RequiredFieldValidator("module"),
    ]))

    # Graph operations with depth limits
    register_validators("graph.call_neighborhood", ValidationChain([
        RequiredFieldValidator("goid"),
        RangeValidator("depth", min_value=1, max_value=10),
    ]))

    # Subsystem operations
    register_validators("subsystems.detail", ValidationChain([
        RequiredFieldValidator("subsystem_id"),
    ]))


__all__ = ["configure_default_validators"]
```

### Deliverables
- [ ] `cli_validation.py` created
- [ ] Base `Validator` protocol defined
- [ ] Built-in validators implemented:
  - [ ] `RequiredFieldValidator`
  - [ ] `PathExistsValidator`
  - [ ] `ChoiceValidator`
  - [ ] `RangeValidator`
  - [ ] `GoidFormatValidator`
- [ ] `ValidationChain` implemented
- [ ] Operation execution integrated with validation
- [ ] Default validators registered
- [ ] Tests added

---

## Testing Strategy

### Unit Tests
Each new module should have corresponding tests:
- `tests/cli/test_cli_types.py`
- `tests/cli/test_stdin_composition.py`
- `tests/cli/test_dry_run.py`
- `tests/cli/test_cli_middleware.py`
- `tests/cli/test_cli_progress.py`
- `tests/cli/test_cli_validation.py`

### Integration Tests
End-to-end tests for command composition:
```python
def test_pipe_operations() -> None:
    """Test piping operations together."""
    # Run high-risk functions
    result1 = subprocess.run(
        ["codeintel", "op", "functions-high-risk", "--json", "--limit", "5"],
        capture_output=True,
        text=True,
    )

    # Pipe to function-summary
    result2 = subprocess.run(
        ["codeintel", "op", "function-summary", "--from-stdin"],
        input=result1.stdout,
        capture_output=True,
        text=True,
    )

    assert result2.returncode == 0
```

### Golden File Tests
For output formatting consistency:
```python
def test_dry_run_output_format(golden) -> None:
    """Verify dry-run output matches expected format."""
    result = run_cli(["op", "function-summary", "--dry-run", "--goid", "test.func"])
    golden.assert_match(result.stdout, "dry_run_function_summary.txt")
```

---

## Implementation Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| 1. Unify Types | 2-3 hours | None |
| 2. Wire stdin | 3-4 hours | Phase 1 |
| 3. Migrate Handlers | 8-12 hours | Phases 1, 2 |
| 4. Dry-Run Mode | 3-4 hours | Phase 3 |
| 5. Middleware | 4-5 hours | Phase 3 |
| 6. Progress | 3-4 hours | Phases 3, 5 |
| 7. Validation | 4-5 hours | Phase 3 |

**Total Estimated Effort: 27-37 hours**

---

## Success Criteria

1. **All quality checks pass**: ruff, pyright, pyrefly, pytest
2. **Coverage maintained**: No regression in test coverage
3. **Documentation updated**: All new features documented
4. **Backward compatible**: Existing scripts continue to work
5. **Performance acceptable**: No noticeable latency increase for simple commands

