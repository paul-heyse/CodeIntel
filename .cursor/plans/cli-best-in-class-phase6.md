# CLI Best-in-Class Implementation Plan (Phase 6)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 5 (Completed)

---

## Executive Summary

Phase 6 represents the **hardening and evolution** of the CLI infrastructure. While Phase 5 activated the foundational components, Phase 6 focuses on completing the migration, enforcing patterns, adding resilience, and introducing advanced features that elevate the CLI to true best-in-class status.

The eight priorities address:

1. **Complete Executor Pipeline Migration** — Wire remaining commands through OperationExecutor
2. **Structured Error Taxonomy Enforcement** — RFC 9457 Problem Details everywhere
3. **CLI-Specific Testing Infrastructure** — Charter-compliant test harnesses
4. **Resilience Pattern Integration** — Retry and circuit breaker in execution pipeline
5. **Configuration Schema Validation** — JSON Schema validation for config files
6. **Observability Deep Integration** — Automatic tracing for all operations
7. **Command Composition & Pipelines** — Chaining and streaming output
8. **Interactive Shell Mode** — REPL for exploratory use

### Why Phase 6 Matters

Phase 5 created the activation layer. Phase 6 ensures:

| Aspect | After Phase 5 | After Phase 6 |
|--------|---------------|---------------|
| Executor Usage | Some commands | All commands |
| Error Handling | Mixed patterns | Uniform RFC 9457 |
| Testing | Ad-hoc | Structured harnesses |
| Resilience | Defined | Integrated |
| Config Validation | Runtime | Schema-enforced |
| Observability | Opt-in | Automatic |
| Composition | None | Full pipeline support |
| Interactivity | None | REPL shell |

---

## Table of Contents

1. [Phase 6.1: Complete Executor Pipeline Migration](#phase-61-complete-executor-pipeline-migration)
2. [Phase 6.2: Structured Error Taxonomy Enforcement](#phase-62-structured-error-taxonomy-enforcement)
3. [Phase 6.3: CLI-Specific Testing Infrastructure](#phase-63-cli-specific-testing-infrastructure)
4. [Phase 6.4: Resilience Pattern Integration](#phase-64-resilience-pattern-integration)
5. [Phase 6.5: Configuration Schema Validation](#phase-65-configuration-schema-validation)
6. [Phase 6.6: Observability Deep Integration](#phase-66-observability-deep-integration)
7. [Phase 6.7: Command Composition & Pipelines](#phase-67-command-composition--pipelines)
8. [Phase 6.8: Interactive Shell Mode](#phase-68-interactive-shell-mode)
9. [Implementation Timeline](#implementation-timeline)
10. [Success Metrics](#success-metrics)

---

## Phase 6.1: Complete Executor Pipeline Migration

### Value Proposition

Phase 5 established the pattern; Phase 6 ensures 100% adoption. Every command flows through `OperationExecutor`, guaranteeing:

- Automatic validation, middleware, progress, and rendering
- Consistent telemetry across all operations
- Uniform error handling and output

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        All CLI Commands                                  │
│   build | op | dataset | docs | graph | storage | history | ide | ...   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OperationExecutor                                   │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│   │ Validate │→│ Trace    │→│ Retry    │→│ Execute  │→│ Render   │     │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Audit all cyclopts command files for direct handler calls
2. Create OperationSpec for every remaining handler
3. Update commands to use `get_executor().execute()`
4. Remove legacy rendering/error handling code
5. Add `--dry-run` support via executor flag

### Implementation

#### File: `src/codeintel/cli/operations/docs_operations.py`

```python
"""Documentation operation specifications."""

from __future__ import annotations

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import register_operation
from codeintel.cli.result_types import DocsGenerateResult, DocsStatusResult
from codeintel.cli.results import CliResult


def _docs_status_handler() -> CliResult[DocsStatusResult]:
    """Check documentation status."""
    # Placeholder - actual impl calls docs infrastructure
    return CliResult.ok(DocsStatusResult(
        generated_count=0,
        pending_count=0,
        stale_count=0,
        last_generated=None,
    ))


def _docs_generate_handler(
    *,
    targets: list[str] | None = None,
    force: bool = False,
) -> CliResult[DocsGenerateResult]:
    """Generate documentation."""
    return CliResult.ok(DocsGenerateResult(
        generated=[],
        skipped=[],
        errors=[],
    ))


DOCS_STATUS_SPEC = register_operation(
    OperationSpec(
        operation_id="docs.status",
        handler=_docs_status_handler,
        category=OperationCategory.READ,
        description="Check documentation generation status",
    )
)

DOCS_GENERATE_SPEC = register_operation(
    OperationSpec(
        operation_id="docs.generate",
        handler=_docs_generate_handler,
        category=OperationCategory.BUILD,
        requires_progress=True,
        estimated_duration=60.0,
        description="Generate documentation",
    )
)
```

#### File: `src/codeintel/cli/operations/graph_operations.py`

```python
"""Graph operation specifications."""

from __future__ import annotations

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import register_operation
from codeintel.cli.result_types import GraphStatsResult
from codeintel.cli.results import CliResult


def _graph_stats_handler(
    *,
    graph_type: str = "call",
) -> CliResult[GraphStatsResult]:
    """Get graph statistics."""
    return CliResult.ok(GraphStatsResult(
        node_count=0,
        edge_count=0,
        density=0.0,
        components=0,
    ))


GRAPH_STATS_SPEC = register_operation(
    OperationSpec(
        operation_id="graph.stats",
        handler=_graph_stats_handler,
        category=OperationCategory.READ,
        description="Show graph statistics",
    )
)
```

### Migration Checklist

For each command group:

- [ ] `cyclopts_docs.py` — Create `docs_operations.py`, update commands
- [ ] `cyclopts_graphs.py` — Create `graph_operations.py`, update commands
- [ ] `cyclopts_storage.py` — Create `storage_operations.py`, update commands
- [ ] `cyclopts_history.py` — Create `history_operations.py`, update commands
- [ ] `cyclopts_ide.py` — Create `ide_operations.py`, update commands
- [ ] `cyclopts_subsystem.py` — Create `subsystem_operations.py`, update commands
- [ ] Verify 100% executor coverage with audit script

#### Migration Audit Script

```python
"""Audit CLI commands for executor usage."""

from pathlib import Path
import ast

def audit_cyclopts_files() -> dict[str, list[str]]:
    """Find commands not using OperationExecutor."""
    cli_path = Path("src/codeintel/cli")
    violations = {}
    
    for path in cli_path.glob("cyclopts_*.py"):
        with open(path) as f:
            tree = ast.parse(f.read())
        
        file_violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function uses get_executor()
                source = ast.unparse(node)
                if "@" in source and "command" in source:
                    if "get_executor" not in source:
                        file_violations.append(node.name)
        
        if file_violations:
            violations[path.name] = file_violations
    
    return violations
```

---

## Phase 6.2: Structured Error Taxonomy Enforcement

### Value Proposition

Currently, errors are raised inconsistently:
- Some handlers raise raw exceptions
- Some return `CliResult.error()` with ad-hoc messages
- Error codes aren't standardized

RFC 9457 Problem Details everywhere enables:
- Machine-parseable error responses
- Consistent error codes for automation
- Actionable messages with suggestions
- Debug mode for stack traces

### Error Taxonomy

```
urn:codeintel:cli:
├── validation/
│   ├── missing-required        # Required parameter missing (400)
│   ├── invalid-type            # Wrong parameter type (400)
│   ├── invalid-format          # Value doesn't match pattern (400)
│   ├── out-of-range            # Value outside allowed range (400)
│   └── constraint-violation    # Business rule violation (400)
├── operation/
│   ├── not-found               # Operation/resource not found (404)
│   ├── already-exists          # Resource already exists (409)
│   ├── timeout                 # Operation timed out (504)
│   ├── dependency-failed       # Prerequisite failed (424)
│   ├── cancelled               # Operation was cancelled (499)
│   └── internal-error          # Unexpected error (500)
├── storage/
│   ├── connection-failed       # Cannot connect to database (503)
│   ├── query-failed            # Query execution failed (500)
│   ├── schema-mismatch         # Schema version mismatch (500)
│   └── corruption-detected     # Data integrity error (500)
├── config/
│   ├── file-not-found          # Config file missing (404)
│   ├── parse-error             # Config file malformed (400)
│   ├── invalid-value           # Config value invalid (400)
│   └── schema-violation        # Config doesn't match schema (400)
├── service/
│   ├── unavailable             # Service not responding (503)
│   ├── rate-limited            # Too many requests (429)
│   ├── authentication-failed   # Auth error (401)
│   └── permission-denied       # Authorization error (403)
└── job/
    ├── not-found               # Job doesn't exist (404)
    ├── already-running         # Job already in progress (409)
    ├── failed                  # Job execution failed (500)
    └── expired                 # Job results expired (410)
```

### Implementation

#### File: `src/codeintel/cli/cli_errors_v2.py`

```python
"""Enhanced error types with RFC 9457 Problem Details.

This module provides a comprehensive error taxonomy with factory
functions for creating consistent, machine-parseable errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from codeintel.cli.cli_errors import ProblemDetail


class ErrorCategory(Enum):
    """Top-level error categories."""

    VALIDATION = "validation"
    OPERATION = "operation"
    STORAGE = "storage"
    CONFIG = "config"
    SERVICE = "service"
    JOB = "job"


@dataclass(frozen=True)
class ErrorCode:
    """Structured error code with metadata.

    Parameters
    ----------
    category
        Error category.
    code
        Specific error code.
    status
        HTTP status code.
    title
        Human-readable title.
    """

    category: ErrorCategory
    code: str
    status: int
    title: str

    @property
    def type_uri(self) -> str:
        """Get fully-qualified error type URI."""
        return f"urn:codeintel:cli:{self.category.value}/{self.code}"


# Validation Errors
MISSING_REQUIRED = ErrorCode(ErrorCategory.VALIDATION, "missing-required", 400, "Missing Required Parameter")
INVALID_TYPE = ErrorCode(ErrorCategory.VALIDATION, "invalid-type", 400, "Invalid Parameter Type")
INVALID_FORMAT = ErrorCode(ErrorCategory.VALIDATION, "invalid-format", 400, "Invalid Parameter Format")
OUT_OF_RANGE = ErrorCode(ErrorCategory.VALIDATION, "out-of-range", 400, "Value Out of Range")
CONSTRAINT_VIOLATION = ErrorCode(ErrorCategory.VALIDATION, "constraint-violation", 400, "Constraint Violation")

# Operation Errors
NOT_FOUND = ErrorCode(ErrorCategory.OPERATION, "not-found", 404, "Resource Not Found")
ALREADY_EXISTS = ErrorCode(ErrorCategory.OPERATION, "already-exists", 409, "Resource Already Exists")
TIMEOUT = ErrorCode(ErrorCategory.OPERATION, "timeout", 504, "Operation Timeout")
DEPENDENCY_FAILED = ErrorCode(ErrorCategory.OPERATION, "dependency-failed", 424, "Dependency Failed")
CANCELLED = ErrorCode(ErrorCategory.OPERATION, "cancelled", 499, "Operation Cancelled")
INTERNAL_ERROR = ErrorCode(ErrorCategory.OPERATION, "internal-error", 500, "Internal Error")

# Storage Errors
CONNECTION_FAILED = ErrorCode(ErrorCategory.STORAGE, "connection-failed", 503, "Storage Connection Failed")
QUERY_FAILED = ErrorCode(ErrorCategory.STORAGE, "query-failed", 500, "Query Failed")
SCHEMA_MISMATCH = ErrorCode(ErrorCategory.STORAGE, "schema-mismatch", 500, "Schema Mismatch")

# Config Errors
CONFIG_NOT_FOUND = ErrorCode(ErrorCategory.CONFIG, "file-not-found", 404, "Configuration File Not Found")
CONFIG_PARSE_ERROR = ErrorCode(ErrorCategory.CONFIG, "parse-error", 400, "Configuration Parse Error")
CONFIG_INVALID = ErrorCode(ErrorCategory.CONFIG, "invalid-value", 400, "Invalid Configuration Value")
CONFIG_SCHEMA_VIOLATION = ErrorCode(ErrorCategory.CONFIG, "schema-violation", 400, "Configuration Schema Violation")

# Service Errors
SERVICE_UNAVAILABLE = ErrorCode(ErrorCategory.SERVICE, "unavailable", 503, "Service Unavailable")
RATE_LIMITED = ErrorCode(ErrorCategory.SERVICE, "rate-limited", 429, "Rate Limited")
AUTH_FAILED = ErrorCode(ErrorCategory.SERVICE, "authentication-failed", 401, "Authentication Failed")
PERMISSION_DENIED = ErrorCode(ErrorCategory.SERVICE, "permission-denied", 403, "Permission Denied")


@dataclass
class CliError(Exception):
    """Base CLI error with Problem Detail support.

    Parameters
    ----------
    code
        Error code with metadata.
    detail
        Human-readable detail message.
    extensions
        Additional context data.
    suggestion
        Suggested fix (optional).
    cause
        Underlying exception (optional).
    """

    code: ErrorCode
    detail: str
    extensions: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None
    cause: Exception | None = None

    def to_problem_detail(self, *, debug: bool = False) -> ProblemDetail:
        """Convert to RFC 9457 Problem Detail.

        Parameters
        ----------
        debug
            Include debug information.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        ext = dict(self.extensions)
        if self.suggestion:
            ext["suggestion"] = self.suggestion
        if debug and self.cause:
            import traceback
            ext["cause_type"] = type(self.cause).__name__
            ext["cause_message"] = str(self.cause)
            ext["traceback"] = traceback.format_exception(self.cause)

        return ProblemDetail(
            type=self.code.type_uri,
            title=self.code.title,
            detail=self.detail,
            status=self.code.status,
            extensions=ext if ext else None,
        )


class ValidationError(CliError):
    """Validation-specific error with field context."""

    field: str = ""
    value: Any = None

    def __init__(
        self,
        code: ErrorCode,
        field: str,
        message: str,
        *,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize validation error."""
        self.field = field
        self.value = value
        extensions = {"field": field}
        if value is not None:
            extensions["value"] = str(value)[:100]
        super().__init__(
            code=code,
            detail=f"{field}: {message}",
            extensions=extensions,
            suggestion=suggestion,
        )


class OperationError(CliError):
    """Operation-specific error with operation context."""

    operation_id: str = ""

    def __init__(
        self,
        code: ErrorCode,
        operation_id: str,
        message: str,
        *,
        cause: Exception | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize operation error."""
        self.operation_id = operation_id
        extensions = {"operation_id": operation_id}
        super().__init__(
            code=code,
            detail=message,
            extensions=extensions,
            suggestion=suggestion,
            cause=cause,
        )


# Factory functions for common errors
def missing_required_error(field: str, *, suggestion: str | None = None) -> ValidationError:
    """Create missing required parameter error."""
    return ValidationError(
        MISSING_REQUIRED,
        field,
        "This parameter is required",
        suggestion=suggestion or f"Provide a value for --{field}",
    )


def not_found_error(
    resource_type: str,
    resource_id: str,
    *,
    suggestion: str | None = None,
) -> CliError:
    """Create resource not found error."""
    return CliError(
        code=NOT_FOUND,
        detail=f"{resource_type} not found: {resource_id}",
        extensions={"resource_type": resource_type, "resource_id": resource_id},
        suggestion=suggestion,
    )


def timeout_error(
    operation_id: str,
    timeout_seconds: float,
    *,
    suggestion: str | None = None,
) -> OperationError:
    """Create operation timeout error."""
    return OperationError(
        TIMEOUT,
        operation_id,
        f"Operation timed out after {timeout_seconds}s",
        suggestion=suggestion or "Try increasing timeout or breaking into smaller operations",
    )
```

### Integration with Executor

Update `executor.py` to use structured errors:

```python
def _handle_exception(
    self,
    exc: Exception,
    spec: OperationSpec[T],
    ctx: ExecutionContext,
) -> CliResult[T]:
    """Convert exception to CliResult with Problem Detail."""
    if isinstance(exc, CliError):
        problem = exc.to_problem_detail(debug=self._debug_mode)
    else:
        # Wrap unknown exceptions
        problem = CliError(
            code=INTERNAL_ERROR,
            detail=str(exc) if self._debug_mode else "An unexpected error occurred",
            cause=exc,
        ).to_problem_detail(debug=self._debug_mode)
    
    return CliResult.error(problem)
```

---

## Phase 6.3: CLI-Specific Testing Infrastructure

### Value Proposition

The Testing Charter forbids monkeypatching. CLI tests need:
- Harnesses that invoke real entry points
- Structured assertions on output
- Golden file testing for output stability
- Integration with test doubles from Phase 3

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Test Code                                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CliTestHarness                                      │
│   • invoke(args) → CliInvocationResult                                  │
│   • invoke_json(args) → dict                                            │
│   • with_config(config) → CliTestHarness                                │
│   • with_env(vars) → CliTestHarness                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌─────────────────┐       ┌─────────────────┐
          │   Real CLI App   │       │   Test Doubles   │
          │  (cyclopts_app)  │       │  (from Phase 3)  │
          └─────────────────┘       └─────────────────┘
```

### Implementation

#### File: `tests/cli/_harness/__init__.py`

```python
"""CLI test harness for charter-compliant testing.

Provides tools for testing CLI commands through real entry points
without monkeypatching, using dependency injection and test doubles.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from codeintel.cli.cyclopts_app import app


@dataclass
class CliInvocationResult:
    """Result of a CLI invocation.

    Parameters
    ----------
    exit_code
        Process exit code.
    stdout
        Captured standard output.
    stderr
        Captured standard error.
    exception
        Exception if raised.
    """

    exit_code: int
    stdout: str
    stderr: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        """Check if invocation succeeded."""
        return self.exit_code == 0

    def json(self) -> dict[str, Any]:
        """Parse stdout as JSON."""
        return json.loads(self.stdout)

    def lines(self) -> list[str]:
        """Get stdout as lines."""
        return self.stdout.strip().split("\n")


@dataclass
class CliTestHarness:
    """Harness for testing CLI commands.

    Provides a clean way to invoke CLI commands and capture output
    without subprocess overhead.

    Parameters
    ----------
    env_overrides
        Environment variable overrides.
    config_overrides
        Configuration overrides.
    working_dir
        Working directory for invocation.
    """

    env_overrides: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    working_dir: Path | None = None

    def with_env(self, **env: str) -> CliTestHarness:
        """Create harness with environment overrides.

        Parameters
        ----------
        **env
            Environment variables.

        Returns
        -------
        CliTestHarness
            New harness with overrides.
        """
        return CliTestHarness(
            env_overrides={**self.env_overrides, **env},
            config_overrides=self.config_overrides,
            working_dir=self.working_dir,
        )

    def with_config(self, **config: Any) -> CliTestHarness:
        """Create harness with config overrides.

        Parameters
        ----------
        **config
            Configuration values.

        Returns
        -------
        CliTestHarness
            New harness with overrides.
        """
        return CliTestHarness(
            env_overrides=self.env_overrides,
            config_overrides={**self.config_overrides, **config},
            working_dir=self.working_dir,
        )

    def with_cwd(self, path: Path) -> CliTestHarness:
        """Create harness with working directory.

        Parameters
        ----------
        path
            Working directory.

        Returns
        -------
        CliTestHarness
            New harness with cwd.
        """
        return CliTestHarness(
            env_overrides=self.env_overrides,
            config_overrides=self.config_overrides,
            working_dir=path,
        )

    def invoke(self, args: list[str]) -> CliInvocationResult:
        """Invoke CLI with arguments.

        Parameters
        ----------
        args
            Command line arguments.

        Returns
        -------
        CliInvocationResult
            Invocation result.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_cwd = os.getcwd()
        old_env = dict(os.environ)

        try:
            # Set up environment
            sys.argv = ["codeintel", *args]
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            for key, value in self.env_overrides.items():
                os.environ[key] = value

            if self.working_dir:
                os.chdir(self.working_dir)

            # Invoke CLI
            exit_code = 0
            exception = None
            try:
                app()
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                exception = e
                exit_code = 1

            return CliInvocationResult(
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=exception,
            )

        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def invoke_json(self, args: list[str]) -> dict[str, Any]:
        """Invoke CLI and parse JSON output.

        Parameters
        ----------
        args
            Command line arguments (--format=json added).

        Returns
        -------
        dict[str, Any]
            Parsed JSON output.

        Raises
        ------
        AssertionError
            If invocation failed.
        """
        result = self.invoke([*args, "--format=json"])
        assert result.success, f"CLI failed: {result.stderr}"
        return result.json()


@dataclass
class GoldenFileAssertion:
    """Helper for golden file testing.

    Parameters
    ----------
    golden_dir
        Directory containing golden files.
    update_mode
        Whether to update golden files.
    """

    golden_dir: Path
    update_mode: bool = False

    def assert_matches(
        self,
        name: str,
        actual: str,
        *,
        normalize: bool = True,
    ) -> None:
        """Assert output matches golden file.

        Parameters
        ----------
        name
            Golden file name.
        actual
            Actual output.
        normalize
            Normalize whitespace.

        Raises
        ------
        AssertionError
            If output doesn't match.
        """
        golden_path = self.golden_dir / name

        if normalize:
            actual = actual.strip() + "\n"

        if self.update_mode:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(actual)
            return

        if not golden_path.exists():
            raise AssertionError(
                f"Golden file not found: {golden_path}\n"
                f"Run with UPDATE_GOLDEN=1 to create it.\n"
                f"Actual output:\n{actual}"
            )

        expected = golden_path.read_text()
        if normalize:
            expected = expected.strip() + "\n"

        if actual != expected:
            raise AssertionError(
                f"Output doesn't match golden file: {golden_path}\n"
                f"Expected:\n{expected}\n"
                f"Actual:\n{actual}"
            )

    def assert_json_matches(
        self,
        name: str,
        actual: dict[str, Any],
        *,
        ignore_keys: set[str] | None = None,
    ) -> None:
        """Assert JSON output matches golden file.

        Parameters
        ----------
        name
            Golden file name.
        actual
            Actual JSON data.
        ignore_keys
            Keys to ignore in comparison.
        """
        actual_str = json.dumps(actual, indent=2, sort_keys=True)
        
        if ignore_keys:
            # Remove ignored keys for comparison
            def remove_keys(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {
                        k: remove_keys(v)
                        for k, v in obj.items()
                        if k not in ignore_keys
                    }
                elif isinstance(obj, list):
                    return [remove_keys(item) for item in obj]
                return obj
            
            actual = remove_keys(actual)
            actual_str = json.dumps(actual, indent=2, sort_keys=True)

        self.assert_matches(name, actual_str)
```

#### File: `tests/cli/conftest.py`

```python
"""Shared fixtures for CLI testing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.cli._harness import CliTestHarness, GoldenFileAssertion

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def cli() -> CliTestHarness:
    """Provide CLI test harness."""
    return CliTestHarness()


@pytest.fixture
def cli_with_json(cli: CliTestHarness) -> CliTestHarness:
    """Provide CLI harness configured for JSON output."""
    return cli.with_env(CODEINTEL_OUTPUT_FORMAT="json")


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenFileAssertion:
    """Provide golden file assertion helper."""
    test_dir = Path(request.fspath).parent
    golden_dir = test_dir / "_golden"
    update_mode = os.environ.get("UPDATE_GOLDEN", "").lower() in ("1", "true")
    return GoldenFileAssertion(golden_dir=golden_dir, update_mode=update_mode)


@pytest.fixture
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """Provide isolated config directory."""
    config_dir = tmp_path / ".codeintel"
    config_dir.mkdir()
    
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)
    
    yield config_dir
    
    if old_home:
        os.environ["HOME"] = old_home
```

#### Example Test

```python
"""Tests for build commands."""

from __future__ import annotations

from tests.cli._harness import CliTestHarness, GoldenFileAssertion


class TestBuildStatus:
    """Tests for build status command."""

    def test_build_status_text_output(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test build status text output matches golden file."""
        result = cli.invoke(["build", "status"])
        
        assert result.success
        golden.assert_matches("build_status_text.txt", result.stdout)

    def test_build_status_json_output(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test build status JSON output matches golden file."""
        data = cli.invoke_json(["build", "status"])
        
        assert data["success"] is True
        golden.assert_json_matches(
            "build_status_json.json",
            data,
            ignore_keys={"timestamp", "duration_ms"},
        )

    def test_build_status_exit_code_on_stale(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test build status returns non-zero when targets are stale."""
        result = cli.with_env(
            CODEINTEL_PROJECT_ROOT="/path/with/stale/targets"
        ).invoke(["build", "status", "--fail-on-stale"])
        
        # Should fail if stale targets exist
        # (actual behavior depends on project state)
        assert result.exit_code in (0, 1)
```

---

## Phase 6.4: Resilience Pattern Integration

### Value Proposition

`cli_resilience.py` defines `RetryPolicy` and `CircuitBreaker`, but they're not integrated. This phase wires resilience into the execution pipeline.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OperationExecutor                                   │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │                    ResilienceMiddleware                         │    │
│   │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │    │
│   │   │ Circuit Check │→ │  Retry Loop   │→ │    Handler    │      │    │
│   │   │ (allow/deny)  │  │ (with backoff)│  │  (execution)  │      │    │
│   │   └───────────────┘  └───────────────┘  └───────────────┘      │    │
│   │          │                  │                   │               │    │
│   │          ▼                  ▼                   ▼               │    │
│   │   CircuitBreaker     RetryPolicy         CliResult[T]          │    │
│   │   (per-operation)   (from OperationSpec)                        │    │
│   └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

#### File: `src/codeintel/cli/resilience_middleware.py`

```python
"""Resilience middleware for operation execution.

Integrates retry policies and circuit breakers into the
OperationExecutor pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from codeintel.cli.cli_middleware import OperationMiddleware
from codeintel.cli.cli_resilience import CircuitBreaker, CircuitState, RetryPolicy
from codeintel.cli.executor import OperationSpec
from codeintel.cli.results import CliResult

LOG = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ResilienceConfig:
    """Configuration for resilience behavior.

    Parameters
    ----------
    default_retry_policy
        Default retry policy for retryable operations.
    circuit_breaker_enabled
        Enable circuit breakers.
    circuit_failure_threshold
        Failures before circuit opens.
    circuit_recovery_timeout
        Seconds before attempting recovery.
    """

    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0


class CircuitBreakerRegistry:
    """Registry of circuit breakers by operation category.

    Parameters
    ----------
    config
        Resilience configuration.
    """

    def __init__(self, config: ResilienceConfig) -> None:
        """Initialize registry."""
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for key.

        Parameters
        ----------
        key
            Circuit breaker key (usually operation category).

        Returns
        -------
        CircuitBreaker
            Circuit breaker instance.
        """
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                failure_threshold=self._config.circuit_failure_threshold,
                recovery_timeout=self._config.circuit_recovery_timeout,
            )
        return self._breakers[key]

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns
        -------
        dict[str, dict[str, Any]]
            Status by key.
        """
        return {
            key: {
                "state": breaker.state.value,
                "failure_count": breaker._failure_count,
                "last_failure": breaker._last_failure_time,
            }
            for key, breaker in self._breakers.items()
        }


class ResilienceMiddleware(OperationMiddleware):
    """Middleware that adds retry and circuit breaker behavior.

    Parameters
    ----------
    config
        Resilience configuration.
    breaker_registry
        Circuit breaker registry.
    on_retry
        Callback for retry events.
    """

    def __init__(
        self,
        config: ResilienceConfig | None = None,
        breaker_registry: CircuitBreakerRegistry | None = None,
        on_retry: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        """Initialize middleware."""
        self._config = config or ResilienceConfig()
        self._breakers = breaker_registry or CircuitBreakerRegistry(self._config)
        self._on_retry = on_retry

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Check circuit breaker before invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.

        Raises
        ------
        CircuitOpenError
            If circuit is open.
        """
        # Get circuit breaker for operation category
        category = self._get_category(op_id)
        if self._config.circuit_breaker_enabled and category:
            breaker = self._breakers.get_breaker(category)
            if not breaker.allow_request():
                from codeintel.cli.cli_errors_v2 import CliError, SERVICE_UNAVAILABLE
                raise CliError(
                    code=SERVICE_UNAVAILABLE,
                    detail=f"Service temporarily unavailable (circuit open for {category})",
                    suggestion="Wait and retry, or check service health",
                )

        return {
            "start_time": time.monotonic(),
            "category": category,
        }

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record success for circuit breaker.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        category = context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_success()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record failure for circuit breaker.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        category = context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_failure()

    def _get_category(self, op_id: str) -> str | None:
        """Extract category from operation ID.

        Parameters
        ----------
        op_id
            Operation identifier.

        Returns
        -------
        str | None
            Category or None.
        """
        if "." in op_id:
            return op_id.split(".")[0]
        return None


def execute_with_retry[T](
    handler: Callable[..., CliResult[T]],
    params: dict[str, Any],
    policy: RetryPolicy,
    *,
    operation_id: str = "",
    on_retry: Callable[[str, int, Exception], None] | None = None,
) -> CliResult[T]:
    """Execute handler with retry policy.

    Parameters
    ----------
    handler
        Handler function.
    params
        Handler parameters.
    policy
        Retry policy.
    operation_id
        Operation identifier for logging.
    on_retry
        Callback for retry events.

    Returns
    -------
    CliResult[T]
        Handler result.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            result = handler(**params)
            if result.success:
                return result
            # Non-success result - check if retryable
            # For now, only retry on actual exceptions
            return result

        except Exception as e:
            last_exception = e

            if not _is_retryable(e, policy):
                raise

            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                LOG.warning(
                    "Operation %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    operation_id,
                    attempt + 1,
                    policy.max_attempts,
                    delay,
                    e,
                )
                if on_retry:
                    on_retry(operation_id, attempt + 1, e)
                time.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception

    # Should not reach here
    from codeintel.cli.cli_errors_v2 import CliError, INTERNAL_ERROR
    raise CliError(
        code=INTERNAL_ERROR,
        detail="Retry loop completed without result",
    )


def _is_retryable(exc: Exception, policy: RetryPolicy) -> bool:
    """Check if exception is retryable.

    Parameters
    ----------
    exc
        Exception to check.
    policy
        Retry policy.

    Returns
    -------
    bool
        True if retryable.
    """
    return isinstance(exc, policy.retryable_exceptions)
```

### Integration with Executor

Update `OperationSpec` and `OperationExecutor`:

```python
# In executor.py

@dataclass(frozen=True)
class OperationSpec(Generic[T]):
    # ... existing fields ...
    retry_policy: RetryPolicy | None = None  # Override default
    circuit_breaker_key: str | None = None   # Custom circuit key


class OperationExecutor:
    def __init__(
        self,
        # ... existing params ...
        resilience_middleware: ResilienceMiddleware | None = None,
    ) -> None:
        self._resilience = resilience_middleware or ResilienceMiddleware()
        # Add to middleware stack
        self._middleware.add(self._resilience)

    def _execute_handler(
        self,
        spec: OperationSpec[T],
        ctx: ExecutionContext,
    ) -> CliResult[T]:
        """Execute handler with resilience."""
        if spec.retryable and spec.retry_policy:
            return execute_with_retry(
                spec.handler,
                ctx.params,
                spec.retry_policy,
                operation_id=spec.operation_id,
            )
        return spec.handler(**ctx.params)
```

---

## Phase 6.5: Configuration Schema Validation

### Value Proposition

Config files are loaded but not validated. This enables:
- Early failure with clear messages
- Self-documenting config format
- IDE autocomplete support via schema

### Implementation

#### File: `src/codeintel/cli/config_schema.py`

```python
"""JSON Schema for CLI configuration.

Provides schema definition and validation for config files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLI_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://codeintel.dev/schemas/cli-config.json",
    "title": "CodeIntel CLI Configuration",
    "description": "Configuration schema for the CodeIntel CLI",
    "type": "object",
    "properties": {
        "output_format": {
            "type": "string",
            "enum": ["text", "json"],
            "default": "text",
            "description": "Default output format",
        },
        "color": {
            "type": "boolean",
            "default": True,
            "description": "Enable colored output",
        },
        "progress": {
            "type": "boolean",
            "default": True,
            "description": "Show progress bars for long operations",
        },
        "progress_threshold": {
            "type": "number",
            "minimum": 0,
            "default": 2.0,
            "description": "Minimum seconds before showing progress",
        },
        "telemetry": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable telemetry collection",
                },
                "endpoint": {
                    "type": "string",
                    "format": "uri",
                    "description": "OTLP collector endpoint",
                },
                "service_name": {
                    "type": "string",
                    "default": "codeintel-cli",
                    "description": "Service name for traces",
                },
            },
            "additionalProperties": False,
        },
        "retry": {
            "type": "object",
            "properties": {
                "max_attempts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Maximum retry attempts",
                },
                "initial_delay": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                    "description": "Initial retry delay in seconds",
                },
                "backoff_factor": {
                    "type": "number",
                    "minimum": 1,
                    "default": 2.0,
                    "description": "Exponential backoff multiplier",
                },
                "max_delay": {
                    "type": "number",
                    "minimum": 0,
                    "default": 30.0,
                    "description": "Maximum retry delay",
                },
            },
            "additionalProperties": False,
        },
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "WARNING",
            "description": "Logging level",
        },
        "project_root": {
            "type": "string",
            "description": "Default project root path",
        },
        "plugins": {
            "type": "object",
            "properties": {
                "directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional plugin directories",
                },
                "disabled": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Plugins to disable",
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


@dataclass
class ConfigValidationError:
    """Configuration validation error.

    Parameters
    ----------
    path
        JSON path to error location.
    message
        Error message.
    value
        Invalid value.
    """

    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        """Format error message."""
        if self.value is not None:
            return f"{self.path}: {self.message} (got: {self.value!r})"
        return f"{self.path}: {self.message}"


def validate_config(config: dict[str, Any]) -> list[ConfigValidationError]:
    """Validate configuration against schema.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        Validation errors (empty if valid).
    """
    errors: list[ConfigValidationError] = []

    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(CLI_CONFIG_SCHEMA)
        for error in validator.iter_errors(config):
            path = ".".join(str(p) for p in error.path) or "(root)"
            errors.append(ConfigValidationError(
                path=path,
                message=error.message,
                value=error.instance if error.path else None,
            ))
    except ImportError:
        # Fallback to basic validation
        errors.extend(_basic_validate(config, CLI_CONFIG_SCHEMA, ""))

    return errors


def _basic_validate(
    config: Any,
    schema: dict[str, Any],
    path: str,
) -> list[ConfigValidationError]:
    """Basic validation without jsonschema library.

    Parameters
    ----------
    config
        Value to validate.
    schema
        Schema to validate against.
    path
        Current path.

    Returns
    -------
    list[ConfigValidationError]
        Validation errors.
    """
    errors: list[ConfigValidationError] = []

    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(config, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)

        for key, value in config.items():
            key_path = f"{path}.{key}" if path else key
            if key in properties:
                errors.extend(_basic_validate(value, properties[key], key_path))
            elif not additional:
                errors.append(ConfigValidationError(
                    path=key_path,
                    message="Unknown property",
                ))

    elif schema_type == "string" and not isinstance(config, str):
        errors.append(ConfigValidationError(
            path=path,
            message="Expected string",
            value=config,
        ))

    elif schema_type == "boolean" and not isinstance(config, bool):
        errors.append(ConfigValidationError(
            path=path,
            message="Expected boolean",
            value=config,
        ))

    elif schema_type == "number" and not isinstance(config, (int, float)):
        errors.append(ConfigValidationError(
            path=path,
            message="Expected number",
            value=config,
        ))

    elif schema_type == "integer" and not isinstance(config, int):
        errors.append(ConfigValidationError(
            path=path,
            message="Expected integer",
            value=config,
        ))

    # Check enum
    if "enum" in schema and config not in schema["enum"]:
        errors.append(ConfigValidationError(
            path=path,
            message=f"Must be one of: {schema['enum']}",
            value=config,
        ))

    return errors


def write_schema(path: Path) -> None:
    """Write schema to file.

    Parameters
    ----------
    path
        Output path.
    """
    path.write_text(json.dumps(CLI_CONFIG_SCHEMA, indent=2))


def get_schema_url() -> str:
    """Get URL for schema file.

    Returns
    -------
    str
        Schema URL.
    """
    return "https://codeintel.dev/schemas/cli-config.json"
```

### Integration with Config Loader

Update `config_loader.py`:

```python
from codeintel.cli.config_schema import validate_config, ConfigValidationError

def load_config(
    *,
    config_file: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> ResolvedConfig:
    """Load configuration from all sources.
    
    Parameters
    ----------
    validate
        Whether to validate against schema.
        
    Raises
    ------
    ConfigValidationError
        If validation fails.
    """
    # ... existing loading code ...
    
    if validate:
        errors = validate_config(merged)
        if errors:
            from codeintel.cli.cli_errors_v2 import CliError, CONFIG_SCHEMA_VIOLATION
            raise CliError(
                code=CONFIG_SCHEMA_VIOLATION,
                detail=f"Configuration validation failed: {len(errors)} error(s)",
                extensions={"errors": [str(e) for e in errors]},
            )
    
    return _build_resolved_config(merged, sources)
```

---

## Phase 6.6: Observability Deep Integration

### Value Proposition

Phase 5 defined telemetry; Phase 6 ensures it's automatic for all operations:
- Every operation gets a trace span
- Metrics are recorded automatically
- Structured logs include trace context

### Implementation

#### File: `src/codeintel/cli/observability.py`

```python
"""Observability integration for CLI operations.

Provides automatic tracing, metrics, and structured logging
for all operations flowing through the executor.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_middleware import OperationMiddleware
from codeintel.cli.telemetry import (
    OperationMetrics,
    TelemetryConfig,
    TelemetryProvider,
    get_operation_metrics,
    get_telemetry_provider,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

LOG = logging.getLogger(__name__)


@dataclass
class ObservabilityConfig:
    """Configuration for observability features.

    Parameters
    ----------
    tracing_enabled
        Enable trace spans.
    metrics_enabled
        Enable metrics collection.
    structured_logging
        Enable structured log format.
    log_params
        Log operation parameters.
    log_results
        Log operation results.
    """

    tracing_enabled: bool = True
    metrics_enabled: bool = True
    structured_logging: bool = True
    log_params: bool = False  # Privacy consideration
    log_results: bool = False  # Performance consideration


class ObservabilityMiddleware(OperationMiddleware):
    """Middleware that adds comprehensive observability.

    Parameters
    ----------
    config
        Observability configuration.
    telemetry
        Telemetry provider.
    metrics
        Metrics collector.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        telemetry: TelemetryProvider | None = None,
        metrics: OperationMetrics | None = None,
    ) -> None:
        """Initialize middleware."""
        self._config = config or ObservabilityConfig()
        self._telemetry = telemetry or get_telemetry_provider()
        self._metrics = metrics or get_operation_metrics()

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Start observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.
        """
        context: dict[str, Any] = {
            "start_time": time.monotonic(),
            "operation_id": op_id,
        }

        # Start trace span
        if self._config.tracing_enabled:
            span = self._start_span(op_id, params)
            context["span"] = span

        # Log operation start
        if self._config.structured_logging:
            extra: dict[str, Any] = {"operation_id": op_id}
            if self._config.log_params:
                extra["params"] = _sanitize_params(params)
            LOG.info("Operation started", extra=extra)

        return context

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Complete observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        duration = time.monotonic() - context["start_time"]

        # Record metrics
        if self._config.metrics_enabled:
            self._metrics.record_operation(
                op_id,
                success=True,
                duration_seconds=duration,
            )

        # End trace span
        span = context.get("span")
        if span is not None:
            span.set_attribute("cli.success", True)
            span.set_attribute("cli.duration_ms", duration * 1000)
            span.end()

        # Log completion
        if self._config.structured_logging:
            extra: dict[str, Any] = {
                "operation_id": op_id,
                "duration_ms": duration * 1000,
                "success": True,
            }
            LOG.info("Operation completed", extra=extra)

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error in observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        duration = time.monotonic() - context["start_time"]

        # Record metrics
        if self._config.metrics_enabled:
            self._metrics.record_operation(
                op_id,
                success=False,
                duration_seconds=duration,
            )

        # End trace span with error
        span = context.get("span")
        if span is not None:
            span.set_attribute("cli.success", False)
            span.set_attribute("cli.error_type", type(exc).__name__)
            span.record_exception(exc)
            span.end()

        # Log error
        if self._config.structured_logging:
            extra: dict[str, Any] = {
                "operation_id": op_id,
                "duration_ms": duration * 1000,
                "success": False,
                "error_type": type(exc).__name__,
            }
            LOG.error("Operation failed", extra=extra, exc_info=exc)

    def _start_span(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> Span | None:
        """Start a trace span for operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        Span | None
            Started span or None.
        """
        tracer = self._telemetry.tracer
        if tracer is None:
            return None

        span = tracer.start_span(
            f"cli.operation.{op_id}",
            attributes={
                "cli.operation_id": op_id,
                "cli.param_count": len(params),
            },
        )
        return span


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive data from params for logging.

    Parameters
    ----------
    params
        Parameters to sanitize.

    Returns
    -------
    dict[str, Any]
        Sanitized parameters.
    """
    sensitive_keys = {"password", "token", "secret", "key", "credential"}
    sanitized = {}
    for key, value in params.items():
        if any(s in key.lower() for s in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > 100:
            sanitized[key] = f"{value[:100]}... (truncated)"
        else:
            sanitized[key] = value
    return sanitized


class StructuredLogFormatter(logging.Formatter):
    """Log formatter that outputs structured JSON.

    Parameters
    ----------
    include_trace
        Include trace context in logs.
    """

    def __init__(self, *, include_trace: bool = True) -> None:
        """Initialize formatter."""
        super().__init__()
        self._include_trace = include_trace

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Parameters
        ----------
        record
            Log record.

        Returns
        -------
        str
            JSON formatted log.
        """
        import json

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        for key in ("operation_id", "duration_ms", "success", "error_type"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add trace context
        if self._include_trace:
            trace_ctx = self._get_trace_context()
            if trace_ctx:
                log_data.update(trace_ctx)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

    def _get_trace_context(self) -> dict[str, str] | None:
        """Get current trace context.

        Returns
        -------
        dict[str, str] | None
            Trace context or None.
        """
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx.trace_id:
                return {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                }
        except ImportError:
            pass
        return None


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    include_trace: bool = True,
) -> None:
    """Configure structured logging for CLI.

    Parameters
    ----------
    level
        Log level.
    include_trace
        Include trace context.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter(include_trace=include_trace))

    root = logging.getLogger("codeintel.cli")
    root.setLevel(level)
    root.addHandler(handler)
```

---

## Phase 6.7: Command Composition & Pipelines

### Value Proposition

Users need to chain operations and integrate with shell pipelines:
- JSON Lines output for streaming
- Pipe-friendly structured output
- Batch operation support

### Implementation

#### File: `src/codeintel/cli/pipelines.py`

```python
"""Pipeline support for CLI operations.

Enables chaining operations, streaming output, and
batch execution from files.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from codeintel.cli.executor import OperationSpec, get_executor
from codeintel.cli.operation_registry import get_operation_registry
from codeintel.cli.results import CliResult


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Parameters
    ----------
    stream_output
        Emit results as JSON Lines.
    fail_fast
        Stop on first error.
    continue_on_error
        Continue batch on error.
    max_parallel
        Maximum parallel executions.
    """

    stream_output: bool = False
    fail_fast: bool = False
    continue_on_error: bool = True
    max_parallel: int = 1


class StreamingRenderer:
    """Render results as JSON Lines for streaming.

    Parameters
    ----------
    output
        Output stream.
    """

    def __init__(self, output: TextIO = sys.stdout) -> None:
        """Initialize renderer."""
        self._output = output

    def emit(self, result: CliResult[Any]) -> None:
        """Emit result as JSON line.

        Parameters
        ----------
        result
            Result to emit.
        """
        data = result.to_dict()
        self._output.write(json.dumps(data))
        self._output.write("\n")
        self._output.flush()

    def emit_progress(self, index: int, total: int, operation_id: str) -> None:
        """Emit progress indicator.

        Parameters
        ----------
        index
            Current index.
        total
            Total items.
        operation_id
            Current operation.
        """
        data = {
            "type": "progress",
            "index": index,
            "total": total,
            "operation_id": operation_id,
        }
        self._output.write(json.dumps(data))
        self._output.write("\n")
        self._output.flush()


@dataclass
class BatchOperation:
    """Single operation in a batch.

    Parameters
    ----------
    operation_id
        Operation to execute.
    params
        Operation parameters.
    name
        Optional name for tracking.
    """

    operation_id: str
    params: dict[str, Any]
    name: str | None = None


@dataclass
class BatchResult:
    """Result of batch execution.

    Parameters
    ----------
    total
        Total operations.
    succeeded
        Successful operations.
    failed
        Failed operations.
    results
        Individual results.
    """

    total: int
    succeeded: int
    failed: int
    results: list[tuple[BatchOperation, CliResult[Any]]]


def load_batch(path: Path) -> list[BatchOperation]:
    """Load batch operations from file.

    Parameters
    ----------
    path
        Path to batch file (YAML or JSON).

    Returns
    -------
    list[BatchOperation]
        Operations to execute.
    """
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    operations = []
    for item in data.get("operations", []):
        operations.append(BatchOperation(
            operation_id=item["operation"],
            params=item.get("params", {}),
            name=item.get("name"),
        ))
    return operations


def execute_batch(
    operations: list[BatchOperation],
    config: PipelineConfig | None = None,
) -> BatchResult:
    """Execute batch of operations.

    Parameters
    ----------
    operations
        Operations to execute.
    config
        Pipeline configuration.

    Returns
    -------
    BatchResult
        Batch execution result.
    """
    config = config or PipelineConfig()
    executor = get_executor()
    registry = get_operation_registry()
    renderer = StreamingRenderer() if config.stream_output else None

    results: list[tuple[BatchOperation, CliResult[Any]]] = []
    succeeded = 0
    failed = 0

    for i, batch_op in enumerate(operations):
        spec = registry.get(batch_op.operation_id)
        if spec is None:
            result = CliResult.error_from_message(
                f"Unknown operation: {batch_op.operation_id}"
            )
        else:
            exec_result = executor.execute(
                spec,
                batch_op.params,
                render=False,
            )
            result = exec_result.result

        if result.success:
            succeeded += 1
        else:
            failed += 1

        results.append((batch_op, result))

        if renderer:
            renderer.emit(result)

        if not result.success and config.fail_fast:
            break

    return BatchResult(
        total=len(operations),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


def read_stdin_operations() -> Iterator[BatchOperation]:
    """Read operations from stdin (JSON Lines).

    Yields
    ------
    BatchOperation
        Operations from stdin.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        yield BatchOperation(
            operation_id=data["operation"],
            params=data.get("params", {}),
            name=data.get("name"),
        )
```

#### Batch Commands

```python
# In cyclopts_ops.py

@op_app.command()
def batch(
    file: Annotated[Path | None, Parameter(help="Batch file path")] = None,
    from_stdin: Annotated[bool, Parameter(help="Read from stdin")] = False,
    stream: Annotated[bool, Parameter(help="Stream output as JSON Lines")] = False,
    fail_fast: Annotated[bool, Parameter(help="Stop on first error")] = False,
) -> None:
    """Execute batch of operations.

    Examples
    --------
    codeintel op batch operations.yaml
    codeintel op batch --from-stdin < operations.jsonl
    echo '{"operation":"build.status"}' | codeintel op batch --from-stdin
    """
    from codeintel.cli.pipelines import (
        BatchResult,
        PipelineConfig,
        execute_batch,
        load_batch,
        read_stdin_operations,
    )

    if from_stdin:
        operations = list(read_stdin_operations())
    elif file:
        operations = load_batch(file)
    else:
        print("Provide --file or --from-stdin")
        raise SystemExit(1)

    config = PipelineConfig(
        stream_output=stream,
        fail_fast=fail_fast,
    )

    result = execute_batch(operations, config)

    if not stream:
        print(f"Executed {result.total} operations: {result.succeeded} succeeded, {result.failed} failed")

    if result.failed > 0:
        raise SystemExit(1)
```

---

## Phase 6.8: Interactive Shell Mode

### Value Proposition

An interactive REPL provides:
- Faster iteration without CLI overhead
- Session state and history
- Tab completion for operations
- Exploratory usage patterns

### Implementation

#### File: `src/codeintel/cli/shell.py`

```python
"""Interactive shell for CLI operations.

Provides REPL-style interaction with the CLI, maintaining
session state and providing rich completion.
"""

from __future__ import annotations

import json
import readline
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.executor import get_executor
from codeintel.cli.introspection import list_all_operations, search_operations
from codeintel.cli.operation_registry import get_operation_registry


@dataclass
class ShellSession:
    """Interactive shell session state.

    Parameters
    ----------
    history
        Command history.
    variables
        Session variables.
    last_result
        Last operation result.
    """

    history: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    last_result: dict[str, Any] | None = None


class ShellCompleter:
    """Tab completion for shell commands.

    Parameters
    ----------
    session
        Shell session.
    """

    def __init__(self, session: ShellSession) -> None:
        """Initialize completer."""
        self._session = session
        self._operations: list[str] = []
        self._refresh_operations()

    def _refresh_operations(self) -> None:
        """Refresh operation list."""
        registry = get_operation_registry()
        self._operations = [spec.operation_id for spec in registry.list_operations()]

    def complete(self, text: str, state: int) -> str | None:
        """Complete text.

        Parameters
        ----------
        text
            Text to complete.
        state
            Completion state.

        Returns
        -------
        str | None
            Completion or None.
        """
        buffer = readline.get_line_buffer()
        parts = buffer.split()

        if not parts or (len(parts) == 1 and not buffer.endswith(" ")):
            # Complete command
            commands = [
                "call", "list", "search", "help", "history",
                "set", "get", "export", "quit", "exit",
            ]
            matches = [c for c in commands if c.startswith(text)]
        elif parts[0] == "call":
            # Complete operation ID
            matches = [op for op in self._operations if op.startswith(text)]
        else:
            matches = []

        if state < len(matches):
            return matches[state]
        return None


class InteractiveShell:
    """Interactive CLI shell.

    Parameters
    ----------
    session
        Shell session state.
    """

    def __init__(self, session: ShellSession | None = None) -> None:
        """Initialize shell."""
        self._session = session or ShellSession()
        self._completer = ShellCompleter(self._session)
        self._running = False

    def run(self) -> None:
        """Run interactive shell."""
        self._setup_readline()
        self._print_banner()
        self._running = True

        while self._running:
            try:
                line = input("codeintel> ").strip()
                if line:
                    self._execute_command(line)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

    def _setup_readline(self) -> None:
        """Set up readline for completion and history."""
        readline.set_completer(self._completer.complete)
        readline.parse_and_bind("tab: complete")

        history_file = Path.home() / ".codeintel" / "shell_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

    def _print_banner(self) -> None:
        """Print welcome banner."""
        print("CodeIntel Interactive Shell")
        print("Type 'help' for commands, 'quit' to exit")
        print()

    def _execute_command(self, line: str) -> None:
        """Execute shell command.

        Parameters
        ----------
        line
            Command line.
        """
        self._session.history.append(line)

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")
            return

        if not parts:
            return

        cmd = parts[0]
        args = parts[1:]

        handlers = {
            "call": self._cmd_call,
            "list": self._cmd_list,
            "search": self._cmd_search,
            "help": self._cmd_help,
            "history": self._cmd_history,
            "set": self._cmd_set,
            "get": self._cmd_get,
            "export": self._cmd_export,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            handler(args)
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")

    def _cmd_call(self, args: list[str]) -> None:
        """Execute operation."""
        if not args:
            print("Usage: call <operation_id> [--param=value ...]")
            return

        operation_id = args[0]
        params = self._parse_params(args[1:])

        registry = get_operation_registry()
        spec = registry.get(operation_id)

        if spec is None:
            print(f"Unknown operation: {operation_id}")
            return

        executor = get_executor()
        result = executor.execute(spec, params, render=False)

        if result.result.success:
            self._session.last_result = result.result.data
            print(json.dumps(result.result.data, indent=2, default=str))
        else:
            print(f"Error: {result.result.error}")

    def _cmd_list(self, args: list[str]) -> None:
        """List operations."""
        operations = list_all_operations()
        for info in sorted(operations, key=lambda x: x.operation_id):
            print(f"  {info.operation_id:30} {info.description}")

    def _cmd_search(self, args: list[str]) -> None:
        """Search operations."""
        if not args:
            print("Usage: search <query>")
            return

        query = " ".join(args)
        results = search_operations(query)
        if not results:
            print(f"No operations matching: {query}")
            return

        for info in results:
            print(f"  {info.operation_id}: {info.description}")

    def _cmd_help(self, args: list[str]) -> None:
        """Show help."""
        print("Commands:")
        print("  call <operation> [params]  Execute operation")
        print("  list                       List all operations")
        print("  search <query>             Search operations")
        print("  set <name> <value>         Set session variable")
        print("  get <name>                 Get session variable")
        print("  history                    Show command history")
        print("  export [file]              Export session as script")
        print("  help                       Show this help")
        print("  quit                       Exit shell")

    def _cmd_history(self, args: list[str]) -> None:
        """Show history."""
        for i, cmd in enumerate(self._session.history[-20:], 1):
            print(f"  {i:3d}  {cmd}")

    def _cmd_set(self, args: list[str]) -> None:
        """Set session variable."""
        if len(args) < 2:
            print("Usage: set <name> <value>")
            return

        name = args[0]
        value = " ".join(args[1:])

        # Try to parse as JSON
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass

        self._session.variables[name] = value
        print(f"Set {name} = {value!r}")

    def _cmd_get(self, args: list[str]) -> None:
        """Get session variable."""
        if not args:
            # Show all variables
            for name, value in self._session.variables.items():
                print(f"  {name} = {value!r}")
            return

        name = args[0]
        value = self._session.variables.get(name)
        if value is not None:
            print(f"{name} = {value!r}")
        else:
            print(f"Variable not set: {name}")

    def _cmd_export(self, args: list[str]) -> None:
        """Export session as script."""
        lines = ["#!/usr/bin/env bash", "# Exported from codeintel shell", ""]

        for cmd in self._session.history:
            if cmd.startswith("call "):
                parts = shlex.split(cmd)
                if len(parts) >= 2:
                    lines.append(f"codeintel op call {' '.join(parts[1:])}")

        script = "\n".join(lines)

        if args:
            path = Path(args[0])
            path.write_text(script)
            print(f"Exported to {path}")
        else:
            print(script)

    def _cmd_quit(self, args: list[str]) -> None:
        """Exit shell."""
        self._running = False

    def _parse_params(self, args: list[str]) -> dict[str, Any]:
        """Parse parameters from command line.

        Parameters
        ----------
        args
            Parameter arguments.

        Returns
        -------
        dict[str, Any]
            Parsed parameters.
        """
        params: dict[str, Any] = {}

        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                key = key.lstrip("-")

                # Try JSON parse
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

                params[key] = value

        # Substitute session variables
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in self._session.variables:
                    params[key] = self._session.variables[var_name]

        return params


def start_shell() -> None:
    """Start interactive shell."""
    shell = InteractiveShell()
    shell.run()
```

#### Shell Command

```python
# In cyclopts_app.py

@app.command()
def shell() -> None:
    """Start interactive shell.

    Examples
    --------
    codeintel shell
    """
    from codeintel.cli.shell import start_shell
    start_shell()
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority | Effort |
|-------|----------|--------------|----------|--------|
| 6.1 Executor Migration | 3-4 days | None | Critical | High |
| 6.2 Error Taxonomy | 2-3 days | 6.1 | High | Medium |
| 6.3 Testing Infrastructure | 3-4 days | 6.1 | High | High |
| 6.4 Resilience Integration | 2-3 days | 6.1 | High | Medium |
| 6.5 Config Validation | 1-2 days | None | Medium | Low |
| 6.6 Observability | 2-3 days | 6.1 | Medium | Medium |
| 6.7 Pipelines | 2-3 days | 6.1 | Medium | Medium |
| 6.8 Interactive Shell | 3-4 days | 6.1 | Low | High |

**Total estimated time: 18-26 days**

### Recommended Order

```
Week 1:       [======= Phase 6.1 =======]
Week 1-2:                  [=== 6.5 ===]
Week 2:            [==== 6.2 ====][==== 6.4 ====]
Week 2-3:                    [======= 6.3 =======]
Week 3:                              [=== 6.6 ===]
Week 3-4:                              [=== 6.7 ===]
Week 4:                                    [==== 6.8 ====]
```

---

## Success Metrics

### Technical Quality

- [ ] 100% of commands use OperationExecutor
- [ ] All errors use RFC 9457 Problem Details with URN types
- [ ] Test coverage ≥ 85% for CLI modules
- [ ] Zero monkeypatch usage in CLI tests
- [ ] Config files validated against JSON Schema

### Operational Excellence

- [ ] All operations have trace spans
- [ ] Retry behavior visible in logs
- [ ] Circuit breakers prevent cascade failures
- [ ] Structured logs include correlation IDs

### Developer Experience

- [ ] CLI test harness available in conftest
- [ ] Golden file testing for output stability
- [ ] Batch execution supports YAML/JSON
- [ ] Interactive shell with tab completion

### User Experience

- [ ] Consistent error messages with suggestions
- [ ] Streaming output for long operations
- [ ] Session state in interactive mode
- [ ] Export session as script

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/cli/cli_errors_v2.py` | Enhanced RFC 9457 error types |
| `src/codeintel/cli/resilience_middleware.py` | Retry/circuit breaker middleware |
| `src/codeintel/cli/config_schema.py` | JSON Schema for configuration |
| `src/codeintel/cli/observability.py` | Deep observability integration |
| `src/codeintel/cli/pipelines.py` | Batch and streaming support |
| `src/codeintel/cli/shell.py` | Interactive REPL |
| `src/codeintel/cli/operations/docs_operations.py` | Docs operation specs |
| `src/codeintel/cli/operations/graph_operations.py` | Graph operation specs |
| `src/codeintel/cli/operations/history_operations.py` | History operation specs |
| `src/codeintel/cli/operations/ide_operations.py` | IDE operation specs |
| `tests/cli/_harness/__init__.py` | Test harness infrastructure |
| `tests/cli/conftest.py` | Shared test fixtures |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/executor.py` | Resilience integration, error handling |
| `src/codeintel/cli/config_loader.py` | Schema validation |
| `src/codeintel/cli/cyclopts_app.py` | Shell command, pipeline commands |
| `src/codeintel/cli/cyclopts_ops.py` | Batch command |
| `src/codeintel/cli/cyclopts_*.py` | Executor migration |

---

*End of Phase 6 Implementation Plan*

