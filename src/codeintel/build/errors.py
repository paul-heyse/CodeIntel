"""Unified error hierarchy for the build system.

This module provides a structured error system with:
- Rich error messages with user_message and actionable_hint
- Error collection for continue-and-collect semantics
- Categorized errors (contract, resource, execution)

All build system errors derive from BuildError, which provides
consistent formatting for CLI display and logging.

Example
-------
>>> from codeintel.build.errors import SchemaNotFoundError, BuildErrorCollection
>>> error = SchemaNotFoundError("scip", "core.scip_symbols")
>>> print(error.user_message)
Target 'scip' outputs table 'core.scip_symbols' which has no schema defined
>>> print(error.actionable_hint)
Add schema for 'core.scip_symbols' to the target's OutputContract
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import TYPE_CHECKING

from codeintel.core.errors.problem_details import ProblemDetail, ProblemDetailBuilder
from codeintel.core.errors.taxonomy import ErrorCode

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "ArtifactNotFoundError",
    "BuildError",
    "BuildErrorCollection",
    "BuildProblemError",
    "ColumnCountMismatchError",
    "ContractError",
    "ContractViolationError",
    "CycleDetectedError",
    "DependencyUnavailableError",
    "ExecutionError",
    "GatewayNotAvailableError",
    "MissingDependencyError",
    "PluginExecutionError",
    "RegistryValidationError",
    "ResourceError",
    "SchemaNotFoundError",
    "SchemaValidationError",
    "SessionNotInitializedError",
    "TargetNotFoundError",
    "TargetTimeoutError",
    "ToolNotAvailableError",
]


class BuildError(Exception):
    """Base class for all build system errors.

    Build errors provide rich context for both humans and machines:
    - user_message: Human-readable description for CLI output
    - actionable_hint: Suggestion for how to fix the error
    - Full context (target, table, plugin names)

    All subclasses must implement user_message property.
    """

    @property
    def user_message(self) -> str:
        """Return human-readable error message for CLI output.

        Returns
        -------
        str
            Description suitable for display to users.
        """
        return str(self)

    @property
    def actionable_hint(self) -> str | None:
        """Return suggestion for fixing the error.

        Returns
        -------
        str | None
            Action the user can take, or None if no suggestion.
        """
        return None

    @property
    def error_code(self) -> str:
        """Return error code for programmatic handling.

        Returns
        -------
        str
            Short identifier like "CONTRACT_SCHEMA_NOT_FOUND".
        """
        return self.__class__.__name__.upper()


class BuildProblemError(BuildError):
    """Build error carrying RFC 9457 Problem Details."""

    def __init__(self, problem_detail: ProblemDetail) -> None:
        """Initialize with a ProblemDetail payload.

        Parameters
        ----------
        problem_detail
            Problem detail payload to attach to the error.
        """
        self.problem_detail = problem_detail
        super().__init__(problem_detail.detail or problem_detail.title)

    @classmethod
    def from_detail(
        cls,
        *,
        code: str,
        title: str,
        detail: str,
        status: int = 500,
        **extras: object,
    ) -> BuildProblemError:
        """Create a BuildProblemError from ProblemDetailBuilder inputs.

        Parameters
        ----------
        code
            Problem code (e.g., "build.error").
        title
            Short problem title.
        detail
            Detailed error message.
        status
            HTTP-style status code.
        **extras
            Additional context fields.

        Returns
        -------
        BuildProblemError
            BuildProblemError with structured problem detail.
        """
        builder = ProblemDetailBuilder(code=code, title=title, status=status)
        problem = builder.build(detail).with_extensions(**extras)
        return cls(problem)

    @classmethod
    def from_error_code(
        cls,
        *,
        error_code: ErrorCode,
        detail: str,
        **extras: object,
    ) -> BuildProblemError:
        """Create a BuildProblemError from a structured ErrorCode.

        Parameters
        ----------
        error_code
            Canonical error taxonomy entry.
        detail
            Detailed error message.
        **extras
            Additional context fields.

        Returns
        -------
        BuildProblemError
            BuildProblemError with structured problem detail.
        """
        builder = ProblemDetailBuilder(
            code=error_code.code,
            title=error_code.title,
            status=error_code.status,
            type_uri=error_code.type_uri,
        )
        problem = builder.build(detail).with_extensions(**extras)
        return cls(problem)

    @property
    def user_message(self) -> str:
        """Return human-readable error message.

        Returns
        -------
        str
            User-facing error description.
        """
        return self.problem_detail.detail or self.problem_detail.title

    @property
    def actionable_hint(self) -> str | None:
        """Return suggestion for fixing the error when provided.

        Returns
        -------
        str | None
            Suggested action when present.
        """
        suggestion = self.problem_detail.extensions.get("suggestion")
        return suggestion if isinstance(suggestion, str) else None

    @property
    def error_code(self) -> str:
        """Return error code from the problem detail when present.

        Returns
        -------
        str
            Problem detail code or fallback error code.
        """
        code = self.problem_detail.extensions.get("code")
        if isinstance(code, str) and code:
            return code
        if self.problem_detail.type:
            return self.problem_detail.type
        return super().error_code


class ContractError(BuildError):
    """Base class for output contract violations.

    These errors indicate issues with target definitions or
    mismatches between declared contracts and actual outputs.
    """


class SchemaNotFoundError(ContractError):
    """Target references a table with no schema defined.

    Attributes
    ----------
    target
        Target name that references the missing schema.
    table_key
        Fully-qualified table name (e.g., "core.scip_symbols").
    """

    def __init__(self, target: str, table_key: str) -> None:
        self.target = target
        self.table_key = table_key
        super().__init__(f"Schema not found for table '{table_key}' in target '{target}'")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return (
            f"Target '{self.target}' outputs table '{self.table_key}' which has no schema defined"
        )

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return f"Add schema for '{self.table_key}' to the target's OutputContract"


class ColumnCountMismatchError(ContractError):
    """Plugin wrote wrong number of columns to a table.

    Attributes
    ----------
    target
        Target name that produced the error.
    table_key
        Fully-qualified table name.
    expected
        Expected number of columns from schema.
    actual
        Actual number of columns in the data.
    row_index
        Index of the first mismatched row (0-based).
    """

    def __init__(
        self,
        target: str,
        table_key: str,
        expected: int,
        actual: int,
        row_index: int = 0,
    ) -> None:
        self.target = target
        self.table_key = table_key
        self.expected = expected
        self.actual = actual
        self.row_index = row_index
        super().__init__(
            f"Column count mismatch in '{table_key}': expected {expected}, got {actual}"
        )

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return (
            f"Table '{self.table_key}' expects {self.expected} columns, "
            f"but row {self.row_index} has {self.actual} columns"
        )

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return (
            f"Check the plugin for target '{self.target}' - "
            f"ensure it writes all {self.expected} columns defined in the schema"
        )


class ContractViolationError(ContractError):
    """Target attempted to write outside its declared contract.

    This error is raised when strict_contracts mode is enabled and a target
    attempts to write to a table or artifact that is not in its OutputContract.

    Attributes
    ----------
    target
        Target name that violated the contract.
    table_key
        Table key that was written to (if applicable).
    artifact_name
        Artifact name that was written to (if applicable).
    allowed_tables
        Set of table keys allowed by the contract.
    """

    def __init__(
        self,
        target: str,
        *,
        table_key: str | None = None,
        artifact_name: str | None = None,
        allowed_tables: set[str] | None = None,
    ) -> None:
        self.target = target
        self.table_key = table_key
        self.artifact_name = artifact_name
        self.allowed_tables = allowed_tables or set()
        if table_key:
            msg = (
                f"Target '{target}' attempted to write to '{table_key}' "
                f"which is not in its contract"
            )
        elif artifact_name:
            msg = (
                f"Target '{target}' attempted to write artifact '{artifact_name}' "
                f"which is not in its contract"
            )
        else:
            msg = f"Target '{target}' violated its output contract"
        super().__init__(msg)

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        if self.table_key:
            return (
                f"Target '{self.target}' attempted to write to '{self.table_key}' "
                f"which is not in its contract"
            )
        if self.artifact_name:
            return (
                f"Target '{self.target}' attempted to write artifact '{self.artifact_name}' "
                f"which is not in its contract"
            )
        return f"Target '{self.target}' violated its output contract"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        if self.table_key and self.allowed_tables:
            return (
                f"Add '{self.table_key}' to the OutputContract for '{self.target}' "
                f"or remove the write. Allowed tables: {sorted(self.allowed_tables)}"
            )
        if self.artifact_name:
            return (
                f"Add '{self.artifact_name}' to the OutputContract for '{self.target}' "
                f"or remove the write"
            )
        return (
            f"Review the OutputContract for '{self.target}' and ensure all writes "
            f"match declared outputs"
        )


class ArtifactNotFoundError(ContractError):
    """Required input artifact does not exist.

    Attributes
    ----------
    target
        Target name that requires the artifact.
    artifact_name
        Name of the missing artifact.
    path
        Expected path where artifact should exist.
    """

    def __init__(self, target: str, artifact_name: str, path: Path) -> None:
        self.target = target
        self.artifact_name = artifact_name
        self.path = path
        super().__init__(f"Artifact '{artifact_name}' not found at {path}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' requires artifact '{self.artifact_name}' at {self.path}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return f"Ensure the dependency that produces '{self.artifact_name}' runs successfully first"


class SchemaValidationError(ContractError):
    """Schema validation failed during materialization.

    This error is raised when data fails Pandera schema validation
    during table materialization.

    Attributes
    ----------
    table_key
        Fully-qualified table name that failed validation.
    errors
        Validation error details from Pandera.
    target
        Optional target name that produced the invalid data.
    """

    def __init__(
        self,
        table_key: str,
        errors: str,
        target: str | None = None,
    ) -> None:
        self.table_key = table_key
        self.errors = errors
        self.target = target
        super().__init__(f"Schema validation failed for '{table_key}': {errors}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        if self.target:
            return (
                f"Target '{self.target}' produced invalid data for table '{self.table_key}': "
                f"{self.errors}"
            )
        return f"Schema validation failed for table '{self.table_key}': {self.errors}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        if self.target:
            return (
                f"Review the output of target '{self.target}' and ensure it matches "
                f"the schema for '{self.table_key}'"
            )
        return f"Ensure data matches the schema for '{self.table_key}'"


class RegistryValidationError(ContractError):
    """Target registration failed validation.

    This error is raised when a target is registered without
    any implementation (neither plugin nor native module).

    Attributes
    ----------
    target
        Target name that failed validation.
    errors
        List of validation error messages.
    """

    def __init__(self, target: str, errors: list[str]) -> None:
        self.target = target
        self.errors = errors
        super().__init__(f"Registry validation failed for '{target}': {'; '.join(errors)}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' failed registry validation: {'; '.join(self.errors)}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return (
            f"Register a plugin or native module for target '{self.target}' "
            "in the registrations module"
        )


class ResourceError(BuildError):
    """Base class for resource-related errors.

    These errors indicate missing external dependencies like
    tools, services, or required inputs.
    """


class ToolNotAvailableError(ResourceError):
    """Required external tool is not available.

    Attributes
    ----------
    target
        Target name that requires the tool.
    tool
        Tool identifier (e.g., "scip-python", "pyright").
    search_path
        Path where the tool was searched for.
    """

    def __init__(self, target: str, tool: str, search_path: str | None = None) -> None:
        self.target = target
        self.tool = tool
        self.search_path = search_path
        super().__init__(f"Tool '{tool}' not available for target '{target}'")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' requires tool '{self.tool}' which was not found"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        hints: dict[str, str] = {
            "scip-python": "Install with: npm install -g @aspect/scip-python",
            "scip": "Install with: npm install -g @sourcegraph/scip",
            "pyright": "Install with: pip install pyright",
            "pyrefly": "Install with: pip install pyrefly",
            "ruff": "Install with: pip install ruff",
            "coverage": "Install with: pip install coverage",
            "pytest": "Install with: pip install pytest",
        }
        return hints.get(self.tool, f"Install '{self.tool}' and ensure it's on PATH")


class DependencyUnavailableError(ResourceError):
    """A target's dependency is blocked or failed.

    Attributes
    ----------
    target
        Target name that has the blocked dependency.
    dependency
        Name of the dependency that is unavailable.
    reason
        Why the dependency is unavailable.
    """

    def __init__(self, target: str, dependency: str, reason: str) -> None:
        self.target = target
        self.dependency = dependency
        self.reason = reason
        super().__init__(f"Dependency '{dependency}' unavailable for '{target}': {reason}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return (
            f"Target '{self.target}' cannot run because "
            f"dependency '{self.dependency}' {self.reason}"
        )

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return f"Fix the issue with '{self.dependency}' first, then retry"


class GatewayNotAvailableError(ResourceError):
    """Storage gateway is not available in the execution context.

    This typically indicates a configuration issue where the build
    context was created without a gateway reference.

    Attributes
    ----------
    context
        Description of the context where the gateway was needed.
    """

    def __init__(self, context: str = "execution context") -> None:
        self.context = context
        super().__init__(f"Gateway not available in {context}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Storage gateway is not available in {self.context}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return "Ensure the build context is properly initialized with a gateway"


class SessionNotInitializedError(ResourceError):
    """Build session has not been initialized.

    This typically indicates a lifecycle issue where session-dependent
    operations are attempted before the session is ready.

    Attributes
    ----------
    operation
        Description of the operation that requires the session.
    """

    def __init__(self, operation: str = "build operation") -> None:
        self.operation = operation
        super().__init__(f"Session not initialized for {operation}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Build session is not initialized for {self.operation}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return "Ensure the build session is initialized before executing targets"


class ExecutionError(BuildError):
    """Base class for execution-time errors.

    These errors occur during actual plugin execution, not
    during planning or validation.
    """


class PluginExecutionError(ExecutionError):
    """Plugin failed during execution.

    Attributes
    ----------
    target
        Target name that failed.
    plugin
        Plugin name that raised the error.
    cause
        The underlying exception.
    """

    def __init__(self, target: str, plugin: str, cause: Exception) -> None:
        self.target = target
        self.plugin = plugin
        self.cause = cause
        super().__init__(f"Plugin '{plugin}' failed for target '{target}': {cause}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' failed: {self.cause}"

    @property
    def actionable_hint(self) -> str | None:
        """Return suggestion for fixing the error."""
        if isinstance(self.cause, BuildError):
            return self.cause.actionable_hint
        return None


class TargetTimeoutError(ExecutionError):
    """Target exceeded its max_runtime.

    Attributes
    ----------
    target
        Target name that timed out.
    timeout_ms
        Configured timeout in milliseconds.
    elapsed_ms
        Actual elapsed time before timeout.
    """

    def __init__(self, target: str, timeout_ms: int, elapsed_ms: int) -> None:
        self.target = target
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms
        super().__init__(f"Target '{target}' timed out after {elapsed_ms}ms")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return (
            f"Target '{self.target}' timed out after {self.elapsed_ms}ms "
            f"(limit: {self.timeout_ms}ms)"
        )

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return (
            f"Increase execution.max_runtime_ms for target '{self.target}' or optimize the plugin"
        )


class TargetNotFoundError(BuildError):
    """Requested target does not exist in the graph.

    Attributes
    ----------
    target
        Target name that was not found.
    available
        List of available target names for suggestions.
    """

    def __init__(self, target: str, available: list[str] | None = None) -> None:
        self.target = target
        self.available = available or []
        super().__init__(f"Target '{target}' not found")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' does not exist"

    @property
    def actionable_hint(self) -> str | None:
        """Return suggestion for fixing the error."""
        if not self.available:
            return "Check the target name and try again"

        matches = get_close_matches(self.target, self.available, n=3, cutoff=0.6)
        if matches:
            suggestions = ", ".join(f"'{m}'" for m in matches)
            return f"Did you mean: {suggestions}?"
        return f"Available targets: {', '.join(sorted(self.available)[:10])}..."


class MissingDependencyError(BuildError):
    """Target depends on a non-existent target.

    Attributes
    ----------
    target
        Target name with the invalid dependency.
    dependency
        Name of the missing dependency.
    """

    def __init__(self, target: str, dependency: str) -> None:
        self.target = target
        self.dependency = dependency
        super().__init__(f"Target '{target}' depends on unknown target '{dependency}'")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        return f"Target '{self.target}' depends on '{self.dependency}' which does not exist"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return (
            f"Either register target '{self.dependency}' or "
            f"remove it from '{self.target}' dependencies"
        )


class CycleDetectedError(BuildError):
    """Dependency cycle detected in target graph.

    Attributes
    ----------
    targets
        Names of targets involved in the cycle.
    """

    def __init__(self, targets: list[str]) -> None:
        self.targets = targets
        cycle_str = " -> ".join(targets)
        super().__init__(f"Dependency cycle detected: {cycle_str}")

    @property
    def user_message(self) -> str:
        """Return human-readable error message."""
        cycle_str = " -> ".join(self.targets)
        return f"Dependency cycle detected: {cycle_str}"

    @property
    def actionable_hint(self) -> str:
        """Return suggestion for fixing the error."""
        return "Restructure target dependencies to remove the cycle"


@dataclass
class BuildErrorCollection:
    """Collection of errors for continue-and-collect semantics.

    Instead of failing on the first error, the build system can
    collect all errors and report them at the end.

    Attributes
    ----------
    errors
        List of collected BuildError instances.
    warnings
        List of non-fatal warnings.

    Examples
    --------
    >>> collection = BuildErrorCollection()
    >>> collection.add(SchemaNotFoundError("scip", "core.scip_symbols"))
    >>> collection.add(ToolNotAvailableError("typing", "pyright"))
    >>> print(len(collection))
    2
    >>> print(collection.has_errors)
    True
    """

    errors: list[BuildError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, error: BuildError) -> None:
        """Add an error to the collection.

        Parameters
        ----------
        error
            BuildError to add.
        """
        self.errors.append(error)

    def add_warning(self, message: str) -> None:
        """Add a non-fatal warning.

        Parameters
        ----------
        message
            Warning message.
        """
        self.warnings.append(message)

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected.

        Returns
        -------
        bool
            True if there are errors.
        """
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were collected.

        Returns
        -------
        bool
            True if there are warnings.
        """
        return len(self.warnings) > 0

    def __len__(self) -> int:
        """Return number of errors collected.

        Returns
        -------
        int
            Error count.
        """
        return len(self.errors)

    def __bool__(self) -> bool:
        """Return True if there are errors.

        Returns
        -------
        bool
            True if has_errors.
        """
        return self.has_errors

    def by_type(self, error_type: type[BuildError]) -> list[BuildError]:
        """Filter errors by type.

        Parameters
        ----------
        error_type
            Error class to filter by.

        Returns
        -------
        list[BuildError]
            Errors matching the type.
        """
        return [e for e in self.errors if isinstance(e, error_type)]

    def by_target(self, target: str) -> list[BuildError]:
        """Filter errors by target name.

        Parameters
        ----------
        target
            Target name to filter by.

        Returns
        -------
        list[BuildError]
            Errors for the specified target.
        """
        result: list[BuildError] = [
            error for error in self.errors if getattr(error, "target", None) == target
        ]
        return result

    def format_summary(self) -> str:
        """Format a summary of all errors.

        Returns
        -------
        str
            Multi-line summary suitable for CLI output.
        """
        if not self.errors:
            return "No errors"

        lines: list[str] = [f"Build failed with {len(self.errors)} error(s):"]
        lines.append("")

        for i, error in enumerate(self.errors, 1):
            lines.append(f"{i}. {error.error_code}")
            lines.append(f"   {error.user_message}")
            if error.actionable_hint:
                lines.append(f"   Hint: {error.actionable_hint}")
            lines.append("")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {warning}" for warning in self.warnings)

        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        """Raise the first error if any were collected.

        This is useful when you want to collect errors during
        a phase but raise at the end. If no errors were collected,
        this method does nothing.
        """
        if self.errors:
            first_error: BuildError = self.errors[0]
            raise first_error

    def merge(self, other: BuildErrorCollection) -> BuildErrorCollection:
        """Merge another collection into this one.

        Parameters
        ----------
        other
            Collection to merge in.

        Returns
        -------
        BuildErrorCollection
            New collection with all errors from both.
        """
        return BuildErrorCollection(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )
