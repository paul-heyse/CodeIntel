"""Base plugin classes for ingestion plugins.

This module provides a hierarchy of base classes that plugins can inherit from
to minimize boilerplate while retaining full flexibility. The design emphasizes
composition over deep inheritance through trait mixins.

Architecture
------------
- `BaseIngestPlugin`: Abstract base with common patterns (validation, execution, error handling)
- `TableWriterIngestPlugin`: For plugins that write to core.* / analytics.* tables
- `ConfiguredIngestPlugin[TConfig]`: Auto-inject typed configuration from context
- `ToolDependentIngestPlugin`: For plugins that require external tools (pyright, scip, etc.)
- `TrackerRequiringPlugin`: For plugins that require change tracker access

Example
-------
>>> @dataclass
... class MyPlugin(ConfiguredIngestPlugin[MyStepConfig], TableWriterIngestPlugin):
...     '''Compute my ingestion.'''
...
...     output_tables = ("core.my_table",)
...     config_type = MyStepConfig
...
...     def compute(self, ctx: IngestExecutionContext) -> dict[str, int]:
...         # Pure business logic only
...         return {"core.my_table": rows_written}

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.ingestion.plugins.protocol import (
    IngestIsolationKind,
    IngestPluginMetadata,
    IngestPluginResult,
    IngestResourceHints,
    IngestSeverity,
    IngestStage,
)

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


# =============================================================================
# Validation Result
# =============================================================================


@dataclass(frozen=True)
class ValidationResult:
    """Result of plugin input validation.

    Attributes
    ----------
    valid
        Whether validation passed.
    errors
        Tuple of error messages if validation failed.
    """

    valid: bool
    errors: tuple[str, ...]

    @staticmethod
    def success() -> ValidationResult:
        """Create a successful validation result.

        Returns
        -------
        ValidationResult
            Result indicating validation passed.
        """
        return ValidationResult(valid=True, errors=())

    @staticmethod
    def failure(errors: tuple[str, ...]) -> ValidationResult:
        """Create a failed validation result.

        Parameters
        ----------
        errors
            Error messages describing validation failures.

        Returns
        -------
        ValidationResult
            Result indicating validation failed.
        """
        return ValidationResult(valid=False, errors=errors)


# =============================================================================
# Config Container
# =============================================================================


@dataclass
class ResolvedConfig[T]:
    """Container for resolved plugin configuration with type-safe access.

    Handles generic configuration types properly, avoiding the type
    inference issues that occur when storing generics directly in
    dataclass fields.

    Attributes
    ----------
    value
        The resolved configuration value, or None if not set.
    resolved
        Whether the configuration has been resolved.

    Example
    -------
    >>> container: ResolvedConfig[MyConfig] = ResolvedConfig()
    >>> container.set(my_config)
    >>> config = container.get()  # returns MyConfig
    """

    value: T | None = None
    resolved: bool = False

    def set(self, config: T) -> None:
        """Set the configuration value.

        Parameters
        ----------
        config
            The configuration to store.
        """
        self.value = config
        self.resolved = True

    def get(self, plugin_name: str = "unknown") -> T:
        """Return the configuration value.

        Parameters
        ----------
        plugin_name
            Name of plugin for error messages.

        Returns
        -------
        T
            The stored configuration.

        Raises
        ------
        ValueError
            If configuration was not resolved.
        """
        if not self.resolved or self.value is None:
            message = f"Config not resolved for {plugin_name}. Call validate_inputs first."
            raise ValueError(message)
        return self.value

    def get_or_none(self) -> T | None:
        """Return configuration value or None if not resolved.

        Returns
        -------
        T | None
            The configuration or None.
        """
        return self.value if self.resolved else None


# =============================================================================
# Base Plugin Class
# =============================================================================


@dataclass
class BaseIngestPlugin(ABC):
    """Abstract base class for all ingestion plugins.

    Provide common patterns for validation, execution, and error handling.
    Subclasses must implement `compute()` with their business logic.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier for the plugin (e.g., "ast_extract").
    plugin_description : str
        Human-readable description of what the plugin does.
    plugin_stage : IngestStage
        Processing stage for ordering (e.g., "scan", "parse", "enrich").
    plugin_version : str
        Version string for cache invalidation.
    enabled_by_default : bool
        Whether this plugin runs when no explicit list is provided.
    severity : IngestSeverity
        How failures should be handled.
    depends_on : tuple[str, ...]
        Explicit plugin dependencies by name.
    provides : tuple[str, ...]
        Capabilities this plugin provides (as strings).
    requires : tuple[str, ...]
        Capabilities this plugin requires (as strings).
    output_tables : tuple[str, ...]
        Tables this plugin writes to.
    tool_dependencies : tuple[str, ...]
        External tools required (e.g., "pyright", "scip").
    supports_incremental : bool
        Whether incremental ingestion is supported.
    resource_hints : IngestResourceHints | None
        Runtime resource hints for scheduling.
    isolation_kind : IngestIsolationKind
        Type of isolation needed for execution.

    Notes
    -----
    The `metadata` property synthesizes `IngestPluginMetadata` from class attributes,
    so subclasses don't need to construct it manually.
    """

    # Core identification (subclasses should override)
    plugin_name: ClassVar[str] = ""
    plugin_description: ClassVar[str] = ""
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "1.0.0"

    # Behavior controls
    enabled_by_default: ClassVar[bool] = True
    severity: ClassVar[IngestSeverity] = "fatal"

    # Dependencies and capabilities
    depends_on: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()

    # Output tables
    output_tables: ClassVar[tuple[str, ...]] = ()

    # Tool dependencies
    tool_dependencies: ClassVar[tuple[str, ...]] = ()

    # Incremental support
    supports_incremental: ClassVar[bool] = False

    # Resource hints
    resource_hints: ClassVar[IngestResourceHints | None] = None

    # Isolation
    isolation_kind: ClassVar[IngestIsolationKind] = "none"

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Synthesize plugin metadata from class attributes.

        Returns
        -------
        IngestPluginMetadata
            Complete metadata for this plugin.
        """
        name = self.plugin_name or self.__class__.__name__
        description = self.plugin_description or (self.__class__.__doc__ or "").split("\n")[0]

        return IngestPluginMetadata(
            name=name,
            description=description.strip(),
            stage=self.plugin_stage,
            severity=self.severity,
            enabled_by_default=self.enabled_by_default,
            depends_on=self.depends_on,
            provides=self.provides,
            requires=self.requires,
            produces_tables=self.output_tables,
            tool_dependencies=self.tool_dependencies,
            supports_incremental=self.supports_incremental,
            resource_hints=self.resource_hints,
            isolation_kind=self.isolation_kind,
            version_hash=self.plugin_version,
        )

    def validate_inputs(self, ctx: IngestExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Default implementation checks for required configs and resources.
        Override or extend in subclasses for additional validation.

        Parameters
        ----------
        ctx
            Execution context to validate against.

        Returns
        -------
        ValidationResult
            Validation outcome.
        """
        errors: list[str] = []
        errors.extend(self._validate_config_requirements(ctx))
        errors.extend(self._validate_resource_requirements(ctx))

        if errors:
            return ValidationResult.failure(tuple(errors))
        return ValidationResult.success()

    def _validate_config_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate configuration requirements.

        Override in subclasses to add config validation.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            List of validation error messages.
        """
        _ = (self, ctx)  # Unused in base implementation
        return []

    def _validate_resource_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate resource requirements (tracker, tools, etc.).

        Override in subclasses to add resource validation.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            List of validation error messages.
        """
        _ = (self, ctx)  # Unused in base implementation
        return []

    def execute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
        """Execute the plugin with standard error handling.

        This method wraps `compute()` with error handling and result
        construction. Subclasses should override `compute()` instead.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        IngestPluginResult
            Execution result.
        """
        try:
            result = self.compute(ctx)
            return self._build_success_result(result, ctx)
        except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as exc:
            log.exception("Plugin %s failed", self.metadata.name)
            return IngestPluginResult.fail(f"{self.metadata.name} failed: {exc}")

    @abstractmethod
    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute the plugin's business logic.

        Subclasses must implement this method with their computation.
        Return a mapping of table names to row counts, or None.

        Parameters
        ----------
        ctx
            Execution context with access to storage, configs, and resources.

        Returns
        -------
        Mapping[str, int] | None
            Table row counts or None if not applicable.
        """
        ...

    def _build_success_result(
        self,
        row_counts: Mapping[str, int] | None,
        ctx: IngestExecutionContext,
    ) -> IngestPluginResult:
        """Build a successful result from compute output.

        Parameters
        ----------
        row_counts
            Row counts from compute, or None.
        ctx
            Execution context.

        Returns
        -------
        IngestPluginResult
            Successful result with row counts.
        """
        _ = (self, ctx)  # Unused in base implementation
        return IngestPluginResult.ok(row_counts=row_counts or {})


# =============================================================================
# Table Writer Plugin
# =============================================================================


@dataclass
class TableWriterIngestPlugin(BaseIngestPlugin, ABC):
    """Abstract base for plugins that write to core.* / analytics.* tables.

    Automatically handles row count computation for declared output tables.
    Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    output_tables : tuple[str, ...]
        Tables this plugin writes to (e.g., ("core.ast_nodes",)).
    min_rows_per_table : Mapping[str, int]
        Optional minimum row expectations per table.
    """

    min_rows_per_table: ClassVar[Mapping[str, int]] = {}

    def compute_row_counts(self, ctx: IngestExecutionContext) -> dict[str, int]:
        """Compute row counts for output tables.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, int]
            Row counts per table.
        """
        if not self.output_tables:
            return {}

        from codeintel.ingestion.infrastructure_utilities.db_queries import (
            safe_count,
        )

        counts: dict[str, int] = {}
        for table in self.output_tables:
            count = safe_count(ctx.gateway, table)
            counts[table] = count if count is not None else 0
        return counts

    def _build_success_result(
        self,
        row_counts: Mapping[str, int] | None,
        ctx: IngestExecutionContext,
    ) -> IngestPluginResult:
        """Build result with auto-computed row counts if not provided.

        Parameters
        ----------
        row_counts
            Explicit row counts from compute, or None for auto-compute.
        ctx
            Execution context.

        Returns
        -------
        IngestPluginResult
            Success result with row counts.
        """
        if row_counts is None:
            row_counts = self.compute_row_counts(ctx)
        return IngestPluginResult.ok(row_counts=dict(row_counts))


# =============================================================================
# Tool Dependent Plugin
# =============================================================================


@dataclass
class ToolDependentIngestPlugin(BaseIngestPlugin, ABC):
    """Abstract base for plugins that require external tools.

    Validate tool availability before execution.
    Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    tool_dependencies : tuple[str, ...]
        External tools required (e.g., ("pyright", "scip")).
    tool_required : bool
        Whether missing tools should fail validation (True) or skip execution (False).
    """

    tool_required: ClassVar[bool] = False

    def _validate_resource_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate tool availability.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors for missing required tools.
        """
        errors = super()._validate_resource_requirements(ctx)

        if not self.tool_required:
            return errors

        # Check if tools provider is available
        if not ctx.has_resource_by_name("ToolsProvider"):
            for tool in self.tool_dependencies:
                errors.append(f"Tool '{tool}' is required for {self.metadata.name}")

        return errors


# =============================================================================
# Tracker Requiring Plugin
# =============================================================================


@dataclass
class TrackerRequiringPlugin(BaseIngestPlugin, ABC):
    """Abstract base for plugins that require change tracker access.

    Validate tracker availability via resources.
    Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    tracker_required : bool
        Whether tracker is strictly required (True) or optional (False).
    """

    tracker_required: ClassVar[bool] = True

    def _validate_resource_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate tracker availability via resources.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors = super()._validate_resource_requirements(ctx)
        if not self.tracker_required:
            return errors

        if not ctx.has_resource_by_name("TrackerProvider"):
            errors.append(f"TrackerProvider is required for {self.metadata.name}")
        return errors

    def get_tracker(self, ctx: IngestExecutionContext) -> ChangeTracker:
        """Get the change tracker from context via TrackerProvider.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ChangeTracker
            The change tracker.
        """
        _ = self  # Required by interface, accessed via ctx
        from codeintel.ingestion.resources.tracker import TrackerProvider

        provider = cast("TrackerProvider", ctx.require_by_name("TrackerProvider"))
        return provider.get()

    def get_tracker_or_none(self, ctx: IngestExecutionContext) -> ChangeTracker | None:
        """Get the change tracker or None if not available.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ChangeTracker | None
            The change tracker or None.
        """
        _ = self  # Required by interface, accessed via ctx
        if not ctx.has_resource_by_name("TrackerProvider"):
            return None

        from codeintel.ingestion.resources.tracker import TrackerProvider

        provider = cast("TrackerProvider", ctx.require_by_name("TrackerProvider"))
        return provider.get()


# =============================================================================
# Config Bound Plugin
# =============================================================================


@dataclass
class ConfiguredIngestPlugin[TConfig](BaseIngestPlugin, ABC):
    """Abstract base for plugins that require typed configuration.

    Automatically validate config availability and provide typed access.
    Subclasses must implement `compute()`.

    The generic type parameter `TConfig` is preserved through direct field typing.

    Class Attributes
    ----------------
    config_type : type[TConfig]
        The configuration type this plugin requires.
    config_required : bool
        Whether the config is required (True) or optional (False).

    Properties
    ----------
    config : TConfig
        The resolved configuration (after validation passes).
    """

    config_type: ClassVar[type[object]] = object
    config_required: ClassVar[bool] = True

    # Config state - directly typed with TConfig for proper type inference
    _config: TConfig | None = field(default=None, init=False, repr=False)
    _config_resolved: bool = field(default=False, init=False, repr=False)

    @property
    def config(self) -> TConfig:
        """Return the resolved configuration.

        Returns
        -------
        TConfig
            The typed configuration.

        Raises
        ------
        ValueError
            If config was not resolved (validate_inputs not called).
        """
        if not self._config_resolved or self._config is None:
            message = f"Config not resolved for {self.metadata.name}. Call validate_inputs first."
            raise ValueError(message)
        return self._config

    def _validate_config_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate config is available and resolve it.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors = super()._validate_config_requirements(ctx)

        if self.config_required:
            if not ctx.has_config(self.config_type):
                errors.append(f"{self.config_type.__name__} is required for {self.metadata.name}")
            else:
                # Cast to TConfig - config_type ClassVar can't use type parameter,
                # but ctx.get_config returns the config instance matching config_type
                self._config = cast("TConfig", ctx.get_config(self.config_type))
                self._config_resolved = True
        else:
            config = ctx.get_optional_config(self.config_type)
            if config is not None:
                self._config = cast("TConfig", config)
                self._config_resolved = True

        return errors

    def get_config_or_none(self) -> TConfig | None:
        """Return the resolved config or None if not required and missing.

        Returns
        -------
        TConfig | None
            The config or None.
        """
        return self._config if self._config_resolved else None


# =============================================================================
# Composite Base Classes (common combinations)
# =============================================================================


@dataclass
class ConfiguredTableWriterPlugin[TConfig](
    ConfiguredIngestPlugin[TConfig],
    TableWriterIngestPlugin,
    ABC,
):
    """Abstract base for plugins that write tables and require typed configuration.

    Combine config binding with table writing. Use when you need both
    typed configuration and automatic row count handling.
    Subclasses must implement `compute()`.
    """

    def _validate_config_requirements(
        self,
        ctx: IngestExecutionContext,
    ) -> list[str]:
        """Validate config requirements from ConfiguredIngestPlugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        # Use super() to call the ConfiguredIngestPlugin implementation
        return super()._validate_config_requirements(ctx)


__all__ = [
    "BaseIngestPlugin",
    "ConfiguredIngestPlugin",
    "ConfiguredTableWriterPlugin",
    "ResolvedConfig",
    "TableWriterIngestPlugin",
    "ToolDependentIngestPlugin",
    "TrackerRequiringPlugin",
    "ValidationResult",
]
