"""Base plugin classes for analytics plugins.

This module provides a hierarchy of base classes that plugins can inherit from
to minimize boilerplate while retaining full flexibility. The design emphasizes
composition over deep inheritance through trait mixins.

Architecture
------------
- `BasePlugin`: Abstract base with common patterns (validation, execution, error handling)
- `TableWriterPlugin`: For plugins that write to analytics.* tables
- `ConfigBoundPlugin[TConfig]`: Auto-inject typed configuration from context
- `CatalogRequiringPlugin`: Plugins that require function catalog access
- `GraphMetricsPlugin`: Common base for graph-based metric computations

Example
-------
>>> @dataclass
... class MyPlugin(ConfigBoundPlugin[MyStepConfig], TableWriterPlugin):
...     '''Compute my analytics.'''
...
...     output_tables = ("analytics.my_table",)
...     config_type = MyStepConfig
...
...     def compute(self, ctx: PluginExecutionContext) -> dict[str, int]:
...         # Pure business logic only
...         return {"analytics.my_table": rows_written}
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.core.contracts import OutputContractSpec
from codeintel.analytics.core.plugin_protocol import (
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.storage.db_helpers import safe_row_counts

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.analytics.resources.catalog import CatalogProvider
    from codeintel.analytics.resources.graphs import GraphProvider
    from codeintel.graphs.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


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
class BasePlugin(ABC):
    """Abstract base class for all analytics plugins.

    Provides common patterns for validation, execution, and error handling.
    Subclasses must implement `compute()` with their business logic.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier for the plugin (e.g., "functions.metrics").
    plugin_description : str
        Human-readable description of what the plugin computes.
    plugin_stage : PluginStage
        Processing stage for ordering (e.g., "function", "graph").
    plugin_version : str
        Version string for cache invalidation.
    enabled_by_default : bool
        Whether this plugin runs when no explicit list is provided.
    severity : PluginSeverity
        How failures should be handled.
    depends_on : tuple[str, ...]
        Explicit plugin dependencies by name.
    provides : tuple[str, ...]
        Capabilities this plugin provides (as strings).
    requires : tuple[str, ...]
        Capabilities this plugin requires (as strings).
    tags : tuple[str, ...]
        Free-form tags for categorization.
    resource_hints : PluginResourceHints | None
        Runtime resource hints for scheduling.
    requires_isolation : bool
        Whether process/thread isolation is needed.
    isolation_kind : str | None
        Type of isolation ("process" or "thread").

    Notes
    -----
    The `metadata` property synthesizes `PluginMetadata` from class attributes,
    so subclasses don't need to construct it manually.
    """

    # Core identification (subclasses should override)
    plugin_name: ClassVar[str] = ""
    plugin_description: ClassVar[str] = ""
    plugin_kind: ClassVar[PluginKind] = "analytics"
    plugin_stage: ClassVar[PluginStage] = "other"
    plugin_version: ClassVar[str] = "1.0.0"

    # Behavior controls
    enabled_by_default: ClassVar[bool] = True
    severity: ClassVar[PluginSeverity] = "fatal"

    # Dependencies and capabilities
    depends_on: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()

    # Categorization
    tags: ClassVar[tuple[str, ...]] = ()

    # Resource hints
    resource_hints: ClassVar[PluginResourceHints | None] = None

    # Isolation
    requires_isolation: ClassVar[bool] = False
    isolation_kind: ClassVar[PluginIsolation] = "none"

    @property
    def metadata(self) -> PluginMetadata:
        """Synthesize plugin metadata from class attributes.

        Returns
        -------
        PluginMetadata
            Complete metadata for this plugin.
        """
        name = self.plugin_name or self.__class__.__name__
        description = self.plugin_description or (self.__class__.__doc__ or "").split("\n")[0]

        return PluginMetadata(
            name=name,
            description=description.strip(),
            kind=self.plugin_kind,
            stage=self.plugin_stage,
            version=self.plugin_version,
            enabled_by_default=self.enabled_by_default,
            severity=self.severity,
            inputs=self.build_input_specs(),
            outputs=self._build_output_specs(),
            depends_on=self.depends_on,
            provides=self.provides,
            requires=self.requires,
            resource_hints=self.resource_hints,
            requires_isolation=self.requires_isolation,
            isolation_kind=self.isolation_kind,
            tags=self.tags,
        )

    @classmethod
    def build_input_specs(cls) -> tuple[PluginInputSpec, ...]:
        """Build input specifications from class attributes.

        Override in subclasses for custom input handling.

        Returns
        -------
        tuple[PluginInputSpec, ...]
            Input specifications for this plugin.
        """
        return ()

    @classmethod
    def _build_output_specs(cls) -> tuple[PluginOutputSpec, ...]:
        """Build output specifications from class attributes.

        Override in subclasses for custom output handling.

        Returns
        -------
        tuple[PluginOutputSpec, ...]
            Output specifications for this plugin.
        """
        return ()

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
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
        errors.extend(self.validate_config_requirements(ctx))
        errors.extend(self.validate_resource_requirements(ctx))

        if errors:
            return ValidationResult.failure(tuple(errors))
        return ValidationResult.success()

    @staticmethod
    def validate_config_requirements(ctx: PluginExecutionContext) -> list[str]:
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
        _ = ctx  # Base implementation doesn't use context
        return []

    @staticmethod
    def validate_resource_requirements(ctx: PluginExecutionContext) -> list[str]:
        """Validate resource requirements (catalog, runtime, etc.).

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
        _ = ctx  # Base implementation doesn't use context
        return []

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin with standard error handling.

        This method wraps `compute()` with error handling and result
        construction. Subclasses should override `compute()` instead.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Execution result.
        """
        try:
            result = self.compute(ctx)
            return self._build_success_result(result, ctx)
        except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as exc:
            log.exception("Plugin %s failed", self.metadata.name)
            return PluginResult.fail(f"{self.metadata.name} failed: {exc}")

    @abstractmethod
    def compute(self, ctx: PluginExecutionContext) -> Mapping[str, int] | None:
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
        ctx: PluginExecutionContext,
    ) -> PluginResult:
        """Build a successful result from compute output.

        Parameters
        ----------
        row_counts
            Row counts from compute, or None.
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Successful result with row counts.
        """
        meta = {
            "plugin_name": self.metadata.name,
            "repo": ctx.repo,
            "commit": ctx.commit,
        }
        if ctx.run_id is not None:
            meta["run_id"] = ctx.run_id

        return PluginResult.ok(
            row_counts=row_counts or {},
            meta=meta,
        )


# =============================================================================
# Table Writer Plugin
# =============================================================================


@dataclass
class TableWriterPlugin(BasePlugin, ABC):
    """Abstract base for plugins that write to analytics.* tables.

    Automatically handles row count computation for declared output tables.
    Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    output_tables : tuple[str, ...]
        Tables this plugin writes to (e.g., ("analytics.function_metrics",)).
    min_rows_per_table : Mapping[str, int]
        Optional minimum row expectations per table.
    required_columns_per_table : Mapping[str, tuple[str, ...]]
        Optional required columns per table.
    """

    output_tables: ClassVar[tuple[str, ...]] = ()
    min_rows_per_table: ClassVar[Mapping[str, int]] = {}
    required_columns_per_table: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @property
    def output_contracts(self) -> tuple[OutputContractSpec, ...]:
        """Build output contracts from declared tables.

        Returns
        -------
        tuple[OutputContractSpec, ...]
            Contracts for output validation.
        """
        return tuple(
            OutputContractSpec(
                table=table,
                min_rows=self.min_rows_per_table.get(table),
                required_columns=self.required_columns_per_table.get(table, ()),
                description=f"Output from {self.metadata.name}",
            )
            for table in self.output_tables
        )

    def _build_output_specs(self) -> tuple[PluginOutputSpec, ...]:
        """Build output specs from output_tables.

        Returns
        -------
        tuple[PluginOutputSpec, ...]
            Output specifications for metadata.
        """
        return tuple(
            PluginOutputSpec(
                name=table.split(".")[-1],
                tables=(table,),
                min_rows=self.min_rows_per_table.get(table),
                required_columns=self.required_columns_per_table.get(table, ()),
            )
            for table in self.output_tables
        )

    def compute_row_counts(self, ctx: PluginExecutionContext) -> dict[str, int]:
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
        connection = getattr(ctx.gateway, "con", None)
        if connection is None:
            return {}
        counts = safe_row_counts(
            connection,
            repo=ctx.repo,
            commit=ctx.commit,
            tables=self.output_tables,
        )
        return counts or {}

    def _build_success_result(
        self,
        row_counts: Mapping[str, int] | None,
        ctx: PluginExecutionContext,
    ) -> PluginResult:
        """Build result with auto-computed row counts if not provided.

        Parameters
        ----------
        row_counts
            Explicit row counts from compute, or None for auto-compute.
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Success result with row counts.
        """
        if row_counts is None:
            row_counts = self.compute_row_counts(ctx)
        return PluginResult.ok(
            row_counts=dict(row_counts),
            meta={"plugin_name": self.metadata.name},
        )


# =============================================================================
# Config Bound Plugin
# =============================================================================


@dataclass
class ConfigBoundPlugin[TConfig](BasePlugin, ABC):
    """Abstract base for plugins that require typed configuration.

    Automatically validates config availability and provides typed access.
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

    def build_input_specs(self) -> tuple[PluginInputSpec, ...]:
        """Build input specs including config requirement.

        Returns
        -------
        tuple[PluginInputSpec, ...]
            Input specifications with config.
        """
        base_inputs = super().build_input_specs()
        config_input = PluginInputSpec(
            name="config",
            type_ref=self.config_type.__name__,
            required=self.config_required,
            source="config",
        )
        return (*base_inputs, config_input)

    def validate_config_requirements(self, ctx: PluginExecutionContext) -> list[str]:
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
        errors = super().validate_config_requirements(ctx)

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
# Catalog Requiring Plugin
# =============================================================================


@dataclass
class CatalogRequiringPlugin(BasePlugin, ABC):
    """Abstract base for plugins that require function catalog access.

    Validates catalog availability via ResourceRegistry.
    Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    catalog_required : bool
        Whether catalog is strictly required (True) or optional (False).

    Properties
    ----------
    requires_catalog : bool
        Returns True, indicating catalog requirement.
    """

    catalog_required: ClassVar[bool] = True

    @property
    def requires_catalog(self) -> bool:
        """Return whether catalog is required.

        Returns
        -------
        bool
            True since this is a catalog-requiring plugin.
        """
        return self.catalog_required

    def validate_resource_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate catalog availability via ResourceRegistry.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors = super().validate_resource_requirements(ctx)
        if not self.catalog_required:
            return errors

        if not ctx.has_resource_by_name("CatalogProvider"):
            errors.append(f"CatalogProvider is required for {self.metadata.name}")
        return errors

    @staticmethod
    def get_catalog(ctx: PluginExecutionContext) -> FunctionCatalogProvider:
        """Get the catalog from context via CatalogProvider.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        FunctionCatalogProvider
            The function catalog.

        Notes
        -----
        Raises `ResourceNotFoundError` if CatalogProvider is not available.
        """
        provider = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
        return provider.get()


# =============================================================================
# Graph Runtime Requiring Plugin
# =============================================================================


@dataclass
class GraphRuntimeRequiringPlugin(BasePlugin, ABC):
    """Abstract base for plugins that require graph runtime access.

    Provides access to graph loading and engine capabilities via
    GraphProvider from ResourceRegistry. Subclasses must implement `compute()`.

    Class Attributes
    ----------------
    graph_runtime_required : bool
        Whether graph runtime is strictly required.
    """

    graph_runtime_required: ClassVar[bool] = True

    def validate_resource_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate graph runtime availability via ResourceRegistry.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors = super().validate_resource_requirements(ctx)
        if not self.graph_runtime_required:
            return errors

        if not ctx.has_resource_by_name("GraphProvider"):
            errors.append(f"GraphProvider is required for {self.metadata.name}")
        return errors

    @staticmethod
    def get_graph_runtime(ctx: PluginExecutionContext) -> GraphRuntime:
        """Get the graph runtime via GraphProvider.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        GraphRuntime
            The graph runtime.

        Raises
        ------
        ValueError
            If GraphProvider has no runtime available.
        """
        provider = cast("GraphProvider", ctx.require_by_name("GraphProvider"))
        runtime = provider.runtime
        if runtime is None:
            message = "GraphProvider has no runtime available"
            raise ValueError(message)
        return runtime


# =============================================================================
# Graph Metrics Plugin (combines common requirements)
# =============================================================================


@dataclass
class GraphMetricsPlugin(
    TableWriterPlugin,
    GraphRuntimeRequiringPlugin,
    CatalogRequiringPlugin,
    ABC,
):
    """Abstract base for graph-based metric computation plugins.

    Combines table writing with graph runtime and catalog access.
    Most graph metric plugins should inherit from this class.
    Subclasses must implement `compute()`.

    This class provides:
    - Automatic row count computation for output tables
    - Graph runtime access for loading graphs
    - Catalog access for symbol resolution
    - Automatic validation of all requirements
    """

    plugin_stage: ClassVar[PluginStage] = "graph"

    def validate_resource_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate all resource requirements.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Combined validation errors from all bases.
        """
        errors: list[str] = []
        # Call each parent's validation - accessing protected methods is intentional
        # TableWriterPlugin inherits static method from BasePlugin, so no self
        errors.extend(TableWriterPlugin.validate_resource_requirements(ctx))
        errors.extend(GraphRuntimeRequiringPlugin.validate_resource_requirements(self, ctx))
        errors.extend(CatalogRequiringPlugin.validate_resource_requirements(self, ctx))
        return errors


# =============================================================================
# Composite Base Classes (common combinations)
# =============================================================================


@dataclass
class ConfiguredTableWriterPlugin[TConfig](ConfigBoundPlugin[TConfig], TableWriterPlugin, ABC):
    """Abstract base for plugins that write tables and require typed configuration.

    Combines config binding with table writing. Use when you need both
    typed configuration and automatic row count handling.
    Subclasses must implement `compute()`.
    """

    def validate_config_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate config requirements from ConfigBoundPlugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        return ConfigBoundPlugin.validate_config_requirements(self, ctx)

    def build_input_specs(self) -> tuple[PluginInputSpec, ...]:
        """Build input specs combining both bases.

        Returns
        -------
        tuple[PluginInputSpec, ...]
            Combined input specifications.
        """
        return ConfigBoundPlugin.build_input_specs(self)


@dataclass
class ConfiguredGraphMetricsPlugin[TConfig](ConfigBoundPlugin[TConfig], GraphMetricsPlugin, ABC):
    """Abstract base for graph metrics plugins with typed configuration.

    The most common base for graph metric plugins that need configuration.
    Provides config binding, table writing, graph runtime, and catalog access.
    Subclasses must implement `compute()`.
    """

    def validate_config_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate config from ConfigBoundPlugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        return ConfigBoundPlugin.validate_config_requirements(self, ctx)

    def validate_resource_requirements(self, ctx: PluginExecutionContext) -> list[str]:
        """Validate all resources from GraphMetricsPlugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[str]
            Validation errors.
        """
        return GraphMetricsPlugin.validate_resource_requirements(self, ctx)

    def build_input_specs(self) -> tuple[PluginInputSpec, ...]:
        """Build input specs from ConfigBoundPlugin.

        Returns
        -------
        tuple[PluginInputSpec, ...]
            Input specifications.
        """
        return ConfigBoundPlugin.build_input_specs(self)


# =============================================================================
# Helper Functions
# =============================================================================


def capabilities_from_tables(tables: Sequence[str]) -> tuple[str, ...]:
    """Generate capability names from table names.

    Parameters
    ----------
    tables
        Table names (e.g., "analytics.function_metrics").

    Returns
    -------
    tuple[str, ...]
        Capability names (same as table names).
    """
    return tuple(tables)


__all__ = [
    "BasePlugin",
    "CatalogRequiringPlugin",
    "ConfigBoundPlugin",
    "ConfiguredGraphMetricsPlugin",
    "ConfiguredTableWriterPlugin",
    "GraphMetricsPlugin",
    "GraphRuntimeRequiringPlugin",
    "TableWriterPlugin",
    "capabilities_from_tables",
]
