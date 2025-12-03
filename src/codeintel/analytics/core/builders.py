"""Fluent builders for plugin metadata and specifications.

This module provides a builder pattern for constructing plugin metadata
with a clean, chainable API that reduces boilerplate.

Example
-------
>>> spec = (
...     PluginSpec.create("functions.metrics")
...     .description("Compute function metrics and complexity")
...     .stage("function")
...     .version("2.0.0")
...     .input(FunctionAnalyticsStepConfig, required=True)
...     .output("analytics.function_metrics", min_rows=1)
...     .provides("analytics.function_metrics")
...     .requires("core.goids")
...     .depends_on("core.catalog")
...     .tag("functions", "metrics")
...     .build()
... )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from codeintel.analytics.core.plugin_protocol import (
    PluginCapability,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
)


@dataclass
class PluginMetaSection:
    """Metadata-focused fields for plugin specification."""

    name: str
    description: str = ""
    stage: PluginStage = "other"
    version: str = "1.0.0"
    enabled_by_default: bool = True
    severity: PluginSeverity = "fatal"
    tags: list[str] = field(default_factory=list)


@dataclass
class PluginContractsSection:
    """Contracts and dependency fields for plugin specification."""

    inputs: list[PluginInputSpec] = field(default_factory=list)
    outputs: list[PluginOutputSpec] = field(default_factory=list)
    capabilities_provided: list[PluginCapability] = field(default_factory=list)
    capabilities_required: list[PluginCapability] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PluginRuntimeSection:
    """Runtime/resource fields for plugin specification."""

    resource_hints: PluginResourceHints | None = None
    requires_isolation: bool = False
    isolation_kind: Literal["process", "thread"] | None = None


class PluginSpecBuilder:
    """Fluent builder for constructing PluginMetadata.

    Provides a clean API for building plugin metadata incrementally.
    All setter methods return self for chaining.
    """

    def __init__(self, name: str) -> None:
        self._meta = PluginMetaSection(name=name)
        self._contracts = PluginContractsSection()
        self._runtime = PluginRuntimeSection()

    def description(self, desc: str) -> PluginSpecBuilder:
        """Set the plugin description.

        Parameters
        ----------
        desc
            Human-readable description of what the plugin does.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.description = desc
        return self

    def stage(self, stage: PluginStage) -> PluginSpecBuilder:
        """Set the processing stage.

        Parameters
        ----------
        stage
            Stage for ordering (e.g., "function", "graph", "test").

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.stage = stage
        return self

    def version(self, ver: str) -> PluginSpecBuilder:
        """Set the plugin version.

        Parameters
        ----------
        ver
            Version string for cache invalidation.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.version = ver
        return self

    def enabled_by_default(self, *, enabled: bool = True) -> PluginSpecBuilder:
        """Set whether plugin runs by default.

        Parameters
        ----------
        enabled
            Whether to run when no explicit list is provided.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.enabled_by_default = enabled
        return self

    def disabled_by_default(self) -> PluginSpecBuilder:
        """Mark plugin as disabled by default.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.enabled_by_default = False
        return self

    def severity(self, sev: PluginSeverity) -> PluginSpecBuilder:
        """Set failure handling severity.

        Parameters
        ----------
        sev
            How failures should be handled ("fatal", "soft_fail", "skip_on_error").

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.severity = sev
        return self

    def fatal(self) -> PluginSpecBuilder:
        """Mark failures as fatal.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.severity = "fatal"
        return self

    def soft_fail(self) -> PluginSpecBuilder:
        """Mark failures as soft (continue execution).

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.severity = "soft_fail"
        return self

    def skip_on_error(self) -> PluginSpecBuilder:
        """Skip plugin on error without failing.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.severity = "skip_on_error"
        return self

    def input(
        self,
        config_type: type[object],
        *,
        name: str | None = None,
        required: bool = True,
        source: Literal["config", "runtime", "prior_plugin"] = "config",
    ) -> PluginSpecBuilder:
        """Add an input requirement.

        Parameters
        ----------
        config_type
            The configuration type required.
        name
            Optional name override (defaults to type name).
        required
            Whether this input is required.
        source
            Where the input comes from.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.inputs.append(
            PluginInputSpec(
                name=name or config_type.__name__,
                type_ref=config_type.__name__,
                required=required,
                source=source,
            )
        )
        return self

    def input_spec(self, spec: PluginInputSpec) -> PluginSpecBuilder:
        """Add a raw input specification.

        Parameters
        ----------
        spec
            Pre-built input spec.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.inputs.append(spec)
        return self

    def output(
        self,
        table: str,
        *,
        name: str | None = None,
        min_rows: int | None = None,
        required_columns: Sequence[str] = (),
    ) -> PluginSpecBuilder:
        """Add an output table.

        Parameters
        ----------
        table
            Full table name (e.g., "analytics.function_metrics").
        name
            Logical name (defaults to table name without schema).
        min_rows
            Minimum expected rows for validation.
        required_columns
            Columns that must be present.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        logical_name = name or table.rsplit(".", maxsplit=1)[-1]
        self._contracts.outputs.append(
            PluginOutputSpec(
                name=logical_name,
                tables=(table,),
                min_rows=min_rows,
                required_columns=tuple(required_columns),
            )
        )
        return self

    def outputs(self, *tables: str) -> PluginSpecBuilder:
        """Add multiple output tables.

        Parameters
        ----------
        tables
            Table names to add.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        for table in tables:
            self.output(table)
        return self

    def output_spec(self, spec: PluginOutputSpec) -> PluginSpecBuilder:
        """Add a raw output specification.

        Parameters
        ----------
        spec
            Pre-built output spec.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.outputs.append(spec)
        return self

    def provides(self, *capabilities: str) -> PluginSpecBuilder:
        """Declare capabilities this plugin provides.

        Parameters
        ----------
        capabilities
            Capability names (e.g., "analytics.function_metrics").

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        for cap in capabilities:
            self._contracts.capabilities_provided.append(PluginCapability(name=cap))
        return self

    def provides_capability(self, capability: PluginCapability) -> PluginSpecBuilder:
        """Add a raw capability.

        Parameters
        ----------
        capability
            Pre-built capability.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.capabilities_provided.append(capability)
        return self

    def requires(self, *capabilities: str) -> PluginSpecBuilder:
        """Declare capabilities this plugin requires.

        Parameters
        ----------
        capabilities
            Capability names (e.g., "core.goids").

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        for cap in capabilities:
            self._contracts.capabilities_required.append(PluginCapability(name=cap))
        return self

    def requires_capability(self, capability: PluginCapability) -> PluginSpecBuilder:
        """Add a raw required capability.

        Parameters
        ----------
        capability
            Pre-built capability.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.capabilities_required.append(capability)
        return self

    def depends_on(self, *plugins: str) -> PluginSpecBuilder:
        """Add explicit plugin dependencies.

        Parameters
        ----------
        plugins
            Plugin names this plugin depends on.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._contracts.depends_on.extend(plugins)
        return self

    def resources(
        self,
        *,
        max_runtime_ms: int | None = None,
        max_memory_mb: int | None = None,
        requires_gpu: bool = False,
        priority: int = 0,
    ) -> PluginSpecBuilder:
        """Set resource hints.

        Parameters
        ----------
        max_runtime_ms
            Maximum expected runtime in milliseconds.
        max_memory_mb
            Maximum expected memory usage in MB.
        requires_gpu
            Whether GPU acceleration is beneficial.
        priority
            Scheduling priority (higher = more important).

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._runtime.resource_hints = PluginResourceHints(
            max_runtime_ms=max_runtime_ms,
            max_memory_mb=max_memory_mb,
            requires_gpu=requires_gpu,
            priority=priority,
        )
        return self

    def isolate(
        self,
        kind: Literal["process", "thread"] = "process",
    ) -> PluginSpecBuilder:
        """Require isolation for this plugin.

        Parameters
        ----------
        kind
            Type of isolation ("process" or "thread").

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._runtime.requires_isolation = True
        self._runtime.isolation_kind = kind
        return self

    def tag(self, *tags: str) -> PluginSpecBuilder:
        """Add tags for categorization.

        Parameters
        ----------
        tags
            Free-form tags.

        Returns
        -------
        PluginSpecBuilder
            Self for chaining.
        """
        self._meta.tags.extend(tags)
        return self

    def build(self) -> PluginMetadata:
        """Build the final PluginMetadata.

        Returns
        -------
        PluginMetadata
            Complete plugin metadata.
        """
        return PluginMetadata(
            name=self._meta.name,
            description=self._meta.description,
            stage=self._meta.stage,
            version=self._meta.version,
            enabled_by_default=self._meta.enabled_by_default,
            severity=self._meta.severity,
            inputs=tuple(self._contracts.inputs),
            outputs=tuple(self._contracts.outputs),
            capabilities_provided=tuple(self._contracts.capabilities_provided),
            capabilities_required=tuple(self._contracts.capabilities_required),
            depends_on=tuple(self._contracts.depends_on),
            resource_hints=self._runtime.resource_hints,
            requires_isolation=self._runtime.requires_isolation,
            isolation_kind=self._runtime.isolation_kind,
            tags=tuple(self._meta.tags),
        )


class PluginSpec:
    """Factory for creating PluginSpecBuilder instances.

    Example
    -------
    >>> spec = PluginSpec.create("my.plugin").description("My plugin").build()
    """

    @staticmethod
    def create(name: str) -> PluginSpecBuilder:
        """Start building a new plugin specification.

        Parameters
        ----------
        name
            Plugin name (e.g., "functions.metrics").

        Returns
        -------
        PluginSpecBuilder
            A new builder instance.
        """
        return PluginSpecBuilder(name)


# =============================================================================
# Resource Hints Builder
# =============================================================================


@dataclass
class ResourceHintsBuilder:
    """Builder for PluginResourceHints.

    Provides a fluent API for building resource hints.
    """

    _max_runtime_ms: int | None = None
    _max_memory_mb: int | None = None
    _requires_gpu: bool = False
    _priority: int = 0

    def max_runtime(self, ms: int) -> ResourceHintsBuilder:
        """Set maximum runtime.

        Parameters
        ----------
        ms
            Maximum runtime in milliseconds.

        Returns
        -------
        ResourceHintsBuilder
            Self for chaining.
        """
        self._max_runtime_ms = ms
        return self

    def max_memory(self, mb: int) -> ResourceHintsBuilder:
        """Set maximum memory.

        Parameters
        ----------
        mb
            Maximum memory in megabytes.

        Returns
        -------
        ResourceHintsBuilder
            Self for chaining.
        """
        self._max_memory_mb = mb
        return self

    def gpu(self, *, required: bool = True) -> ResourceHintsBuilder:
        """Set GPU requirement.

        Parameters
        ----------
        required
            Whether GPU is beneficial.

        Returns
        -------
        ResourceHintsBuilder
            Self for chaining.
        """
        self._requires_gpu = required
        return self

    def priority(self, p: int) -> ResourceHintsBuilder:
        """Set scheduling priority.

        Parameters
        ----------
        p
            Priority value (higher = more important).

        Returns
        -------
        ResourceHintsBuilder
            Self for chaining.
        """
        self._priority = p
        return self

    def build(self) -> PluginResourceHints:
        """Build the resource hints.

        Returns
        -------
        PluginResourceHints
            Complete resource hints.
        """
        return PluginResourceHints(
            max_runtime_ms=self._max_runtime_ms,
            max_memory_mb=self._max_memory_mb,
            requires_gpu=self._requires_gpu,
            priority=self._priority,
        )


class ResourceHints:
    """Factory for creating ResourceHintsBuilder instances."""

    @staticmethod
    def create() -> ResourceHintsBuilder:
        """Start building resource hints.

        Returns
        -------
        ResourceHintsBuilder
            A new builder instance.
        """
        return ResourceHintsBuilder()

    @staticmethod
    def default() -> PluginResourceHints:
        """Return default resource hints.

        Returns
        -------
        PluginResourceHints
            Default hints with no constraints.
        """
        return PluginResourceHints()

    @staticmethod
    def quick(
        *,
        max_runtime_ms: int = 30_000,
        priority: int = 0,
    ) -> PluginResourceHints:
        """Create hints for a quick plugin.

        Parameters
        ----------
        max_runtime_ms
            Maximum runtime (default 30s).
        priority
            Scheduling priority.

        Returns
        -------
        PluginResourceHints
            Hints for a quick plugin.
        """
        return PluginResourceHints(
            max_runtime_ms=max_runtime_ms,
            priority=priority,
        )

    @staticmethod
    def heavy(
        *,
        max_runtime_ms: int = 300_000,
        max_memory_mb: int = 4096,
        priority: int = 50,
    ) -> PluginResourceHints:
        """Create hints for a heavy plugin.

        Parameters
        ----------
        max_runtime_ms
            Maximum runtime (default 5min).
        max_memory_mb
            Maximum memory (default 4GB).
        priority
            Scheduling priority.

        Returns
        -------
        PluginResourceHints
            Hints for a heavy plugin.
        """
        return PluginResourceHints(
            max_runtime_ms=max_runtime_ms,
            max_memory_mb=max_memory_mb,
            priority=priority,
        )


# =============================================================================
# Output Contract Builder
# =============================================================================


@dataclass
class OutputSpecBuilder:
    """Builder for PluginOutputSpec.

    Provides a fluent API for building output specifications.
    """

    _name: str
    _tables: list[str] = field(default_factory=list)
    _artifact_type: str | None = None
    _min_rows: int | None = None
    _required_columns: list[str] = field(default_factory=list)

    def table(self, name: str) -> OutputSpecBuilder:
        """Add an output table.

        Parameters
        ----------
        name
            Full table name.

        Returns
        -------
        OutputSpecBuilder
            Self for chaining.
        """
        self._tables.append(name)
        return self

    def tables(self, *names: str) -> OutputSpecBuilder:
        """Add multiple output tables.

        Parameters
        ----------
        names
            Table names.

        Returns
        -------
        OutputSpecBuilder
            Self for chaining.
        """
        self._tables.extend(names)
        return self

    def artifact(self, artifact_type: str) -> OutputSpecBuilder:
        """Set artifact type for non-table outputs.

        Parameters
        ----------
        artifact_type
            Artifact type identifier.

        Returns
        -------
        OutputSpecBuilder
            Self for chaining.
        """
        self._artifact_type = artifact_type
        return self

    def min_rows(self, count: int) -> OutputSpecBuilder:
        """Set minimum expected rows.

        Parameters
        ----------
        count
            Minimum row count.

        Returns
        -------
        OutputSpecBuilder
            Self for chaining.
        """
        self._min_rows = count
        return self

    def columns(self, *names: str) -> OutputSpecBuilder:
        """Add required columns.

        Parameters
        ----------
        names
            Column names.

        Returns
        -------
        OutputSpecBuilder
            Self for chaining.
        """
        self._required_columns.extend(names)
        return self

    def build(self) -> PluginOutputSpec:
        """Build the output specification.

        Returns
        -------
        PluginOutputSpec
            Complete output spec.
        """
        return PluginOutputSpec(
            name=self._name,
            tables=tuple(self._tables),
            artifact_type=self._artifact_type,
            min_rows=self._min_rows,
            required_columns=tuple(self._required_columns),
        )


class OutputSpec:
    """Factory for creating OutputSpecBuilder instances."""

    @staticmethod
    def create(name: str) -> OutputSpecBuilder:
        """Start building an output specification.

        Parameters
        ----------
        name
            Logical name for the output.

        Returns
        -------
        OutputSpecBuilder
            A new builder instance.
        """
        return OutputSpecBuilder(_name=name)

    @staticmethod
    def table(
        table: str,
        *,
        min_rows: int | None = None,
        columns: Sequence[str] = (),
    ) -> PluginOutputSpec:
        """Create a simple table output spec.

        Parameters
        ----------
        table
            Full table name.
        min_rows
            Minimum expected rows.
        columns
            Required columns.

        Returns
        -------
        PluginOutputSpec
            Output specification.
        """
        return PluginOutputSpec(
            name=table.rsplit(".", maxsplit=1)[-1],
            tables=(table,),
            min_rows=min_rows,
            required_columns=tuple(columns),
        )


__all__ = [
    "PluginContractsSection",
    "PluginMetaSection",
    "PluginRuntimeSection",
    "OutputSpec",
    "OutputSpecBuilder",
    "PluginSpec",
    "PluginSpecBuilder",
    "ResourceHints",
    "ResourceHintsBuilder",
]
