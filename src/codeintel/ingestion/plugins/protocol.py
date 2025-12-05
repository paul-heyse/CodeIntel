"""Core protocol and types for ingestion plugins.

This module defines the protocol and types for ingestion plugins, providing
a modernized interface aligned with the analytics graph plugin architecture
while preserving ingestion-specific functionality.

Migration Note
--------------
The following types are deprecated and will be removed in a future version:

- ``IngestRuntimeScratch`` -> Use ``PluginScratch`` from ``codeintel.core.plugins``
- ``IngestResourceHints`` -> Use ``PluginResourceHints`` from ``codeintel.core.plugins``

These aliases exist for backward compatibility. New code should import directly
from ``codeintel.core.plugins``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard, runtime_checkable

from pydantic import BaseModel

from codeintel.core.plugins.context import PluginScratch
from codeintel.core.plugins.protocol import PluginResourceHints
from codeintel.core.plugins.result import BasePluginResult

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

# =============================================================================
# Deprecated Aliases (will be removed in future version)
# =============================================================================

IngestRuntimeScratch = PluginScratch
"""DEPRECATED: Use ``PluginScratch`` from ``codeintel.core.plugins.context`` instead.

This alias exists for backward compatibility with existing code.
Migrate to the core type:

.. code-block:: python

    # Old (deprecated):
    from codeintel.ingestion.plugins.protocol import IngestRuntimeScratch

    # New (recommended):
    from codeintel.core.plugins import PluginScratch
"""

IngestResourceHints = PluginResourceHints
"""DEPRECATED: Use ``PluginResourceHints`` from ``codeintel.core.plugins.protocol`` instead.

This alias exists for backward compatibility with existing code.
Migrate to the core type:

.. code-block:: python

    # Old (deprecated):
    from codeintel.ingestion.plugins.protocol import IngestResourceHints

    # New (recommended):
    from codeintel.core.plugins import PluginResourceHints
"""

# =============================================================================
# Ingestion-Specific Types
# =============================================================================

IngestStage = Literal[
    "scan",
    "parse",
    "index",
    "enrich",
    "validate",
]

IngestSeverity = Literal[
    "fatal",
    "soft_fail",
    "skip_on_error",
]

IngestIsolationKind = Literal[
    "process",
    "thread",
    "none",
]


@dataclass(frozen=True)
class IngestPluginMetadata:
    """Metadata for an ingestion plugin.

    Captures all declarative information about an ingestion plugin for
    introspection, documentation, dependency resolution, and planning.

    Attributes
    ----------
    name
        Unique plugin identifier (e.g., "ast_extract").
    description
        Human-readable description of what the plugin does.
    stage
        Processing stage in the ingestion pipeline.
    severity
        How failures should be handled.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    depends_on
        Explicit plugin dependencies that must run first.
    provides
        Capabilities or artifacts this plugin produces.
    requires
        Capabilities required from other plugins.
    produces_tables
        DuckDB table keys populated by this plugin.
    tool_dependencies
        Tool plugins required (e.g., "pyright", "scip").
    resource_hints
        Runtime resource hints for planning.
    supports_incremental
        Whether incremental ingestion is supported.
    isolation_kind
        Type of isolation needed for execution.
    options_model
        Optional Pydantic model for plugin options validation.
    options_default
        Default options value.
    version_hash
        Version hash for cache invalidation.
    config_schema_ref
        Reference to configuration schema.
    config_class
        Step config class to auto-build from context.
    config_mapping
        Custom field mapping for config building (config_field -> context_attr).
    """

    name: str
    description: str
    stage: IngestStage
    severity: IngestSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    tool_dependencies: tuple[str, ...] = ()
    resource_hints: IngestResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: IngestIsolationKind = "none"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    config_schema_ref: str | None = None
    config_class: type | None = None
    config_mapping: Mapping[str, str] | None = None


@dataclass(frozen=True)
class IngestPluginResult(BasePluginResult):
    """Result returned by ingestion plugin execution.

    Extend BasePluginResult with ingestion-specific artifact tracking.
    The artifacts field uses Path specifically for file-based artifacts
    produced during ingestion.

    Attributes
    ----------
    artifacts
        Mapping of artifact names to file paths produced.
    """

    # Ingestion-specific: artifacts are always file paths
    artifacts: Mapping[str, Path] | None = None

    @classmethod
    def ok(
        cls,
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, Path] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
    ) -> IngestPluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        artifacts
            Optional mapping of artifact names to paths.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.

        Returns
        -------
        IngestPluginResult
            Result object marked as successful.
        """
        return cls(
            success=True,
            row_counts=row_counts,
            artifacts=artifacts,
            input_hash=input_hash,
            options_hash=options_hash,
        )


@runtime_checkable
class IngestPluginProtocol(Protocol):
    """Protocol for ingestion plugins.

    Ingestion plugins implement this protocol to be registered and executed
    by the ingestion runtime.
    """

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        IngestPluginMetadata
            Metadata describing the plugin.
        """
        ...

    def execute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Ingestion plugin execution context.

        Returns
        -------
        IngestPluginResult
            Result of plugin execution.
        """
        ...


def is_ingest_plugin(obj: object) -> TypeGuard[IngestPluginProtocol]:
    """Validate an object conforms to IngestPluginProtocol.

    This function performs runtime validation and provides type narrowing
    for the static type checker via TypeGuard. It checks that the object
    has both a metadata property returning IngestPluginMetadata and a
    callable execute method.

    Parameters
    ----------
    obj
        Object to validate.

    Returns
    -------
    TypeGuard[IngestPluginProtocol]
        True if obj conforms to the protocol, enabling type narrowing.

    Examples
    --------
    >>> from codeintel.ingestion.plugins.protocol import is_ingest_plugin
    >>> class MyPlugin:
    ...     @property
    ...     def metadata(self):
    ...         return IngestPluginMetadata(name="test", description="test", stage="parse")
    ...
    ...     def execute(self, ctx):
    ...         return IngestPluginResult.ok()
    >>> is_ingest_plugin(MyPlugin())
    True
    """
    # Check for required attributes
    if not hasattr(obj, "metadata") or not hasattr(obj, "execute"):
        return False

    # Verify execute is callable
    execute_attr = getattr(obj, "execute", None)
    if not callable(execute_attr):
        return False

    # Verify metadata returns an IngestPluginMetadata instance
    meta = getattr(obj, "metadata", None)
    return isinstance(meta, IngestPluginMetadata)


@dataclass(frozen=True)
class IngestPluginSkip:
    """Skip metadata for planned plugins that will not execute.

    Structurally equivalent to ``codeintel.core.plugins.registry.PluginSkip``
    but with ingestion-specific skip reasons. The core type uses `str` for
    maximum flexibility; this type uses a Literal for domain-specific type safety.

    Attributes
    ----------
    name
        Plugin name.
    reason
        Reason for skipping (ingestion-specific values).
    """

    name: str
    reason: Literal[
        "disabled",
        "missing_dependency",
        "missing_tool",
        "config_error",
        "incremental_skip",
    ]


@dataclass(frozen=True)
class IngestPluginPlan:
    """Resolved execution plan for ingestion plugins.

    Attributes
    ----------
    plugins
        Ordered plugins to execute.
    plan_id
        Unique identifier for this plan.
    skipped_plugins
        Plugins that were skipped during planning.
    dep_graph
        Dependency graph mapping plugin names to dependencies.
    """

    plugins: tuple[IngestPluginProtocol, ...]
    plan_id: str
    skipped_plugins: tuple[IngestPluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return plugin names in execution order.

        Returns
        -------
        tuple[str, ...]
            Plugin names in execution order.
        """
        return tuple(plugin.metadata.name for plugin in self.plugins)


DEFAULT_INGEST_PLUGINS: tuple[str, ...] = (
    "repo_scan",
    "scip_ingest",
    "cst_extract",
    "ast_extract",
    "typing_ingest",
    "coverage_ingest",
    "tests_ingest",
    "docstrings_ingest",
    "config_ingest",
)


__all__ = [
    "DEFAULT_INGEST_PLUGINS",
    "IngestIsolationKind",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestResourceHints",
    "IngestRuntimeScratch",
    "IngestSeverity",
    "IngestStage",
    "is_ingest_plugin",
]
