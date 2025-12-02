"""Decorator-based plugin registration for ingestion plugins.

This module provides the @ingest_plugin decorator for creating
ingestion plugins from functions, with automatic registration
to the global registry.

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from codeintel.ingestion.plugins.protocol import (
    IngestIsolationKind,
    IngestPluginContext,
    IngestPluginMetadata,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestResourceHints,
    IngestSeverity,
    IngestStage,
)


@dataclass
class FunctionalIngestPlugin:
    """Plugin implementation wrapping a callable.

    Provides a simple way to create ingestion plugins from functions.
    """

    _metadata: IngestPluginMetadata
    _execute_fn: Callable[[IngestPluginContext], IngestPluginResult]

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        IngestPluginMetadata
            Metadata for the wrapped plugin.
        """
        return self._metadata

    def execute(self, ctx: IngestPluginContext) -> IngestPluginResult:
        """Execute the wrapped function.

        Parameters
        ----------
        ctx
            Ingestion plugin execution context.

        Returns
        -------
        IngestPluginResult
            Result produced by the underlying callable.
        """
        return self._execute_fn(ctx)


def ingest_plugin(  # noqa: PLR0913
    *,
    name: str,
    description: str,
    stage: IngestStage,
    severity: IngestSeverity = "fatal",
    enabled_by_default: bool = True,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    produces_tables: tuple[str, ...] = (),
    tool_dependencies: tuple[str, ...] = (),
    resource_hints: IngestResourceHints | None = None,
    supports_incremental: bool = False,
    isolation_kind: IngestIsolationKind = "none",
    options_model: type[BaseModel] | None = None,
    options_default: object | None = None,
    version_hash: str | None = None,
    config_schema_ref: str | None = None,
    register: bool = True,
) -> Callable[[Callable[[IngestPluginContext], IngestPluginResult]], FunctionalIngestPlugin]:
    """Decorate a function as an ingestion plugin.

    Parameters
    ----------
    name
        Unique plugin identifier.
    description
        Human-readable description.
    stage
        Processing stage in the pipeline.
    severity
        How failures should be handled.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    depends_on
        Explicit plugin dependencies.
    provides
        Capabilities provided.
    requires
        Capabilities required.
    produces_tables
        DuckDB table keys populated.
    tool_dependencies
        Tool plugins required.
    resource_hints
        Runtime hints.
    supports_incremental
        Whether incremental ingestion is supported.
    isolation_kind
        Type of isolation needed.
    options_model
        Pydantic model for options validation.
    options_default
        Default options value.
    version_hash
        Version hash for caching.
    config_schema_ref
        Reference to config schema.
    register
        Whether to auto-register with global registry.

    Returns
    -------
    Callable
        Decorator that creates a FunctionalIngestPlugin.

    Examples
    --------
    >>> @ingest_plugin(
    ...     name="my_plugin",
    ...     description="Example plugin",
    ...     stage="enrich",
    ...     produces_tables=("analytics.my_table",),
    ...     depends_on=("repo_scan",),
    ...     register=False,
    ... )
    ... def my_plugin_fn(ctx: IngestPluginContext) -> IngestPluginResult:
    ...     return IngestPluginResult.ok()
    """

    def decorator(
        fn: Callable[[IngestPluginContext], IngestPluginResult],
    ) -> FunctionalIngestPlugin:
        meta = IngestPluginMetadata(
            name=name,
            description=description,
            stage=stage,
            severity=severity,
            enabled_by_default=enabled_by_default,
            depends_on=depends_on,
            provides=provides,
            requires=requires,
            produces_tables=produces_tables,
            tool_dependencies=tool_dependencies,
            resource_hints=resource_hints,
            supports_incremental=supports_incremental,
            isolation_kind=isolation_kind,
            options_model=options_model,
            options_default=options_default,
            version_hash=version_hash,
            config_schema_ref=config_schema_ref,
        )

        plugin_instance = FunctionalIngestPlugin(_metadata=meta, _execute_fn=fn)

        if register:
            from codeintel.ingestion.plugins.registry import register_ingest_plugin

            register_ingest_plugin(plugin_instance)

        return plugin_instance

    return decorator


class ClassBasedIngestPlugin:
    """Base class for class-based ingestion plugins.

    Provides a convenient base for plugins that need to maintain state
    or have complex initialization logic.

    Examples
    --------
    >>> class MyPlugin(ClassBasedIngestPlugin):
    ...     name = "my_class_plugin"
    ...     description = "Example class-based plugin"
    ...     stage: Literal["enrich"] = "enrich"
    ...     produces_tables = ("analytics.my_table",)
    ...     depends_on = ("repo_scan",)
    ...
    ...     def execute(self, ctx: IngestPluginContext) -> IngestPluginResult:
    ...         return IngestPluginResult.ok()
    """

    # Class attributes to override
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

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Return plugin metadata derived from class attributes.

        Returns
        -------
        IngestPluginMetadata
            Metadata for this plugin.
        """
        return IngestPluginMetadata(
            name=self.name,
            description=self.description,
            stage=self.stage,
            severity=self.severity,
            enabled_by_default=self.enabled_by_default,
            depends_on=self.depends_on,
            provides=self.provides,
            requires=self.requires,
            produces_tables=self.produces_tables,
            tool_dependencies=self.tool_dependencies,
            resource_hints=self.resource_hints,
            supports_incremental=self.supports_incremental,
            isolation_kind=self.isolation_kind,
            options_model=self.options_model,
            options_default=self.options_default,
            version_hash=self.version_hash,
            config_schema_ref=self.config_schema_ref,
        )

    def execute(self, ctx: IngestPluginContext) -> IngestPluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Ingestion plugin execution context.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        message = f"{type(self).__name__} must implement execute()"
        raise NotImplementedError(message)


def register_class_plugin(
    plugin_class: type[ClassBasedIngestPlugin],
) -> IngestPluginProtocol:
    """Instantiate and register a class-based plugin.

    Parameters
    ----------
    plugin_class
        Plugin class to instantiate and register.

    Returns
    -------
    IngestPluginProtocol
        The instantiated plugin.
    """
    from codeintel.ingestion.plugins.registry import register_ingest_plugin

    instance = plugin_class()
    register_ingest_plugin(instance)
    return instance


__all__ = [
    "ClassBasedIngestPlugin",
    "FunctionalIngestPlugin",
    "ingest_plugin",
    "register_class_plugin",
]
