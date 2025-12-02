"""Factory functions for creating graph plugins with minimal boilerplate.

This module provides factory functions that handle all common plugin concerns:
logging, error handling, row counting, and result wrapping. New plugins can
be defined in ~5 lines instead of ~50 lines.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.graphs.core.computation import ComputationFn
from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.protocol import (
    GraphPluginIsolation,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginResourceHints,
    GraphPluginSeverity,
    GraphPluginStage,
)
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.engine import GraphKind
from codeintel.storage.db_helpers import safe_row_counts

if TYPE_CHECKING:
    from pydantic import BaseModel

log = logging.getLogger(__name__)


def _auto_row_counts(
    ctx: GraphExecutionContext,
    tables: tuple[str, ...],
) -> dict[str, int]:
    """Compute row counts for tables scoped by repo/commit.

    Parameters
    ----------
    ctx
        Plugin execution context.
    tables
        Table names to count.

    Returns
    -------
    dict[str, int]
        Mapping of table name to row count.
    """
    if not tables:
        return {}
    connection = getattr(ctx.gateway, "con", None)
    counts = safe_row_counts(connection, repo=ctx.repo, commit=ctx.commit, tables=tables)
    if counts is None:
        log.debug("row_count.failed repo=%s commit=%s tables=%s", ctx.repo, ctx.commit, tables)
        return {}
    return counts


@dataclass
class FactoryPlugin:
    """Plugin created by factory functions with automatic logging/error handling.

    This class handles all common plugin concerns so that computation functions
    only need to focus on their core logic.
    """

    _metadata: GraphPluginMetadata
    _computation: ComputationFn
    _row_count_tables: tuple[str, ...]

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the plugin.
        """
        return self._metadata

    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult:
        """Execute the plugin with automatic logging and error handling.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        GraphPluginResult
            Result of plugin execution.
        """
        name = self._metadata.name
        log.info("%s.start repo=%s commit=%s", name, ctx.repo, ctx.commit)

        try:
            result = self._computation(ctx)

            if not result.success:
                log.warning(
                    "%s.soft_fail repo=%s commit=%s msg=%s",
                    name,
                    ctx.repo,
                    ctx.commit,
                    result.message,
                )
                return GraphPluginResult.fail(
                    result.message or "computation failed",
                    error_kind="compute_error",
                )

            # Use provided row counts or auto-compute from tables
            row_counts = result.row_counts or _auto_row_counts(ctx, self._row_count_tables)

            total_rows = sum(row_counts.values()) if row_counts else 0
            log.info(
                "%s.complete repo=%s commit=%s rows=%d",
                name,
                ctx.repo,
                ctx.commit,
                total_rows,
            )

            artifacts = result.artifacts
            typed_artifacts = artifacts if isinstance(artifacts, Mapping) else None
            return GraphPluginResult.ok(row_counts=row_counts, artifacts=typed_artifacts)

        except Exception as exc:
            log.exception("%s.failed repo=%s commit=%s", name, ctx.repo, ctx.commit)
            return GraphPluginResult.fail(str(exc), error_kind="compute_error")


def make_graph_plugin(  # noqa: PLR0913
    *,
    name: str,
    computation: ComputationFn,
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    description: str | None = None,
    severity: GraphPluginSeverity = "fatal",
    enabled_by_default: bool = True,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    produces_tables: tuple[str, ...] = (),
    produces_graphs: tuple[GraphKind, ...] = (),
    requires_graphs: tuple[GraphKind, ...] = (),
    resource_hints: GraphPluginResourceHints | None = None,
    supports_incremental: bool = False,
    isolation_kind: GraphPluginIsolation = "none",
    options_model: type[BaseModel] | None = None,
    options_default: object | None = None,
    version_hash: str | None = None,
    config_schema_ref: str | None = None,
    row_count_tables: tuple[str, ...] | None = None,
    cache_populates: tuple[str, ...] = (),
    cache_consumes: tuple[str, ...] = (),
    register: bool = True,
) -> GraphPluginProtocol:
    """Create a graph plugin from a computation function.

    This factory handles all common concerns: logging, error handling,
    row counting, and result wrapping. The computation function only
    needs to focus on its core logic.

    Parameters
    ----------
    name
        Unique plugin identifier.
    computation
        Function implementing the plugin logic.
    kind
        Plugin kind: builder, metric, or validation.
    stage
        Processing stage in the graph pipeline.
    description
        Human-readable description. Defaults to computation docstring.
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
    produces_graphs
        GraphKind values this plugin builds (for builders).
    requires_graphs
        GraphKind values this plugin needs (for metrics).
    resource_hints
        Runtime resource hints for planning.
    supports_incremental
        Whether incremental execution is supported.
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
    row_count_tables
        Tables to report row counts from. Defaults to produces_tables.
    cache_populates
        Cache keys this plugin populates.
    cache_consumes
        Cache keys this plugin consumes.
    register
        Whether to auto-register with global registry.

    Returns
    -------
    GraphPluginProtocol
        A fully configured plugin ready for execution.
    """
    # Auto-extract description from docstring if not provided
    resolved_description = description
    if resolved_description is None:
        resolved_description = computation.__doc__
        if resolved_description:
            # Take first line of docstring
            resolved_description = resolved_description.strip().split("\n")[0]
        else:
            resolved_description = f"{name} plugin"

    # Default row_count_tables to produces_tables
    resolved_row_count_tables = (
        row_count_tables if row_count_tables is not None else produces_tables
    )

    metadata = GraphPluginMetadata(
        name=name,
        description=resolved_description,
        kind=kind,
        stage=stage,
        severity=severity,
        enabled_by_default=enabled_by_default,
        depends_on=depends_on,
        provides=provides,
        requires=requires,
        produces_tables=produces_tables,
        produces_graphs=produces_graphs,
        requires_graphs=requires_graphs,
        resource_hints=resource_hints,
        supports_incremental=supports_incremental,
        isolation_kind=isolation_kind,
        options_model=options_model,
        options_default=options_default,
        version_hash=version_hash,
        config_schema_ref=config_schema_ref,
        row_count_tables=resolved_row_count_tables,
        cache_populates=cache_populates,
        cache_consumes=cache_consumes,
    )

    plugin = FactoryPlugin(
        _metadata=metadata,
        _computation=computation,
        _row_count_tables=resolved_row_count_tables,
    )

    if register:
        from codeintel.graphs.core.registry import register_graph_plugin  # noqa: PLC0415

        register_graph_plugin(plugin)

    return plugin


def make_metric_plugin(
    name: str,
    computation: ComputationFn,
    stage: GraphPluginStage,
    *,
    description: str | None = None,
    depends_on: tuple[str, ...] = (),
    produces_tables: tuple[str, ...] = (),
    requires_graphs: tuple[GraphKind, ...] = (),
    row_count_tables: tuple[str, ...] | None = None,
    register: bool = True,
    **kwargs: object,
) -> GraphPluginProtocol:
    """Create a metric plugin with sensible defaults.

    This is a shorthand for make_graph_plugin with kind="metric".

    Parameters
    ----------
    name
        Unique plugin identifier.
    computation
        Function implementing the metric computation.
    stage
        Processing stage (e.g., "core", "cfg", "symbol").
    description
        Human-readable description.
    depends_on
        Plugins that must run first.
    produces_tables
        Tables this plugin writes to.
    requires_graphs
        Graph types this plugin needs.
    row_count_tables
        Tables to report row counts from.
    register
        Whether to auto-register.
    **kwargs
        Additional metadata fields.

    Returns
    -------
    GraphPluginProtocol
        A metric plugin ready for execution.
    """
    return make_graph_plugin(
        name=name,
        computation=computation,
        kind="metric",
        stage=stage,
        description=description,
        depends_on=depends_on,
        produces_tables=produces_tables,
        requires_graphs=requires_graphs,
        row_count_tables=row_count_tables,
        register=register,
        **kwargs,  # type: ignore[arg-type]
    )


def make_builder_plugin(
    name: str,
    computation: ComputationFn,
    stage: GraphPluginStage,
    *,
    produces_graphs: tuple[GraphKind, ...],
    description: str | None = None,
    depends_on: tuple[str, ...] = (),
    produces_tables: tuple[str, ...] = (),
    row_count_tables: tuple[str, ...] | None = None,
    register: bool = True,
    **kwargs: object,
) -> GraphPluginProtocol:
    """Create a builder plugin with sensible defaults.

    This is a shorthand for make_graph_plugin with kind="builder".

    Parameters
    ----------
    name
        Unique plugin identifier.
    computation
        Function implementing the graph building.
    stage
        Processing stage (e.g., "goid", "edges", "structure").
    produces_graphs
        Graph types this builder creates.
    description
        Human-readable description.
    depends_on
        Plugins that must run first.
    produces_tables
        Tables this plugin writes to.
    row_count_tables
        Tables to report row counts from.
    register
        Whether to auto-register.
    **kwargs
        Additional metadata fields.

    Returns
    -------
    GraphPluginProtocol
        A builder plugin ready for execution.
    """
    return make_graph_plugin(
        name=name,
        computation=computation,
        kind="builder",
        stage=stage,
        produces_graphs=produces_graphs,
        description=description,
        depends_on=depends_on,
        produces_tables=produces_tables,
        row_count_tables=row_count_tables,
        register=register,
        **kwargs,  # type: ignore[arg-type]
    )


def make_validation_plugin(
    name: str,
    computation: ComputationFn,
    *,
    description: str | None = None,
    depends_on: tuple[str, ...] = (),
    requires_graphs: tuple[GraphKind, ...] = (),
    register: bool = True,
    **kwargs: object,
) -> GraphPluginProtocol:
    """Create a validation plugin with sensible defaults.

    This is a shorthand for make_graph_plugin with kind="validation".

    Parameters
    ----------
    name
        Unique plugin identifier.
    computation
        Function implementing the validation.
    description
        Human-readable description.
    depends_on
        Plugins that must run first.
    requires_graphs
        Graph types this validation checks.
    register
        Whether to auto-register.
    **kwargs
        Additional metadata fields.

    Returns
    -------
    GraphPluginProtocol
        A validation plugin ready for execution.
    """
    return make_graph_plugin(
        name=name,
        computation=computation,
        kind="validation",
        stage="validation",
        description=description,
        depends_on=depends_on,
        requires_graphs=requires_graphs,
        register=register,
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = [
    "FactoryPlugin",
    "make_builder_plugin",
    "make_graph_plugin",
    "make_metric_plugin",
    "make_validation_plugin",
]
