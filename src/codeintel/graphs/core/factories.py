"""Factory functions for creating graph plugins with minimal boilerplate.

This module provides factory functions that handle all common plugin concerns:
logging, error handling, row counting, and result wrapping. New plugins can
be defined in ~5 lines instead of ~50 lines.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

from codeintel.core.plugins.protocol import PluginIsolation, PluginResourceHints, PluginSeverity
from codeintel.core.plugins.result import PluginResult
from codeintel.graphs.core.computation import ComputationFn
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import (
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginMetaOptions,
    GraphPluginMetaOptionsInput,
    GraphPluginProtocol,
    GraphPluginStage,
    create_graph_metadata,
)
from codeintel.graphs.engine import GraphKind
from codeintel.storage.db_helpers import safe_row_counts

if TYPE_CHECKING:
    from pydantic import BaseModel


class _MetricMetaInput(TypedDict, total=False):
    """Typed kwargs for metric plugins excluding kind/stage."""

    name: str
    description: str
    severity: PluginSeverity
    enabled_by_default: bool
    depends_on: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    produces_tables: tuple[str, ...]
    requires_graph_kinds: tuple[GraphKind, ...]
    resource_hints: PluginResourceHints | None
    supports_incremental: bool
    isolation_kind: PluginIsolation
    options_model: type[BaseModel] | None
    options_default: object | None
    version_hash: str | None
    config_schema_ref: str | None
    row_count_tables: tuple[str, ...]
    cache_populates: tuple[str, ...]
    cache_consumes: tuple[str, ...]
    requires_isolation: bool
    scope_aware: bool


def _register_plugin(plugin: GraphPluginProtocol) -> None:
    """Register a plugin via the registry without inline imports."""
    registry = importlib.import_module("codeintel.graphs.core.registry")
    registry.register_graph_plugin(plugin)


@dataclass(frozen=True)
class GraphPluginSpec:
    """Specification for creating a graph plugin.

    Bundles all configuration for graph plugin creation to reduce
    function argument count.

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
        Human-readable description.
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
    produces_graph_kinds
        GraphKind values this plugin builds (for builders).
    requires_graph_kinds
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
        Tables to report row counts from.
    cache_populates
        Cache slots this plugin populates.
    cache_consumes
        Cache slots this plugin reads from.
    register
        Whether to register with the global registry.
    """

    name: str
    computation: ComputationFn
    kind: GraphPluginKind
    stage: GraphPluginStage
    description: str | None = None
    severity: PluginSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    produces_graph_kinds: tuple[GraphKind, ...] = ()
    requires_graph_kinds: tuple[GraphKind, ...] = ()
    resource_hints: PluginResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: PluginIsolation = "none"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] | None = None
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()
    requires_isolation: bool = False
    scope_aware: bool = False
    supported_scopes: tuple[str, ...] = ()
    contract_checkers: tuple[str, ...] = ()
    register: bool = True

    @staticmethod
    def from_meta(
        *,
        meta: GraphPluginMetaOptions,
        computation: ComputationFn,
        register: bool = True,
    ) -> GraphPluginSpec:
        """
        Construct a specification from metadata options and a computation.

        Returns
        -------
        GraphPluginSpec
            Immutable specification ready to materialize a plugin.
        """
        return GraphPluginSpec(
            name=meta.name or computation.__name__,
            computation=computation,
            kind=meta.kind or "metric",
            stage=meta.stage or "core",
            description=meta.description,
            severity=meta.severity,
            enabled_by_default=meta.enabled_by_default,
            depends_on=meta.depends_on,
            provides=meta.provides,
            requires=meta.requires,
            produces_tables=meta.produces_tables,
            produces_graph_kinds=meta.produces_graph_kinds,
            requires_graph_kinds=meta.requires_graph_kinds,
            resource_hints=meta.resource_hints,
            supports_incremental=meta.supports_incremental,
            isolation_kind=meta.isolation_kind,
            options_model=meta.options_model,
            options_default=meta.options_default,
            version_hash=meta.version_hash,
            config_schema_ref=meta.config_schema_ref,
            row_count_tables=meta.row_count_tables,
            cache_populates=meta.cache_populates,
            cache_consumes=meta.cache_consumes,
            requires_isolation=meta.requires_isolation,
            scope_aware=meta.scope_aware,
            supported_scopes=meta.supported_scopes,
            contract_checkers=meta.contract_checkers,
            register=register,
        )


log = logging.getLogger(__name__)


def _auto_row_counts(
    ctx: GraphPluginExecutionContext,
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

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        """Execute the plugin with automatic logging and error handling.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        PluginResult
            Result of plugin execution.
        """
        name = self._metadata.name
        catchable_errors: tuple[type[Exception], ...] = (
            RuntimeError,
            ValueError,
            TypeError,
            LookupError,
            OSError,
        )
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
                return PluginResult.fail(
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

            # Filter artifacts to only include Path values
            path_artifacts = {
                key: value for key, value in result.artifacts.items() if isinstance(value, Path)
            }

            return PluginResult.ok(
                row_counts=row_counts,
                artifacts=path_artifacts if path_artifacts else None,
            )

        except catchable_errors as exc:
            log.exception("%s.failed repo=%s commit=%s", name, ctx.repo, ctx.commit)
            return PluginResult.fail(str(exc), error_kind="compute_error")


def make_plugin_from_spec(spec: GraphPluginSpec) -> GraphPluginProtocol:
    """Create a graph plugin from a specification object.

    This factory handles all common concerns: logging, error handling,
    row counting, and result wrapping.

    Parameters
    ----------
    spec
        Complete plugin specification.

    Returns
    -------
    GraphPluginProtocol
        A fully configured plugin ready for execution.
    """
    # Auto-extract description from docstring if not provided
    resolved_description = spec.description
    if resolved_description is None:
        resolved_description = spec.computation.__doc__
        if resolved_description:
            resolved_description = resolved_description.strip().split("\n")[0]
        else:
            resolved_description = f"{spec.name} plugin"

    # Default row_count_tables to produces_tables
    resolved_row_count_tables = (
        spec.row_count_tables if spec.row_count_tables is not None else spec.produces_tables
    )

    metadata = create_graph_metadata(
        name=spec.name,
        description=resolved_description,
        kind=spec.kind,
        stage=spec.stage,
        severity=spec.severity,
        enabled_by_default=spec.enabled_by_default,
        depends_on=spec.depends_on,
        provides=spec.provides,
        requires=spec.requires,
        produces_tables=spec.produces_tables,
        produces_graph_kinds=spec.produces_graph_kinds,
        requires_graph_kinds=spec.requires_graph_kinds,
        resource_hints=spec.resource_hints,
        supports_incremental=spec.supports_incremental,
        isolation_kind=spec.isolation_kind,
        options_model=spec.options_model,
        options_default=spec.options_default,
        version_hash=spec.version_hash,
        config_schema_ref=spec.config_schema_ref,
        row_count_tables=resolved_row_count_tables,
        cache_populates=spec.cache_populates,
        cache_consumes=spec.cache_consumes,
        requires_isolation=spec.requires_isolation,
        scope_aware=spec.scope_aware,
        supported_scopes=spec.supported_scopes,
        contract_checkers=spec.contract_checkers,
    )

    plugin = FactoryPlugin(
        _metadata=metadata,
        _computation=spec.computation,
        _row_count_tables=resolved_row_count_tables,
    )

    if spec.register:
        _register_plugin(plugin)

    return plugin


def make_graph_plugin(
    *,
    computation: ComputationFn,
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[GraphPluginMetaOptionsInput],
) -> GraphPluginProtocol:
    """
    Create a graph plugin from a computation function using metadata options.

    Raises
    ------
    ValueError
        When both meta and keyword metadata overrides are provided.

    Returns
    -------
    GraphPluginProtocol
        Configured plugin ready for registration or execution.
    """
    if meta is not None and kwargs:
        message = "Provide either meta or keyword metadata, not both."
        raise ValueError(message)

    options = meta or GraphPluginMetaOptions.from_kwargs(**kwargs)
    spec = GraphPluginSpec.from_meta(meta=options, computation=computation, register=register)
    return make_plugin_from_spec(spec)


def make_metric_plugin(
    *,
    computation: ComputationFn,
    stage: GraphPluginStage,
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[_MetricMetaInput],
) -> GraphPluginProtocol:
    """
    Create a metric plugin with sensible defaults.

    Returns
    -------
    GraphPluginProtocol
        Configured metric plugin.
    """
    payload: GraphPluginMetaOptionsInput = cast(
        "GraphPluginMetaOptionsInput", {"kind": "metric", "stage": stage, **kwargs}
    )
    options = meta or GraphPluginMetaOptions.from_kwargs(**payload)
    return make_graph_plugin(computation=computation, meta=options, register=register)


def make_builder_plugin(
    *,
    computation: ComputationFn,
    stage: GraphPluginStage,
    produces_graph_kinds: tuple[GraphKind, ...],
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[_MetricMetaInput],
) -> GraphPluginProtocol:
    """
    Create a builder plugin with sensible defaults.

    Returns
    -------
    GraphPluginProtocol
        Configured builder plugin.
    """
    payload: GraphPluginMetaOptionsInput = cast(
        "GraphPluginMetaOptionsInput",
        {
            "kind": "builder",
            "stage": stage,
            "produces_graph_kinds": produces_graph_kinds,
            **kwargs,
        },
    )
    options = meta or GraphPluginMetaOptions.from_kwargs(**payload)
    return make_graph_plugin(computation=computation, meta=options, register=register)


def make_validation_plugin(
    *,
    computation: ComputationFn,
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[GraphPluginMetaOptionsInput],
) -> GraphPluginProtocol:
    """
    Create a validation plugin with sensible defaults.

    Returns
    -------
    GraphPluginProtocol
        Configured validation plugin.
    """
    payload: GraphPluginMetaOptionsInput = cast(
        "GraphPluginMetaOptionsInput",
        {
            "kind": "validation",
            "stage": "validation",
            **kwargs,
        },
    )
    options = meta or GraphPluginMetaOptions.from_kwargs(**payload)
    return make_graph_plugin(computation=computation, meta=options, register=register)


__all__ = [
    "FactoryPlugin",
    "GraphPluginSpec",
    "make_builder_plugin",
    "make_graph_plugin",
    "make_metric_plugin",
    "make_validation_plugin",
]
