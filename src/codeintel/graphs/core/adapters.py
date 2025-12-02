"""Adapters for converting legacy computation functions to standard signatures.

This module provides utilities to adapt existing computation functions
(with various signatures) to the standardized ComputationFn signature.

Architecture Notes
------------------
This module imports from analytics.graph_runtime when runtime=True is requested.
This is an intentional delegation - graphs orchestrates plugin execution but
delegates runtime construction to analytics (Option B architecture).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from codeintel.graphs.core.computation import ComputationFn, ComputationResult

if TYPE_CHECKING:
    from codeintel.graphs.core.context import GraphExecutionContext


def adapt_legacy_computation(  # noqa: PLR0913
    fn: Callable[..., Any],
    *,
    gateway: bool = True,
    repo: bool = True,
    commit: bool = True,
    runtime: bool = False,
    context_arg: str | None = None,
    extra_kwargs: dict[str, Callable[[GraphExecutionContext], object]] | None = None,
) -> ComputationFn:
    """Wrap a legacy computation function to match ComputationFn signature.

    This adapter allows existing computation functions with various signatures
    to be used with the factory pattern without modification.

    Parameters
    ----------
    fn
        The legacy computation function to adapt.
    gateway
        Whether to pass ctx.gateway as 'gateway' argument.
    repo
        Whether to pass ctx.repo as 'repo' argument.
    commit
        Whether to pass ctx.commit as 'commit' argument.
    runtime
        Whether to resolve and pass a GraphRuntime as 'runtime' argument.
    context_arg
        If set, pass None as this argument name (for optional context params).
    extra_kwargs
        Additional keyword arguments as callables that extract values from ctx.

    Returns
    -------
    ComputationFn
        A function matching the standard computation signature.

    Examples
    --------
    Adapt a function with gateway/repo/commit signature:

    >>> adapted = adapt_legacy_computation(compute_cfg_metrics, context_arg="context")
    >>> result = adapted(ctx)

    Adapt a function that needs runtime:

    >>> adapted = adapt_legacy_computation(
    ...     compute_metrics,
    ...     runtime=True,
    ...     extra_kwargs={"catalog": lambda ctx: ctx.catalog_provider},
    ... )
    """
    resolved_extra_kwargs = extra_kwargs or {}

    def adapted(ctx: GraphExecutionContext) -> ComputationResult:
        """
        Execute the adapted legacy function.

        Returns
        -------
        ComputationResult
            Success result for the computation.
        """
        kwargs: dict[str, object] = {}

        if gateway:
            kwargs["gateway"] = ctx.gateway

        if repo:
            kwargs["repo"] = ctx.repo

        if commit:
            kwargs["commit"] = ctx.commit

        if runtime:
            # Resolve graph runtime from context
            from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
                GraphRuntimeOptions,
                resolve_graph_runtime,
            )
            from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

            resolved_runtime = resolve_graph_runtime(
                ctx.gateway,
                ctx.snapshot,
                GraphRuntimeOptions(
                    snapshot=ctx.snapshot,
                    backend=GraphBackendConfig(),
                ),
            )
            kwargs["runtime"] = resolved_runtime

        if context_arg is not None:
            kwargs[context_arg] = None

        # Apply extra kwargs
        for key, extractor in resolved_extra_kwargs.items():
            kwargs[key] = extractor(ctx)

        # Call the legacy function
        fn(**kwargs)

        # Return success - row counts will be auto-computed by FactoryPlugin
        return ComputationResult.ok()

    # Preserve docstring for description extraction
    adapted.__doc__ = fn.__doc__
    adapted.__name__ = getattr(fn, "__name__", "adapted")

    return adapted


def adapt_with_row_counts(  # noqa: PLR0913
    fn: Callable[..., dict[str, int] | None],
    *,
    gateway: bool = True,
    repo: bool = True,
    commit: bool = True,
    runtime: bool = False,
    context_arg: str | None = None,
    extra_kwargs: dict[str, Callable[[GraphExecutionContext], object]] | None = None,
) -> ComputationFn:
    """Wrap a legacy function that returns row counts.

    Similar to adapt_legacy_computation but for functions that return
    a dict of row counts instead of None.

    Parameters
    ----------
    fn
        Legacy function returning dict[str, int] or None.
    gateway
        Whether to pass ctx.gateway.
    repo
        Whether to pass ctx.repo.
    commit
        Whether to pass ctx.commit.
    runtime
        Whether to resolve and pass GraphRuntime.
    context_arg
        Optional context argument name.
    extra_kwargs
        Additional kwargs from context.

    Returns
    -------
    ComputationFn
        Adapted function.
    """
    resolved_extra_kwargs = extra_kwargs or {}

    def adapted(ctx: GraphExecutionContext) -> ComputationResult:
        """
        Execute the adapted legacy function.

        Returns
        -------
        ComputationResult
            Success result including any row counts.
        """
        kwargs: dict[str, object] = {}

        if gateway:
            kwargs["gateway"] = ctx.gateway

        if repo:
            kwargs["repo"] = ctx.repo

        if commit:
            kwargs["commit"] = ctx.commit

        if runtime:
            from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
                GraphRuntimeOptions,
                resolve_graph_runtime,
            )
            from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

            resolved_runtime = resolve_graph_runtime(
                ctx.gateway,
                ctx.snapshot,
                GraphRuntimeOptions(
                    snapshot=ctx.snapshot,
                    backend=GraphBackendConfig(),
                ),
            )
            kwargs["runtime"] = resolved_runtime

        if context_arg is not None:
            kwargs[context_arg] = None

        for key, extractor in resolved_extra_kwargs.items():
            kwargs[key] = extractor(ctx)

        row_counts = fn(**kwargs)
        return ComputationResult.ok(row_counts=row_counts or {})

    adapted.__doc__ = fn.__doc__
    adapted.__name__ = getattr(fn, "__name__", "adapted")

    return adapted


def adapt_simple(
    fn: Callable[[GraphExecutionContext], None],
) -> ComputationFn:
    """Wrap a simple function that takes only context and returns None.

    Parameters
    ----------
    fn
        Function taking GraphExecutionContext and returning None.

    Returns
    -------
    ComputationFn
        Adapted function returning ComputationResult.
    """

    def adapted(ctx: GraphExecutionContext) -> ComputationResult:
        """
        Execute the simple function.

        Returns
        -------
        ComputationResult
            Success result for the computation.
        """
        fn(ctx)
        return ComputationResult.ok()

    adapted.__doc__ = fn.__doc__
    adapted.__name__ = getattr(fn, "__name__", "adapted")

    return adapted


__all__ = [
    "adapt_legacy_computation",
    "adapt_simple",
    "adapt_with_row_counts",
]
