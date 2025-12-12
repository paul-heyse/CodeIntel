"""Explicit Hamilton nodes for the Phase 0 execution chain.

This module defines Hamilton nodes that wrap existing target plugins.
Each node function represents one target in the build system, with
dependencies expressed via function parameters (Hamilton's convention).

Phase 0 Chain
-------------
The initial chain covers a vertical slice through all modules:
- modules (ingestion) - no dependencies
- scip (ingestion) - depends on modules
- ast (ingestion) - depends on modules
- goids (graphs) - depends on scip, ast
- function_metrics (analytics) - depends on goids, ast

Later phases will generate nodes from the TargetGraph rather than
defining them explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import tag

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.hamilton.manifest_hook import (
    ManifestSaveRequest,
    TargetRunRecord,
    compute_target_input_hash,
    compute_target_options_hash,
    save_manifest,
    should_skip,
)
from codeintel.build.hamilton.metadata_bridge import from_plugin_or_target
from codeintel.build.plugin_registry import get_plugin_for_target

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph

log = logging.getLogger(__name__)


# =============================================================================
# Internal Execution Helper
# =============================================================================


def _run_target(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
) -> TargetRunRecord:
    """Execute a target plugin and return execution record.

    This helper encapsulates the shared logic for running any target:
    1. Retrieve target metadata from graph and plugin
    2. Compute options hash and input hash
    3. Check if target can be skipped (output still valid)
    4. Build execution context matching BuildExecutor
    5. Execute plugin and record timing
    6. Persist manifest on success

    Parameters
    ----------
    env
        Build environment with all dependencies.
    graph
        Target graph for looking up target metadata.
    target_name
        Name of the target to execute.

    Returns
    -------
    TargetRunRecord
        Execution record with status, timing, and row counts.
    """
    target = graph.get(target_name)

    # Get plugin for this target
    try:
        plugin = get_plugin_for_target(target_name)
    except KeyError as exc:
        log.warning("No plugin registered for target '%s': %s", target_name, exc)
        return TargetRunRecord(
            target=target_name,
            plugin_name=target.plugin,
            status="failed",
            input_hash=None,
            error=f"No plugin registered: {exc}",
        )

    # Extract metadata from plugin or target
    meta = from_plugin_or_target(plugin=plugin, target=target)

    # Compute options hash from config parameters
    raw_params = env.config.parameters_for(target_name)
    options_hash = compute_target_options_hash(raw_params) if raw_params else None

    # Compute input hash including upstream hashes
    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        options_hash=options_hash,
    )

    # Check if we can skip this target
    if should_skip(
        gateway=env.gateway,
        target=target_name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
    ):
        log.info(
            "build.hamilton.skip target=%s input_hash=%s",
            target_name,
            input_hash,
        )
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=0.0,
        )

    # Build execution context exactly like BuildExecutor
    resources = ContextResources(
        providers=env.providers,
        gateway=env.gateway,
        modules=(),  # Will be loaded from DB if needed
    )

    ctx = TargetExecutionContext(
        target=target,
        snapshot=env.snapshot,
        paths=env.paths,
        resources=resources,
        parameters=raw_params,
    )

    # Execute the plugin
    start = time.perf_counter()
    try:
        result = asyncio.run(plugin.execute(ctx))
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        log.exception(
            "build.hamilton.error target=%s duration_ms=%.1f",
            target_name,
            duration_ms,
        )
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="failed",
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            error=str(exc),
        )

    duration_ms = (time.perf_counter() - start) * 1000
    row_counts = dict(result.row_counts or {})

    # Persist manifest on success
    if result.success:
        request = ManifestSaveRequest(
            target=target_name,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            plugin=target.plugin,
            duration_ms=duration_ms,
            input_hash=input_hash,
            row_count=sum(row_counts.values()) if row_counts else None,
            options_hash=options_hash,
        )
        save_manifest(gateway=env.gateway, request=request)
        log.info(
            "build.hamilton.complete target=%s duration_ms=%.1f rows=%d",
            target_name,
            duration_ms,
            sum(row_counts.values()) if row_counts else 0,
        )
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="succeeded",
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            row_counts=row_counts,
        )

    # Plugin execution failed
    log.warning(
        "build.hamilton.failed target=%s error=%s",
        target_name,
        result.error_message,
    )
    return TargetRunRecord(
        target=target_name,
        plugin_name=meta.name,
        status="failed",
        input_hash=input_hash,
        options_hash=options_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
        error=result.error_message,
    )


# =============================================================================
# Phase 0 Hamilton Nodes
# =============================================================================

# Note: Hamilton determines dependencies from function parameter names.
# Each node function receives env and graph as common inputs, plus any
# upstream node outputs as dependencies.


@tag(domain="ingestion", target="modules")
def t__modules(env: BuildEnv, graph: TargetGraph) -> TargetRunRecord:
    """Execute the modules target (repository scan).

    This is the root node with no upstream dependencies.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and providers.
    graph
        Target graph for metadata lookup.

    Returns
    -------
    TargetRunRecord
        Execution record for the modules target.
    """
    return _run_target(env=env, graph=graph, target_name="modules")


@tag(domain="ingestion", target="scip")
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the scip target (SCIP index ingestion).

    Depends on modules being computed first.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__modules
        Execution record from the modules node.

    Returns
    -------
    TargetRunRecord
        Execution record for the scip target.
    """
    # t__modules is received to establish Hamilton DAG dependency
    _ = t__modules  # Used for dependency tracking
    return _run_target(env=env, graph=graph, target_name="scip")


@tag(domain="ingestion", target="ast")
def t__ast(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the ast target (AST extraction).

    Depends on modules being computed first.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__modules
        Execution record from the modules node.

    Returns
    -------
    TargetRunRecord
        Execution record for the ast target.
    """
    # t__modules is received to establish Hamilton DAG dependency
    _ = t__modules  # Used for dependency tracking
    return _run_target(env=env, graph=graph, target_name="ast")


@tag(domain="graphs", target="goids")
def t__goids(
    env: BuildEnv,
    graph: TargetGraph,
    t__scip: TargetRunRecord,
    t__ast: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the goids target (GOID resolution).

    Depends on both scip and ast being computed.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__scip
        Execution record from the scip node.
    t__ast
        Execution record from the ast node.

    Returns
    -------
    TargetRunRecord
        Execution record for the goids target.
    """
    # Upstream params establish Hamilton DAG dependencies
    _ = (t__scip, t__ast)  # Used for dependency tracking
    return _run_target(env=env, graph=graph, target_name="goids")


@tag(domain="analytics", target="function_metrics")
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids: TargetRunRecord,
    t__ast: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the function_metrics target (function structural metrics).

    Depends on goids and ast being computed.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__goids
        Execution record from the goids node.
    t__ast
        Execution record from the ast node.

    Returns
    -------
    TargetRunRecord
        Execution record for the function_metrics target.
    """
    # Upstream params establish Hamilton DAG dependencies
    _ = (t__goids, t__ast)  # Used for dependency tracking
    return _run_target(env=env, graph=graph, target_name="function_metrics")


# =============================================================================
# Node Registry
# =============================================================================

# These are the available node functions for Phase 0.
# Hamilton will discover them automatically from this module.

PHASE0_NODES: tuple[Any, ...] = (
    t__modules,
    t__scip,
    t__ast,
    t__goids,
    t__function_metrics,
)

# Mapping from target name to node name for executor lookups
TARGET_TO_NODE: dict[str, str] = {
    "modules": "t__modules",
    "scip": "t__scip",
    "ast": "t__ast",
    "goids": "t__goids",
    "function_metrics": "t__function_metrics",
}


__all__ = [
    "PHASE0_NODES",
    "TARGET_TO_NODE",
    "t__ast",
    "t__function_metrics",
    "t__goids",
    "t__modules",
    "t__scip",
]
