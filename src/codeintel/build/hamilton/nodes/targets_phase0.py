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
- call_graph (graphs) - depends on goids, scip
- function_metrics (analytics) - depends on goids, ast
- risk_factors (analytics) - depends on function_metrics, call_graph

Features
--------
- Upstream failure gating: downstream targets skip if upstream fails
- Force bypass: targets in env.force_targets skip manifest checks
- Dataset population: successful targets populate TargetRunRecord.datasets

Later phases will generate nodes from the TargetGraph rather than
defining them explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import tag

from codeintel.build.parameters import EMPTY_PARAMETERS

if TYPE_CHECKING:
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.plugin import TargetPluginProtocol
    from codeintel.build.result import TargetResult

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.dataset_ref import refs_from_target_result, refs_to_tuple
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
from codeintel.build.targets import OutputTarget, TargetGraph

log = logging.getLogger(__name__)


# =============================================================================
# Internal Helpers
# =============================================================================


def _check_upstream_failures(upstream: tuple[TargetRunRecord, ...]) -> tuple[str, ...]:
    """Return names of upstream targets that failed.

    Parameters
    ----------
    upstream
        Records from upstream nodes.

    Returns
    -------
    tuple[str, ...]
        Target names with status="failed".
    """
    return tuple(r.target for r in upstream if r.status == "failed")


@dataclass(frozen=True)
class HashComputation:
    """Hashes and raw parameters for a target execution."""

    input_hash: str | None
    options_hash: str | None
    raw_params: TargetParameters | None


def _compute_hashes(
    env: BuildEnv,
    target: OutputTarget,
    target_name: str,
) -> HashComputation:
    """Compute options and input hashes for a target.

    Parameters
    ----------
    env
        Build environment containing snapshot and gateway.
    target
        Target metadata.
    target_name
        Human-readable target name.

    Returns
    -------
    HashComputation
        Input hash, options hash, and raw parameters used for execution.
    """
    raw_params = env.config.parameters_for(target_name)
    options_hash = compute_target_options_hash(raw_params) if raw_params else None
    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        options_hash=options_hash,
    )
    return HashComputation(
        input_hash=input_hash,
        options_hash=options_hash,
        raw_params=raw_params,
    )


def _should_skip_target(
    env: BuildEnv,
    target_name: str,
    input_hash: str | None,
) -> bool:
    """Check if target should be skipped based on manifest.

    Parameters
    ----------
    env
        Build environment with manifest access.
    target_name
        Target under evaluation.
    input_hash
        Computed input hash for the target.

    Returns
    -------
    bool
        True if target has already been computed with the same hash.
    """
    if env.is_forced(target_name):
        log.info("build.hamilton.force target=%s input_hash=%s", target_name, input_hash)
        return False
    if input_hash is None:
        return False
    return should_skip(
        gateway=env.gateway,
        target=target_name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
    )


def _execute_plugin(
    env: BuildEnv,
    target: OutputTarget,
    raw_params: TargetParameters | None,
    plugin: TargetPluginProtocol,
) -> tuple[TargetResult, float]:
    """Execute the plugin and return (result, duration_ms).

    Parameters
    ----------
    env
        Build environment passed into plugin.
    target
        Target metadata used to build the execution context.
    raw_params
        Raw parameters resolved from configuration.
    plugin
        Plugin implementing TargetPluginProtocol.

    Returns
    -------
    tuple[TargetResult, float]
        Plugin result and elapsed duration in milliseconds.
    """
    resources = ContextResources(
        providers=env.providers,
        gateway=env.gateway,
        modules=(),
    )
    ctx = TargetExecutionContext(
        target=target,
        snapshot=env.snapshot,
        paths=env.paths,
        resources=resources,
        parameters=raw_params or EMPTY_PARAMETERS,
    )
    start = time.perf_counter()
    result = asyncio.run(plugin.execute(ctx))
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms


@dataclass(frozen=True)
class _SuccessRecordParams:
    """Parameters for building a success record."""

    env: BuildEnv
    target: OutputTarget
    target_name: str
    meta_name: str
    input_hash: str | None
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int]


def _build_success_record(params: _SuccessRecordParams) -> TargetRunRecord:
    """Build record and save manifest for successful execution.

    Parameters
    ----------
    params
        Aggregated parameters for manifest persistence and record creation.

    Returns
    -------
    TargetRunRecord
        Execution record for the successfully completed target.
    """
    if params.input_hash is not None:
        request = ManifestSaveRequest(
            target=params.target_name,
            repo=params.env.snapshot.repo,
            commit=params.env.snapshot.commit,
            plugin=params.target.plugin,
            duration_ms=params.duration_ms,
            input_hash=params.input_hash,
            row_count=sum(params.row_counts.values()) if params.row_counts else None,
            options_hash=params.options_hash,
        )
        save_manifest(gateway=params.env.gateway, request=request)

    table_keys = params.target.contract.table_keys or params.target.table_keys
    refs = refs_from_target_result(
        target_name=params.target_name,
        table_keys=table_keys,
        row_counts=params.row_counts,
    )
    datasets = refs_to_tuple(refs)

    log.info(
        "build.hamilton.complete target=%s duration_ms=%.1f rows=%d datasets=%d",
        params.target_name,
        params.duration_ms,
        sum(params.row_counts.values()) if params.row_counts else 0,
        len(datasets),
    )
    return TargetRunRecord(
        target=params.target_name,
        plugin_name=params.meta_name,
        status="succeeded",
        input_hash=params.input_hash,
        options_hash=params.options_hash,
        duration_ms=params.duration_ms,
        row_counts=params.row_counts,
        datasets=datasets,
    )


# =============================================================================
# Main Execution Helper
# =============================================================================


def _run_target(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    upstream: tuple[TargetRunRecord, ...] = (),
) -> TargetRunRecord:
    """Execute a target plugin and return execution record.

    Parameters
    ----------
    env
        Build environment with all dependencies.
    graph
        Target graph for looking up target metadata.
    target_name
        Name of the target to execute.
    upstream
        Tuple of upstream TargetRunRecord objects for failure gating.

    Returns
    -------
    TargetRunRecord
        Execution record with status, timing, row counts, and datasets.
    """
    target = graph.get(target_name)

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

    meta = from_plugin_or_target(plugin=plugin, target=target)

    # Check upstream failures
    failed_upstream = _check_upstream_failures(upstream)
    if failed_upstream:
        log.info("build.hamilton.upstream_failed target=%s", target_name)
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            input_hash=None,
            error=f"upstream_failed:{','.join(failed_upstream)}",
        )

    # Compute hashes
    hashes = _compute_hashes(env, target, target_name)

    # Check skip
    if _should_skip_target(env, target_name, hashes.input_hash):
        log.info(
            "build.hamilton.skip target=%s input_hash=%s",
            target_name,
            hashes.input_hash,
        )
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            input_hash=hashes.input_hash,
            options_hash=hashes.options_hash,
            duration_ms=0.0,
        )

    # Execute plugin
    try:
        result, duration_ms = _execute_plugin(env, target, hashes.raw_params, plugin)
    except Exception as exc:
        log.exception("build.hamilton.error target=%s", target_name)
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="failed",
            input_hash=hashes.input_hash,
            options_hash=hashes.options_hash,
            error=str(exc),
        )

    row_counts = dict(result.row_counts or {})

    if result.success:
        params = _SuccessRecordParams(
            env=env,
            target=target,
            target_name=target_name,
            meta_name=meta.name,
            input_hash=hashes.input_hash,
            options_hash=hashes.options_hash,
            duration_ms=duration_ms,
            row_counts=row_counts,
        )
        return _build_success_record(params)

    log.warning("build.hamilton.failed target=%s error=%s", target_name, result.error_message)
    return TargetRunRecord(
        target=target_name,
        plugin_name=meta.name,
        status="failed",
        input_hash=hashes.input_hash,
        options_hash=hashes.options_hash,
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
    return _run_target(env=env, graph=graph, target_name="modules", upstream=())


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
    return _run_target(
        env=env,
        graph=graph,
        target_name="scip",
        upstream=(t__modules,),
    )


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
    return _run_target(
        env=env,
        graph=graph,
        target_name="ast",
        upstream=(t__modules,),
    )


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
    return _run_target(
        env=env,
        graph=graph,
        target_name="goids",
        upstream=(t__scip, t__ast),
    )


@tag(domain="graphs", target="call_graph")
def t__call_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids: TargetRunRecord,
    t__scip: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the call_graph target (function call graph construction).

    Depends on goids and scip being computed.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__goids
        Execution record from the goids node.
    t__scip
        Execution record from the scip node.

    Returns
    -------
    TargetRunRecord
        Execution record for the call_graph target.
    """
    return _run_target(
        env=env,
        graph=graph,
        target_name="call_graph",
        upstream=(t__goids, t__scip),
    )


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
    return _run_target(
        env=env,
        graph=graph,
        target_name="function_metrics",
        upstream=(t__goids, t__ast),
    )


@tag(domain="analytics", target="risk_factors")
def t__risk_factors(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_metrics: TargetRunRecord,
    t__call_graph: TargetRunRecord,
) -> TargetRunRecord:
    """Execute the risk_factors target (composite risk factors per function).

    Depends on function_metrics and call_graph being computed.

    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph.
    t__function_metrics
        Execution record from the function_metrics node.
    t__call_graph
        Execution record from the call_graph node.

    Returns
    -------
    TargetRunRecord
        Execution record for the risk_factors target.
    """
    return _run_target(
        env=env,
        graph=graph,
        target_name="risk_factors",
        upstream=(t__function_metrics, t__call_graph),
    )


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
    t__call_graph,
    t__function_metrics,
    t__risk_factors,
)

# Mapping from target name to node name for executor lookups
TARGET_TO_NODE: dict[str, str] = {
    "modules": "t__modules",
    "scip": "t__scip",
    "ast": "t__ast",
    "goids": "t__goids",
    "call_graph": "t__call_graph",
    "function_metrics": "t__function_metrics",
    "risk_factors": "t__risk_factors",
}


__all__ = [
    "PHASE0_NODES",
    "TARGET_TO_NODE",
    "_run_target",
    "t__ast",
    "t__call_graph",
    "t__function_metrics",
    "t__goids",
    "t__modules",
    "t__risk_factors",
    "t__scip",
]
