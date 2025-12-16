"""Native Hamilton implementation for function_effects target.

This module provides the Hamilton native nodes for function effects classification:
- `t__function_effects__compute`: Pure compute node for effects classification
- `t__function_effects`: Materialize node that writes both tables

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.functions import compute_function_effects
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class FunctionEffectsResult:
    """Result from function effects computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="function_effects", node_type="compute")
def t__function_effects__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
) -> FunctionEffectsResult:
    """Compute side effects classification for functions.

    This is a compute node that calls the function effects computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__call_graph
        Upstream call_graph target result (for dependency).

    Returns
    -------
    FunctionEffectsResult
        Result indicating success or failure.

    Notes
    -----
    The effects classified include:
    - Pure functions vs impure
    - Side effect types (I/O, state mutation, etc.)
    - Effect evidence and reasoning
    """
    if t__call_graph.status != "succeeded":
        return FunctionEffectsResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    try:
        # Load catalog and graph runtime
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        try:
            graph_runtime = resolve_graph_runtime(
                env.gateway,
                env.snapshot,
                GraphRuntimeOptions(),
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to resolve graph runtime: %s", exc)
            graph_runtime = None

        # Build options
        opts = FunctionEffectsOptions(
            max_call_depth=3,
            require_all_callees_pure=True,
        )

        # Build inputs
        inputs = FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=graph_runtime,
            ast_map=None,
            missing_goids=None,
        )

        # Compute effects (handles persistence internally)
        compute_function_effects(env.gateway, env.snapshot, options=opts, inputs=inputs)

        return FunctionEffectsResult(success=True)

    except Exception as exc:
        log.exception("Function effects computation failed")
        return FunctionEffectsResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="function_effects", node_type="materialize")
def t__function_effects(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_effects__compute: FunctionEffectsResult,
) -> TargetRunRecord:
    """Materialize function effects target.

    This is the entry point for the function_effects target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__function_effects__compute
        Computed function effects result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.function_effects
    - analytics.function_effects_evidence
    """
    executor = NativeTargetExecutor.for_target(env, graph, "function_effects")

    if executor.should_skip():
        return executor.skip()

    if not t__function_effects__compute.success:
        return executor.fail(
            RuntimeError(t__function_effects__compute.error or "Function effects failed")
        )

    def compute() -> dict[str, int]:
        # Effects are persisted during compute - return empty counts
        return {
            "analytics.function_effects": 0,
            "analytics.function_effects_evidence": 0,
        }

    return executor.execute(compute)


__all__ = [
    "FunctionEffectsResult",
    "t__function_effects",
    "t__function_effects__compute",
]
