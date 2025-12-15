"""Native Hamilton implementation for entrypoints target.

This module provides the Hamilton native nodes for entrypoint detection:
- `t__entrypoints__compute`: Pure compute node for entrypoint detection
- `t__entrypoints`: Materialize node that writes both tables

The compute node calls pure functions from `codeintel.analytics.entrypoints.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.entrypoints.compute import (
    EntrypointsResult,
    compute_entrypoints_pure,
)
from codeintel.analytics.entrypoints.core import (
    ENTRYPOINT_TESTS_COLS,
    ENTRYPOINTS_COLS,
    EntrypointBuildInputs,
)
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.storage.helpers.module_index import load_module_map

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures


def _build_inputs(env: BuildEnv) -> EntrypointBuildInputs | None:
    """Build inputs for entrypoint detection.

    Loads catalog, module map, and features needed for entrypoint detection.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and providers.

    Returns
    -------
    EntrypointBuildInputs | None
        Inputs for entrypoint detection, or None if unavailable.
    """
    try:
        catalog = CatalogService.from_db(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
    except (RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return None

    module_map = load_module_map(
        env.gateway,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        logger=log,
    )

    # Load function features
    features_map: dict[int, FunctionAstFeatures] = {}
    try:
        provider = FeaturesProvider(
            gateway=env.gateway,
            snapshot=env.snapshot,
            catalog_provider=catalog,
        )
        features_map = provider.get()
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to compute function features: %s", exc)

    return EntrypointBuildInputs(
        catalog_provider=catalog,
        module_map=module_map,
        features_map=features_map,
    )


@tag(domain="analytics", target="entrypoints", node_type="compute")
def t__entrypoints__compute(env: BuildEnv) -> EntrypointsResult:
    """Compute entrypoints for all modules in the snapshot.

    This is a pure compute node with no side effects. It scans source files
    for HTTP, CLI, and job entrypoints and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    EntrypointsResult
        Container with rows for entrypoints and entrypoint_tests tables.

    Notes
    -----
    The detection identifies:
    - HTTP endpoints (FastAPI, Flask, Django, etc.)
    - CLI commands (Click, argparse, Typer)
    - Scheduled jobs and background tasks
    - Event handlers and message consumers
    """
    inputs = _build_inputs(env)
    if inputs is None:
        return EntrypointsResult(entrypoint_rows=(), test_rows=())

    return compute_entrypoints_pure(env.gateway, env.snapshot, inputs)


@tag(domain="analytics", target="entrypoints", node_type="materialize")
def t__entrypoints(
    env: BuildEnv,
    graph: TargetGraph,
    t__entrypoints__compute: EntrypointsResult,
) -> TargetRunRecord:
    """Materialize both entrypoint tables to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed entrypoints to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__entrypoints__compute
        Computed entrypoints from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.entrypoints
    - analytics.entrypoint_tests
    """
    executor = NativeTargetExecutor.for_target(env, graph, "entrypoints")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure tables exist
        backend = env.gateway.policy
        backend.ensure_table("analytics.entrypoints")
        backend.ensure_table("analytics.entrypoint_tests")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="entrypoints",
            input_hash=executor.input_hash,
        )

        row_counts: dict[str, int] = {}

        # Materialize entrypoints table
        ep_ref = materialize_rows(
            ctx,
            "analytics.entrypoints",
            t__entrypoints__compute.entrypoint_rows,
            ENTRYPOINTS_COLS,
        )
        row_counts["analytics.entrypoints"] = ep_ref.row_count or 0

        # Materialize entrypoint_tests table
        tests_ref = materialize_rows(
            ctx,
            "analytics.entrypoint_tests",
            t__entrypoints__compute.test_rows,
            ENTRYPOINT_TESTS_COLS,
        )
        row_counts["analytics.entrypoint_tests"] = tests_ref.row_count or 0

        return row_counts

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__entrypoints",
    "t__entrypoints__compute",
]
