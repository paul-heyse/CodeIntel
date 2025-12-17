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
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

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
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
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
def t__entrypoints__compute(env: BuildEnv, graph: TargetGraph) -> EntrypointsResult | None:
    """Compute entrypoints for all modules in the snapshot.

    This is a pure compute node with no side effects. It scans source files
    for HTTP, CLI, and job entrypoints and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    EntrypointsResult | None
        Container with rows for entrypoints and entrypoint_tests tables.
        Returns None when manifest-skip indicates the target is current.

    Notes
    -----
    The detection identifies:
    - HTTP endpoints (FastAPI, Flask, Django, etc.)
    - CLI commands (Click, argparse, Typer)
    - Scheduled jobs and background tasks
    - Event handlers and message consumers
    """
    target = graph.get("entrypoints")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    inputs = _build_inputs(env)
    if inputs is None:
        return EntrypointsResult(entrypoint_rows=(), test_rows=())

    return compute_entrypoints_pure(env.gateway, env.snapshot, inputs)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.entrypoints"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("entrypoints"),
    table_key=value("analytics.entrypoints"),
    columns=value(tuple(ENTRYPOINTS_COLS)),
)
@tag(domain="analytics", target="entrypoints", node_type="compute", target_="entrypoints__entrypoint_rows")
def entrypoints__entrypoint_rows(
    t__entrypoints__compute: EntrypointsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.entrypoints."""
    if t__entrypoints__compute is None:
        return None
    return tuple(t__entrypoints__compute.entrypoint_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.entrypoint_tests"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("entrypoints"),
    table_key=value("analytics.entrypoint_tests"),
    columns=value(tuple(ENTRYPOINT_TESTS_COLS)),
)
@tag(domain="analytics", target="entrypoints", node_type="compute", target_="entrypoints__test_rows")
def entrypoints__test_rows(
    t__entrypoints__compute: EntrypointsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.entrypoint_tests."""
    if t__entrypoints__compute is None:
        return None
    return tuple(t__entrypoints__compute.test_rows)


@tag(domain="analytics", target="entrypoints", node_type="materialize")
def t__entrypoints(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__entrypoints: dict[str, Any],
    m__analytics__entrypoint_tests: dict[str, Any],
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
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="entrypoints",
        materializations={
            "analytics.entrypoints": m__analytics__entrypoints,
            "analytics.entrypoint_tests": m__analytics__entrypoint_tests,
        },
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__entrypoints",
    "t__entrypoints__compute",
]
