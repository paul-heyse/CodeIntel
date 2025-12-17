"""Native Hamilton implementation for external_deps target.

This module provides the Hamilton native nodes for external dependency analysis:
- `t__external_deps__compute_calls`: Pure compute node for dependency call detection
- `t__external_deps`: Materialize node that writes both tables

The compute node calls pure functions from `codeintel.analytics.dependencies.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.

Special Consideration
---------------------
The aggregated dependencies computation (`compute_external_dependencies_pure`)
reads from `analytics.external_dependency_calls` after it has been written.
Therefore, materialization of calls must happen before computing dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.dependencies.compute import (
    DependencyCallsResult,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.analytics.dependencies.core import (
    EXTERNAL_DEPENDENCIES_COLS,
    EXTERNAL_DEPENDENCY_CALLS_COLS,
    ExternalDependencyInputs,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
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

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures


log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


def _build_inputs(env: BuildEnv) -> ExternalDependencyInputs | None:
    """Build inputs for external dependency analysis.

    Loads function ASTs and builds the ExternalDependencyInputs structure
    needed for dependency call detection.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and providers.

    Returns
    -------
    ExternalDependencyInputs | None
        Inputs for dependency analysis, or None if unavailable.
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

    module_map: dict[str, str] = dict(catalog.catalog().module_by_path)
    missing_goids: set[int] = set()
    features_map: dict[int, FunctionAstFeatures] = {}

    try:
        ast_by_goid, missing_goids = load_function_asts(
            env.gateway,
            FunctionAstLoadRequest(
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                repo_root=env.snapshot.repo_root,
                catalog_provider=catalog,
            ),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to load function ASTs: %s", exc)
        return None

    for func_ast in ast_by_goid.values():
        module = catalog.module_for_path(func_ast.rel_path)
        if module is None:
            module = func_ast.rel_path.replace("/", ".").removesuffix(".py")
        module_map[func_ast.rel_path] = module

    return ExternalDependencyInputs(
        catalog_provider=catalog,
        module_map=module_map,
        ast_by_goid=ast_by_goid,
        features_map=features_map,
        missing_goids=missing_goids,
    )


@tag(domain="analytics", target="external_deps", node_type="compute")
def t__external_deps__compute_calls(env: BuildEnv, graph: TargetGraph) -> DependencyCallsResult | None:
    """Compute external dependency calls for all functions in the snapshot.

    This is a pure compute node with no side effects. It loads function
    ASTs and analyzes them for external dependency usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    DependencyCallsResult | None
        Container with rows for external_dependency_calls table.
        Returns None when manifest-skip indicates the target is current.

    Notes
    -----
    The analysis identifies:
    - Library calls matching dependency patterns
    - Usage modes (read, write, admin, etc.)
    - Evidence with code snippets
    """
    target = graph.get("external_deps")
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
        return DependencyCallsResult(rows=())

    return compute_dependency_calls_pure(
        env.gateway,
        env.snapshot,
        inputs,
    )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.external_dependency_calls"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("external_deps"),
    table_key=value("analytics.external_dependency_calls"),
    columns=value(tuple(EXTERNAL_DEPENDENCY_CALLS_COLS)),
)
@tag(domain="analytics", target="external_deps", node_type="compute", target_="external_deps__calls_rows")
def external_deps__calls_rows(
    t__external_deps__compute_calls: DependencyCallsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.external_dependency_calls."""
    if t__external_deps__compute_calls is None:
        return None
    return tuple(t__external_deps__compute_calls.rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.external_dependencies"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("external_deps"),
    table_key=value("analytics.external_dependencies"),
    columns=value(tuple(EXTERNAL_DEPENDENCIES_COLS)),
)
@tag(
    domain="analytics",
    target="external_deps",
    node_type="compute",
    target_="external_deps__dependencies_rows",
)
def external_deps__dependencies_rows(
    env: BuildEnv,
    m__analytics__external_dependency_calls: dict[str, Any],
) -> tuple[tuple[object, ...], ...] | None:
    """Compute rows for analytics.external_dependencies after calls are written."""
    status = m__analytics__external_dependency_calls.get("status")
    if status != "succeeded":
        return None

    result = compute_external_dependencies_pure(env.gateway, env.snapshot)
    return tuple(result.rows)


@tag(domain="analytics", target="external_deps", node_type="materialize")
def t__external_deps(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__external_dependency_calls: dict[str, Any],
    m__analytics__external_dependencies: dict[str, Any],
) -> TargetRunRecord:
    """Materialize both external dependency tables to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed dependency calls first, then computes and writes the
    aggregated dependencies (which reads from the calls table).

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__external_deps__compute_calls
        Computed dependency calls from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables in order:
    1. analytics.external_dependency_calls (from computed result)
    2. analytics.external_dependencies (computed after calls written)

    The order matters because external_dependencies reads from the
    external_dependency_calls table.
    """
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="external_deps",
        materializations={
            "analytics.external_dependency_calls": m__analytics__external_dependency_calls,
            "analytics.external_dependencies": m__analytics__external_dependencies,
        },
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__external_deps",
    "t__external_deps__compute_calls",
]
