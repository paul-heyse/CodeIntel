"""Native Hamilton implementations for dependency and entrypoint targets.

This module consolidates related targets that analyze dependency structure:

- ``external_deps``: External dependency calls + aggregated dependencies.
- ``entrypoints``: Application entrypoint and test detection.

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

from hamilton.function_modifiers import cache, source, tag, value
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
from codeintel.analytics.entrypoints.compute import EntrypointsResult, compute_entrypoints_pure
from codeintel.analytics.entrypoints.core import (
    ENTRYPOINT_TESTS_COLS,
    ENTRYPOINTS_COLS,
    EntrypointBuildInputs,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.storage.helpers.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures


log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    TargetGraph,
    TargetRunRecord,
    DependencyCallsResult,
    EntrypointsResult,
)


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


@cache(format="memory")
@tag(node_type="helper", domain="analytics", target="external_deps")
def external_deps_inputs(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
) -> ExternalDependencyInputs | None:
    """Build and cache inputs for external dependency analysis.

    Returns
    -------
    ExternalDependencyInputs | None
        Prepared inputs, or None when upstream call graph failed.
    """
    if t__call_graph.status != "succeeded":
        return None
    return _build_inputs(env)


@tag(domain="analytics", target="external_deps", node_type="compute")
def t__external_deps__compute_calls(
    env: BuildEnv,
    graph: TargetGraph,
    external_deps_inputs: ExternalDependencyInputs | None,
) -> DependencyCallsResult | None:
    """Compute external dependency calls for all functions in the snapshot.

    This is a pure compute node with no side effects. It loads function
    ASTs and analyzes them for external dependency usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for manifest-driven skip checks.
    external_deps_inputs
        Pre-built AST and module inputs for dependency analysis.

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

    if external_deps_inputs is None:
        return None

    return compute_dependency_calls_pure(
        env.gateway,
        env.snapshot,
        external_deps_inputs,
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
@tag(
    domain="analytics",
    target="external_deps",
    node_type="compute",
    target_="external_deps__calls_rows",
)
def external_deps__calls_rows(
    t__external_deps__compute_calls: DependencyCallsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.external_dependency_calls.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
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
    """Compute rows for analytics.external_dependencies after calls are written.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the calls write failed.
    """
    status = m__analytics__external_dependency_calls.get("status")
    if status != "succeeded":
        return None

    result = compute_external_dependencies_pure(env.gateway, env.snapshot)
    return tuple(result.rows)


@tag(domain="analytics", target="external_deps", node_type="materialize")
def t__external_deps(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph: TargetRunRecord,
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
    t__call_graph
        Upstream call graph record (must succeed for correct attribution).
    m__analytics__external_dependency_calls
        Materialization metadata for analytics.external_dependency_calls.
    m__analytics__external_dependencies
        Materialization metadata for analytics.external_dependencies.

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
    if t__call_graph.status != "succeeded":
        executor = NativeTargetExecutor.for_target(env, graph, "external_deps")
        return executor.fail(
            RuntimeError(f"Upstream call_graph target failed: {t__call_graph.error}")
        )
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
    "t__entrypoints",
    "t__entrypoints__compute",
    "t__external_deps",
    "t__external_deps__compute_calls",
]


# ---------------------------------------------------------------------------
# entrypoints target
# ---------------------------------------------------------------------------


def _build_entrypoint_inputs(env: BuildEnv) -> EntrypointBuildInputs | None:
    """Build inputs for entrypoint detection.

    Returns
    -------
    EntrypointBuildInputs | None
        Prepared inputs, or None when required data is unavailable.
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


@cache(format="memory")
@tag(node_type="helper", domain="analytics", target="entrypoints")
def entrypoints_inputs(
    env: BuildEnv,
    t__goids: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
    t__test_profile: TargetRunRecord,
) -> EntrypointBuildInputs | None:
    """Build and cache inputs for entrypoint detection.

    Returns
    -------
    EntrypointBuildInputs | None
        Prepared inputs, or None when required upstream targets failed.
    """
    if t__goids.status != "succeeded":
        return None
    if t__semantic_roles.status != "succeeded":
        return None
    if t__test_profile.status != "succeeded":
        return None
    return _build_entrypoint_inputs(env)


@tag(domain="analytics", target="entrypoints", node_type="compute")
def t__entrypoints__compute(
    env: BuildEnv,
    graph: TargetGraph,
    entrypoints_inputs: EntrypointBuildInputs | None,
) -> EntrypointsResult | None:
    """Compute entrypoints for all modules in the snapshot.

    Returns
    -------
    EntrypointsResult | None
        Computed entrypoints, or None when skipped or inputs are unavailable.
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

    if entrypoints_inputs is None:
        return None

    return compute_entrypoints_pure(env.gateway, env.snapshot, entrypoints_inputs)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.entrypoints"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("entrypoints"),
    table_key=value("analytics.entrypoints"),
    columns=value(tuple(ENTRYPOINTS_COLS)),
)
@tag(
    domain="analytics",
    target="entrypoints",
    node_type="compute",
    target_="entrypoints__entrypoint_rows",
)
def entrypoints__entrypoint_rows(
    t__entrypoints__compute: EntrypointsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.entrypoints.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
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
@tag(
    domain="analytics", target="entrypoints", node_type="compute", target_="entrypoints__test_rows"
)
def entrypoints__test_rows(
    t__entrypoints__compute: EntrypointsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.entrypoint_tests.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
    if t__entrypoints__compute is None:
        return None
    return tuple(t__entrypoints__compute.test_rows)


@tag(domain="analytics", target="entrypoints", node_type="helper")
def entrypoints__upstream_error(
    t__goids: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
    t__test_profile: TargetRunRecord,
) -> str | None:
    """Return an upstream failure message for entrypoints, if any.

    Returns
    -------
    str | None
        Failure message when any prerequisite target failed, otherwise None.
    """
    if t__goids.status != "succeeded":
        return f"Upstream goids target failed: {t__goids.error}"
    if t__semantic_roles.status != "succeeded":
        return f"Upstream semantic_roles target failed: {t__semantic_roles.error}"
    if t__test_profile.status != "succeeded":
        return f"Upstream test_profile target failed: {t__test_profile.error}"
    return None


@tag(domain="analytics", target="entrypoints", node_type="helper")
def entrypoints__materializations(
    m__analytics__entrypoints: dict[str, Any],
    m__analytics__entrypoint_tests: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect entrypoints materialization payloads into a single mapping.

    Returns
    -------
    dict[str, dict[str, Any]]
        Materialization metadata keyed by table key.
    """
    return {
        "analytics.entrypoints": m__analytics__entrypoints,
        "analytics.entrypoint_tests": m__analytics__entrypoint_tests,
    }


@tag(domain="analytics", target="entrypoints", node_type="materialize")
def t__entrypoints(
    env: BuildEnv,
    graph: TargetGraph,
    entrypoints__upstream_error: str | None,
    entrypoints__materializations: dict[str, dict[str, Any]],
) -> TargetRunRecord:
    """Materialize both entrypoint tables to DuckDB.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if entrypoints__upstream_error is not None:
        executor = NativeTargetExecutor.for_target(env, graph, "entrypoints")
        return executor.fail(RuntimeError(entrypoints__upstream_error))

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="entrypoints",
        materializations=entrypoints__materializations,
    )
