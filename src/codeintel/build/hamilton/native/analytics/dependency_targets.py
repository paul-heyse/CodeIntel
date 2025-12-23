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
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.analytics.dependencies.compute import (
    DependencyCallsResult,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.analytics.dependencies.core import ExternalDependencyInputs
from codeintel.analytics.entrypoints.compute import EntrypointsResult, compute_entrypoints_pure
from codeintel.analytics.entrypoints.core import EntrypointBuildInputs
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.resources import ResourceNotFoundError

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

EXTERNAL_DEPS_TARGET_NAME = "external_deps"
ENTRYPOINTS_TARGET_NAME = "entrypoints"

EXTERNAL_DEPENDENCIES_TABLE_KEY = "analytics.external_dependencies"
EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY = "analytics.external_dependency_calls"
EXTERNAL_DEPS_TABLE_KEYS = (
    EXTERNAL_DEPENDENCIES_TABLE_KEY,
    EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
)

ENTRYPOINTS_TABLE_KEY = "analytics.entrypoints"
ENTRYPOINT_TESTS_TABLE_KEY = "analytics.entrypoint_tests"
ENTRYPOINTS_TABLE_KEYS = (ENTRYPOINTS_TABLE_KEY, ENTRYPOINT_TESTS_TABLE_KEY)


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
    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(
            include_graphs=False,
            include_asts=True,
            include_module_map=True,
        ),
    )

    try:
        catalog = registry.require(CatalogProvider).get()
    except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return None

    module_map = dict(registry.require(ModuleMapProvider).module_map)
    features_map: dict[int, FunctionAstFeatures] = {}

    try:
        ast_data = registry.require(AstProvider).get()
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to load function ASTs: %s", exc)
        return None

    for func_ast in ast_data.function_ast_map.values():
        module = catalog.module_for_path(func_ast.rel_path)
        if module is None:
            module = func_ast.rel_path.replace("/", ".").removesuffix(".py")
        module_map[func_ast.rel_path] = module

    return ExternalDependencyInputs(
        catalog_provider=catalog,
        module_map=module_map,
        ast_by_goid=ast_data.function_ast_map,
        features_map=features_map,
        missing_goids=ast_data.missing_function_goids,
    )


@tag_helper(domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME)
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


@tag_compute(domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME)
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
    target = graph.get(EXTERNAL_DEPS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, EXTERNAL_DEPS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXTERNAL_DEPS_TARGET_NAME),
    table_key=value(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME, target_="external_deps__calls_rows"
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(EXTERNAL_DEPENDENCIES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(EXTERNAL_DEPS_TARGET_NAME),
    table_key=value(EXTERNAL_DEPENDENCIES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(EXTERNAL_DEPENDENCIES_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    target_="external_deps__dependencies_rows",
)
def external_deps__dependencies_rows(
    env: BuildEnv,
    m__analytics__external_dependency_calls: MaterializationMetadata,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute rows for analytics.external_dependencies after calls are written.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when the calls write failed.
    """
    meta = DuckDBMaterializationMetadata.from_mapping(
        m__analytics__external_dependency_calls,
        default_table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
    )
    if meta.status != "succeeded":
        return None

    result = compute_external_dependencies_pure(env.gateway, env.snapshot)
    return tuple(result.rows)


@codeintel_target(domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME)
def t__external_deps(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph: TargetRunRecord,
    m__analytics__external_dependency_calls: MaterializationMetadata,
    m__analytics__external_dependencies: MaterializationMetadata,
) -> TargetRunRecord:
    """External library dependency analysis.

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
        executor = NativeTargetExecutor.for_target(env, graph, EXTERNAL_DEPS_TARGET_NAME)
        return executor.fail(
            RuntimeError(f"Upstream call_graph target failed: {t__call_graph.error}")
        )
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=EXTERNAL_DEPS_TARGET_NAME,
        materializations={
            EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY: m__analytics__external_dependency_calls,
            EXTERNAL_DEPENDENCIES_TABLE_KEY: m__analytics__external_dependencies,
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
    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(
            include_graphs=False,
            include_features=True,
            include_module_map=True,
        ),
    )

    try:
        catalog = registry.require(CatalogProvider).get()
    except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return None

    module_map = registry.require(ModuleMapProvider).module_map

    try:
        features_map = registry.require(FeaturesProvider).get()
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to compute function features: %s", exc)
        features_map = {}

    return EntrypointBuildInputs(
        catalog_provider=catalog,
        module_map=dict(module_map),
        features_map=features_map,
    )


@tag_helper(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
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


@tag_compute(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
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
    target = graph.get(ENTRYPOINTS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, ENTRYPOINTS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    if entrypoints_inputs is None:
        return None

    return compute_entrypoints_pure(env.gateway, env.snapshot, entrypoints_inputs)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(ENTRYPOINTS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(ENTRYPOINTS_TARGET_NAME),
    table_key=value(ENTRYPOINTS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(ENTRYPOINTS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics", target=ENTRYPOINTS_TARGET_NAME, target_="entrypoints__entrypoint_rows"
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(ENTRYPOINT_TESTS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(ENTRYPOINTS_TARGET_NAME),
    table_key=value(ENTRYPOINT_TESTS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(ENTRYPOINT_TESTS_TABLE_KEY)),
)
@tag_compute(domain="analytics", target=ENTRYPOINTS_TARGET_NAME, target_="entrypoints__test_rows")
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


@tag_helper(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
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


@tag_helper(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
def entrypoints__materializations(
    m__analytics__entrypoints: MaterializationMetadata,
    m__analytics__entrypoint_tests: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect entrypoints materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Materialization metadata keyed by table key.
    """
    return {
        ENTRYPOINTS_TABLE_KEY: m__analytics__entrypoints,
        ENTRYPOINT_TESTS_TABLE_KEY: m__analytics__entrypoint_tests,
    }


@codeintel_target(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
def t__entrypoints(
    env: BuildEnv,
    graph: TargetGraph,
    entrypoints__upstream_error: str | None,
    entrypoints__materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """External entrypoint detection (HTTP, CLI, etc.).

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if entrypoints__upstream_error is not None:
        executor = NativeTargetExecutor.for_target(env, graph, ENTRYPOINTS_TARGET_NAME)
        return executor.fail(RuntimeError(entrypoints__upstream_error))

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=ENTRYPOINTS_TARGET_NAME,
        materializations=entrypoints__materializations,
    )
