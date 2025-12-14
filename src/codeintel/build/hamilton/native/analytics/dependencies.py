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
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

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
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.core.catalog import CatalogService
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph


log = logging.getLogger(__name__)


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
def t__external_deps__compute_calls(env: BuildEnv) -> DependencyCallsResult:
    """Compute external dependency calls for all functions in the snapshot.

    This is a pure compute node with no side effects. It loads function
    ASTs and analyzes them for external dependency usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    DependencyCallsResult
        Container with rows for external_dependency_calls table.

    Notes
    -----
    The analysis identifies:
    - Library calls matching dependency patterns
    - Usage modes (read, write, admin, etc.)
    - Evidence with code snippets
    """
    inputs = _build_inputs(env)
    if inputs is None:
        return DependencyCallsResult(rows=())

    return compute_dependency_calls_pure(
        env.gateway,
        env.snapshot,
        inputs,
    )


@tag(domain="analytics", target="external_deps", node_type="materialize")
def t__external_deps(
    env: BuildEnv,
    graph: TargetGraph,
    t__external_deps__compute_calls: DependencyCallsResult,
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
    executor = NativeTargetExecutor.for_target(env, graph, "external_deps")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure tables exist
        backend = DuckDBPolicyBackend(env.gateway)
        backend.ensure_table("analytics.external_dependency_calls")
        backend.ensure_table("analytics.external_dependencies")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="external_deps",
            input_hash=executor.input_hash,
        )

        row_counts: dict[str, int] = {}

        # Step 1: Materialize dependency calls first
        calls_ref = materialize_rows(
            ctx,
            "analytics.external_dependency_calls",
            t__external_deps__compute_calls.rows,
            EXTERNAL_DEPENDENCY_CALLS_COLS,
        )
        row_counts["analytics.external_dependency_calls"] = calls_ref.row_count or 0

        # Step 2: Compute and materialize aggregated dependencies
        # This MUST happen after calls are written since it reads from the table
        deps_result = compute_external_dependencies_pure(env.gateway, env.snapshot)
        deps_ref = materialize_rows(
            ctx,
            "analytics.external_dependencies",
            deps_result.rows,
            EXTERNAL_DEPENDENCIES_COLS,
        )
        row_counts["analytics.external_dependencies"] = deps_ref.row_count or 0

        return row_counts

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__external_deps",
    "t__external_deps__compute_calls",
]
