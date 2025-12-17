"""Native Hamilton implementation for semantic_roles target.

This module provides the Hamilton native nodes for semantic roles computation
with DAG-visible I/O via SaveToDecorator/DuckDBRowsSaver:

- `t__semantic_roles__compute`: Pure compute node returning role rows
- `semantic_roles__functions_rows`: Extract function rows for materialization
- `semantic_roles__modules_rows`: Extract module rows for materialization
- `t__semantic_roles`: Materialize node combining table writes

The compute node calls `build_semantic_roles_rows` which returns pure rows
without persistence. Persistence is handled by DuckDBRowsSaver via SaveToDecorator,
making I/O visible in the Hamilton DAG for caching and observability.

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.semantic_roles import SemanticRolesResult, build_semantic_roles_rows
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
    from codeintel.analytics.parsing.ast_cache import FunctionAst

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, SemanticRolesResult)

# Column definitions - must match the bulk_insert order in core.py
SEMANTIC_ROLES_FUNCTIONS_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "role",
    "framework",
    "role_confidence",
    "role_sources_json",
    "created_at",
)

SEMANTIC_ROLES_MODULES_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "module",
    "role",
    "role_confidence",
    "role_sources_json",
    "created_at",
)


@tag(domain="analytics", target="semantic_roles", node_type="compute")
def t__semantic_roles__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    t__function_ast_features: TargetRunRecord,
) -> SemanticRolesResult | None:
    """Compute semantic roles for functions and modules.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for skip detection.
    t__modules
        Upstream modules target result (for dependency).
    t__function_ast_features
        Upstream function_ast_features target result (for dependency).

    Returns
    -------
    SemanticRolesResult | None
        Result containing function and module rows, or None if skipped.

    Notes
    -----
    The classifications include:
    - Semantic role (producer, consumer, transformer, etc.)
    - Data flow patterns
    - Behavioral classification
    """
    if t__modules.status != "succeeded":
        log.warning("Upstream modules target failed: %s", t__modules.error)
        return None

    if t__function_ast_features.status != "succeeded":
        log.warning("Upstream function_ast_features target failed: %s", t__function_ast_features.error)
        return None

    target = graph.get("semantic_roles")
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

    try:
        # Load catalog for module info
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
            module_by_path = dict(catalog.catalog().module_by_path)
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            module_by_path = {}

        # AST and features maps (currently not loaded from upstream)
        ast_map: dict[int, FunctionAst] = {}
        features_map: dict[int, FunctionAstFeatures] = {}

        # Compute semantic roles (pure compute - no persistence)
        return build_semantic_roles_rows(
            env.gateway,
            env.snapshot,
            module_by_path=module_by_path,
            ast_map=ast_map,
            features_map=features_map,
        )

    except Exception:
        log.exception("Semantic roles computation failed")
        return None


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.semantic_roles_functions"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("semantic_roles"),
    table_key=value("analytics.semantic_roles_functions"),
    columns=value(SEMANTIC_ROLES_FUNCTIONS_COLS),
)
@tag(
    domain="analytics",
    target="semantic_roles",
    node_type="compute",
    target_="semantic_roles__functions_rows",
)
def semantic_roles__functions_rows(
    t__semantic_roles__compute: SemanticRolesResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.semantic_roles_functions table.

    Parameters
    ----------
    t__semantic_roles__compute
        Computed semantic roles result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.semantic_roles_functions table, or None if
        compute result is None.
    """
    if t__semantic_roles__compute is None:
        return None
    return tuple(t__semantic_roles__compute.function_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.semantic_roles_modules"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("semantic_roles"),
    table_key=value("analytics.semantic_roles_modules"),
    columns=value(SEMANTIC_ROLES_MODULES_COLS),
)
@tag(
    domain="analytics",
    target="semantic_roles",
    node_type="compute",
    target_="semantic_roles__modules_rows",
)
def semantic_roles__modules_rows(
    t__semantic_roles__compute: SemanticRolesResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.semantic_roles_modules table.

    Parameters
    ----------
    t__semantic_roles__compute
        Computed semantic roles result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.semantic_roles_modules table, or None if
        compute result is None.
    """
    if t__semantic_roles__compute is None:
        return None
    return tuple(t__semantic_roles__compute.module_rows)


@tag(domain="analytics", target="semantic_roles", node_type="materialize")
def t__semantic_roles(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__semantic_roles_functions: dict[str, Any],
    m__analytics__semantic_roles_modules: dict[str, Any],
) -> TargetRunRecord:
    """Materialize semantic roles target.

    Combines materialization metadata from both table writes into a
    single TargetRunRecord for the semantic_roles target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__semantic_roles_functions
        Materialization metadata for semantic_roles_functions table.
    m__analytics__semantic_roles_modules
        Materialization metadata for semantic_roles_modules table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="semantic_roles",
        materializations={
            "analytics.semantic_roles_functions": m__analytics__semantic_roles_functions,
            "analytics.semantic_roles_modules": m__analytics__semantic_roles_modules,
        },
    )


__all__ = [
    "SEMANTIC_ROLES_FUNCTIONS_COLS",
    "SEMANTIC_ROLES_MODULES_COLS",
    "SemanticRolesResult",
    "semantic_roles__functions_rows",
    "semantic_roles__modules_rows",
    "t__semantic_roles",
    "t__semantic_roles__compute",
]
