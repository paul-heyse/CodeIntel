"""Native Hamilton implementations for code classification targets.

This module consolidates targets that classify code and tests:

- ``semantic_roles``: Function/module semantic role classification.
- ``test_profile``: Per-test profiling with coverage context.

Both targets use DAG-visible I/O via Hamilton saver nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.semantic_roles import SemanticRolesResult, build_semantic_roles_rows
from codeintel.analytics.testing.profiles.builder import build_test_profile_result
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import make_output_target
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.analytics.testing.profiles.builder import TestProfileBuildResult

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, SemanticRolesResult)

TARGET_SPECS = (
    make_output_target(
        name="semantic_roles",
        module="analytics",
        description="Semantic role classification (handler, utility, etc.).",
        table_keys=(
            "analytics.semantic_roles_functions",
            "analytics.semantic_roles_modules",
        ),
    ),
    make_output_target(
        name="test_profile",
        module="analytics",
        description="Per-test profile with coverage and characteristics.",
        table_keys=("analytics.test_profile",),
    ),
)

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
        log.warning(
            "Upstream function_ast_features target failed: %s", t__function_ast_features.error
        )
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
    "TEST_PROFILE_COLS",
    "SemanticRolesResult",
    "TestProfileComputeResult",
    "semantic_roles__functions_rows",
    "semantic_roles__modules_rows",
    "t__semantic_roles",
    "t__semantic_roles__compute",
    "t__test_profile",
    "t__test_profile__compute",
    "test_profile__rows",
]


# ---------------------------------------------------------------------------
# test_profile target
# ---------------------------------------------------------------------------


TEST_PROFILE_COLS = (
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "urn",
    "rel_path",
    "module",
    "qualname",
    "language",
    "kind",
    "status",
    "duration_ms",
    "markers",
    "flaky",
    "last_run_at",
    "functions_covered",
    "functions_covered_count",
    "primary_function_goids",
    "subsystems_covered",
    "subsystems_covered_count",
    "primary_subsystem_id",
    "assert_count",
    "raise_count",
    "uses_parametrize",
    "uses_fixtures",
    "io_bound",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "flakiness_score",
    "importance_score",
    "notes",
    "tg_degree",
    "tg_weighted_degree",
    "tg_proj_degree",
    "tg_proj_weight",
    "tg_proj_clustering",
    "tg_proj_betweenness",
    "created_at",
)


@dataclass(frozen=True)
class TestProfileComputeResult:
    """Result from test profile computation."""

    result: TestProfileBuildResult | None
    error: str | None = None


def _test_profile_row_to_tuple(
    row: Mapping[str, object], cols: tuple[str, ...]
) -> tuple[object, ...]:
    return tuple(row.get(col) for col in cols)


@tag(domain="analytics", target="test_profile", node_type="compute")
def t__test_profile__compute(
    env: BuildEnv,
    t__coverage_test_edges: TargetRunRecord,
) -> TestProfileComputeResult:
    """Build per-test profiles with coverage and subsystem context.

    Returns
    -------
    TestProfileComputeResult
        Computed profile rows and optional error message.
    """
    if t__coverage_test_edges.status != "succeeded":
        return TestProfileComputeResult(
            result=None,
            error=(f"Upstream coverage_test_edges target failed: {t__coverage_test_edges.error}"),
        )

    try:
        build_result = build_test_profile_result(env.gateway, env.snapshot)
        return TestProfileComputeResult(result=build_result)
    except Exception as exc:
        log.exception("Test profile computation failed")
        return TestProfileComputeResult(result=None, error=str(exc))


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_profile"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_profile"),
    table_key=value("analytics.test_profile"),
    columns=value(TEST_PROFILE_COLS),
)
@tag(domain="analytics", target="test_profile", node_type="compute", target_="test_profile__rows")
def test_profile__rows(
    t__test_profile__compute: TestProfileComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_profile table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when computation produced no rows.
    """
    if t__test_profile__compute.result is None:
        return None
    if t__test_profile__compute.result.rows is None:
        return None
    return tuple(
        _test_profile_row_to_tuple(row, TEST_PROFILE_COLS)
        for row in t__test_profile__compute.result.rows
    )


@tag(domain="analytics", target="test_profile", node_type="materialize")
def t__test_profile(
    env: BuildEnv,
    graph: TargetGraph,
    t__test_profile__compute: TestProfileComputeResult,
    m__analytics__test_profile: dict[str, Any],
) -> TargetRunRecord:
    """Materialize test profile target.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if t__test_profile__compute.error:
        return TargetRunRecord(
            target="test_profile",
            plugin_name="native:test_profile",
            status="failed",
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=t__test_profile__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="test_profile",
        expected_table_key="analytics.test_profile",
        materialization=m__analytics__test_profile,
    )
