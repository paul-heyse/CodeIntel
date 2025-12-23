"""Native Hamilton implementations for code classification targets.

This module consolidates targets that classify code and tests:

- ``semantic_roles``: Function/module semantic role classification.
- ``test_profile``: Per-test profiling with coverage context.

Both targets use DAG-visible I/O via Hamilton saver nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.semantic_roles.core import SemanticRolesResult, build_semantic_roles_rows
from codeintel.analytics.testing.profiles.builder import build_test_profile_result
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.analytics.testing.profiles.builder import TestProfileBuildResult

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, SemanticRolesResult)

SEMANTIC_ROLES_TARGET_NAME = "semantic_roles"
TEST_PROFILE_TARGET_NAME = "test_profile"

SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY = "analytics.semantic_roles_functions"
SEMANTIC_ROLES_MODULES_TABLE_KEY = "analytics.semantic_roles_modules"

TEST_PROFILE_TABLE_KEY = "analytics.test_profile"


@tag_compute(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
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

    target = graph.get(SEMANTIC_ROLES_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, SEMANTIC_ROLES_TARGET_NAME)
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

    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        try:
            catalog = registry.require(CatalogProvider).get()
            module_by_path = dict(catalog.catalog().module_by_path)
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SEMANTIC_ROLES_TARGET_NAME),
    table_key=value(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SEMANTIC_ROLES_MODULES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SEMANTIC_ROLES_TARGET_NAME),
    table_key=value(SEMANTIC_ROLES_MODULES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SEMANTIC_ROLES_MODULES_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
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


@codeintel_target(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
def t__semantic_roles(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__semantic_roles_functions: MaterializationMetadata,
    m__analytics__semantic_roles_modules: MaterializationMetadata,
) -> TargetRunRecord:
    """Classify semantic roles (handler, utility, etc.).

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
        target_name=SEMANTIC_ROLES_TARGET_NAME,
        materializations={
            SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY: m__analytics__semantic_roles_functions,
            SEMANTIC_ROLES_MODULES_TABLE_KEY: m__analytics__semantic_roles_modules,
        },
    )


__all__ = [
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


@dataclass(frozen=True)
class TestProfileComputeResult:
    """Result from test profile computation."""

    result: TestProfileBuildResult | None
    error: str | None = None


@tag_compute(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(TEST_PROFILE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TEST_PROFILE_TARGET_NAME),
    table_key=value(TEST_PROFILE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(TEST_PROFILE_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=TEST_PROFILE_TARGET_NAME,
    target_="test_profile__rows",
)
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
        row_to_tuple(TEST_PROFILE_TABLE_KEY, row) for row in t__test_profile__compute.result.rows
    )


@codeintel_target(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
def t__test_profile(
    env: BuildEnv,
    graph: TargetGraph,
    t__test_profile__compute: TestProfileComputeResult,
    m__analytics__test_profile: MaterializationMetadata,
) -> TargetRunRecord:
    """Build per-test profiles with coverage and characteristics.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if t__test_profile__compute.error:
        options_hash = options_hash_for_target(env, TEST_PROFILE_TARGET_NAME)
        return TargetRunRecord(
            target=TEST_PROFILE_TARGET_NAME,
            plugin_name=f"native:{TEST_PROFILE_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__test_profile__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=TEST_PROFILE_TARGET_NAME,
        expected_table_key=TEST_PROFILE_TABLE_KEY,
        materialization=m__analytics__test_profile,
    )
