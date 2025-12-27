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

from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.semantic_roles.core import SemanticRolesResult, build_semantic_roles_rows
from codeintel.analytics.testing.profiles.builder import build_test_profile_result
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.analytics.testing.profiles.builder import TestProfileBuildResult

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, SemanticRolesResult)

SEMANTIC_ROLES_TARGET_NAME = "semantic_roles"
TEST_PROFILE_TARGET_NAME = "test_profile"

SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY = "analytics.semantic_roles_functions"
SEMANTIC_ROLES_MODULES_TABLE_KEY = "analytics.semantic_roles_modules"
SEMANTIC_ROLES_TABLE_KEYS = (
    SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
    SEMANTIC_ROLES_MODULES_TABLE_KEY,
)

TEST_PROFILE_TABLE_KEY = "analytics.test_profile"
TEST_PROFILE_TABLE_KEYS = (TEST_PROFILE_TABLE_KEY,)
SEMANTIC_ROLES_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
)
TEST_PROFILE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=TEST_PROFILE_TARGET_NAME,
)






@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for classification nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@tag_compute(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
def t__semantic_roles__compute(
    env: BuildEnv,
    gateway: StorageGateway,
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
    gateway
        Storage gateway for analytics queries.
    t__modules
        Upstream modules target result (for dependency).
    t__function_ast_features
        Upstream function_ast_features target result (for dependency).
        Skip flag derived from manifest-based input hash evaluation.
        Skip flag derived from manifest-based input hash evaluation.

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


    registry = build_registry(
        gateway=gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        try:
            catalog = registry.require(CatalogProvider)
            module_by_path = dict(catalog.catalog().module_by_path)
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            module_by_path = {}

        # AST and features maps (currently not loaded from upstream)
        ast_map: dict[int, FunctionAst] = {}
        features_map: dict[int, FunctionAstFeatures] = {}

        # Compute semantic roles (pure compute - no persistence)
        return build_semantic_roles_rows(
            gateway,
            env.snapshot,
            module_by_path=module_by_path,
            ast_map=ast_map,
            features_map=features_map,
        )

    except Exception:
        log.exception("Semantic roles computation failed")
        return None


@save_rows(
    context=SEMANTIC_ROLES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY),
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


@save_rows(
    context=SEMANTIC_ROLES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY),
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
    catalog: DagCatalog,
    semantic_roles__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Classify semantic roles (handler, utility, etc.).

    Combines materialization results from both table writes into a
    single TargetRunRecord for the semantic_roles target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    semantic_roles__table_materializations
        Materialization results for semantic_roles tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SEMANTIC_ROLES_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=semantic_roles__table_materializations,
    )


semantic_roles__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    table_keys=SEMANTIC_ROLES_TABLE_KEYS,
)


__all__ = [
    "SemanticRolesResult",
    "TestProfileComputeResult",
    "semantic_roles__functions_rows",
    "semantic_roles__modules_rows",
    "semantic_roles__table_materializations",
    "t__semantic_roles",
    "t__semantic_roles__compute",
    "t__test_profile",
    "t__test_profile__compute",
    "test_profile__rows",
    "test_profile__table_materializations",
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
    gateway: StorageGateway,
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
        build_result = build_test_profile_result(gateway, env.snapshot)
        return TestProfileComputeResult(result=build_result)
    except Exception as exc:
        log.exception("Test profile computation failed")
        return TestProfileComputeResult(result=None, error=str(exc))


@save_rows(
    context=TEST_PROFILE_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
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
    catalog: DagCatalog,
    t__test_profile__compute: TestProfileComputeResult,
    test_profile__table_materializations: dict[str, MaterializationResult],
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
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__test_profile__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=TEST_PROFILE_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=test_profile__table_materializations,
    )


test_profile__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=TEST_PROFILE_TARGET_NAME,
    table_keys=TEST_PROFILE_TABLE_KEYS,
)
