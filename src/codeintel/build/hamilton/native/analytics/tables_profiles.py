"""Profile analytics tables built from stored datasets."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.analytics.profiles.files import (
    build_file_profile_rows,
    compute_file_profile_inputs,
)
from codeintel.analytics.profiles.functions import (
    FunctionProfileViews,
    build_function_profile_rows,
    compute_function_profile_inputs,
    join_function_contracts,
    join_function_coverage,
    join_function_docs,
    join_function_effects,
    join_function_history,
    join_function_risk,
    join_function_roles,
    load_function_base_info,
)
from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.testing.profiles.builder import build_test_profile_result
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_duckdb_materialization,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.types import TabularInput
from codeintel.core.schemas.contracts import arrow_contract_for_table_schema
from codeintel.storage.duckdb_types import DuckDBError

LOG = logging.getLogger(__name__)

PROFILES_TARGET_NAME = "profiles"
TEST_PROFILE_TARGET_NAME = "test_profile"

FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
FILE_PROFILE_TABLE_KEY = "analytics.file_profile"
MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
TEST_PROFILE_TABLE_KEY = "analytics.test_profile"

PROFILES_TABLE_KEYS = (
    FUNCTION_PROFILE_TABLE_KEY,
    FILE_PROFILE_TABLE_KEY,
    MODULE_PROFILE_TABLE_KEY,
)

PROFILES_SAVE_CONTEXT = SaverContext(domain="analytics", target=PROFILES_TARGET_NAME)
TEST_PROFILE_SAVE_CONTEXT = SaverContext(domain="analytics", target=TEST_PROFILE_TARGET_NAME)


def _touch_dependencies(*_deps: object) -> None:
    if not _deps:
        return


def _table_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.Table:
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_contract_for_table_schema(table_schema=schema)
    if not rows:
        return pa.Table.from_batches([], schema=arrow_schema)
    return pa.Table.from_pylist(rows, schema=arrow_schema)


def _function_profile_rows(env: BuildEnv) -> list[dict[str, object]]:
    try:
        inputs = compute_function_profile_inputs(env.gateway, env.snapshot)
        views = FunctionProfileViews(
            base_by_func=load_function_base_info(inputs),
            risk_by_func=join_function_risk(inputs),
            coverage_by_func=join_function_coverage(inputs),
            graph_by_func=summarize_graph_for_function_profile(inputs),
            effects_by_func=join_function_effects(inputs),
            contracts_by_func=join_function_contracts(inputs),
            roles_by_func=join_function_roles(inputs),
            docs_by_func=join_function_docs(inputs),
            history_by_func=join_function_history(inputs),
        )
        return list(build_function_profile_rows(inputs, views=views))
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("function_profile build failed: %s", exc)
        return []


def _file_profile_rows(env: BuildEnv) -> list[dict[str, object]]:
    try:
        inputs = compute_file_profile_inputs(env.gateway, env.snapshot)
        return list(build_file_profile_rows(inputs))
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("file_profile build failed: %s", exc)
        return []


def _test_profile_rows(env: BuildEnv) -> list[dict[str, object]]:
    try:
        result = build_test_profile_result(env.gateway, env.snapshot)
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("test_profile build failed: %s", exc)
        return []
    return list(result.rows or [])


@save_dataset(
    context=PROFILES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_PROFILE_TABLE_KEY),
)
def function_profile__table(
    env: BuildEnv,
    q__core__goids: TabularInput,
    q__analytics__function_metrics: TabularInput,
    q__analytics__coverage_functions: TabularInput,
    q__analytics__goid_risk_factors: TabularInput,
) -> pa.Table:
    """Build analytics.function_profile from stored inputs."""
    _touch_dependencies(
        q__core__goids,
        q__analytics__function_metrics,
        q__analytics__coverage_functions,
        q__analytics__goid_risk_factors,
    )
    return _table_from_rows(FUNCTION_PROFILE_TABLE_KEY, _function_profile_rows(env))


@save_dataset(
    context=PROFILES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FILE_PROFILE_TABLE_KEY),
)
def file_profile__table(
    env: BuildEnv,
    q__core__ast_metrics: TabularInput,
    q__core__modules: TabularInput,
) -> pa.Table:
    """Build analytics.file_profile from stored inputs."""
    _touch_dependencies(q__core__ast_metrics, q__core__modules)
    return _table_from_rows(FILE_PROFILE_TABLE_KEY, _file_profile_rows(env))


@save_dataset(
    context=TEST_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
)
def test_profile__table(env: BuildEnv) -> pa.Table:
    """Build analytics.test_profile from stored inputs."""
    return _table_from_rows(TEST_PROFILE_TABLE_KEY, _test_profile_rows(env))


profiles__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=PROFILES_TARGET_NAME,
    table_keys=PROFILES_TABLE_KEYS,
    node_name="profiles__table_materializations",
)


@codeintel_target(domain="analytics", target=PROFILES_TARGET_NAME)
def t__profiles(
    env: BuildEnv,
    catalog: DagCatalog,
    profiles__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize profiles target run record."""
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=PROFILES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=profiles__table_materializations,
    )


@codeintel_target(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
def t__test_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__test_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize test_profile target run record."""
    return record_from_duckdb_materialization(
        env=env,
        catalog=catalog,
        target_name=TEST_PROFILE_TARGET_NAME,
        expected_table_key=TEST_PROFILE_TABLE_KEY,
        materialization=m__analytics__test_profile,
    )


__all__ = [
    "FILE_PROFILE_TABLE_KEY",
    "FUNCTION_PROFILE_TABLE_KEY",
    "MODULE_PROFILE_TABLE_KEY",
    "PROFILES_TABLE_KEYS",
    "PROFILES_TARGET_NAME",
    "TEST_PROFILE_TABLE_KEY",
    "TEST_PROFILE_TARGET_NAME",
    "file_profile__table",
    "function_profile__table",
    "profiles__table_materializations",
    "t__profiles",
    "t__test_profile",
    "test_profile__table",
]
