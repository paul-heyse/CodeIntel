"""Profile analytics tables built from stored datasets."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

import duckdb
import pyarrow as pa
from polars.exceptions import PolarsError

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
from codeintel.core.columnar.tabular_adapter import to_table
from codeintel.core.schemas.contracts import arrow_contract_for_table_schema
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBError, DuckDBRelation
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.helpers.table_key import split_table_key

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

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


def _duckdb_connection_from_input(value: TabularInput) -> DuckDBConnection | None:
    """Return a DuckDB connection from a relation-backed input when available.

    Returns
    -------
    DuckDBConnection | None
        Connection from the relation when accessible; otherwise None.
    """
    if not isinstance(value, DuckDBRelation):
        return None
    candidate = getattr(value, "connection", None)
    if isinstance(candidate, DuckDBConnection):
        return candidate
    if callable(candidate):
        try:
            connection = candidate()
        except TypeError:
            return None
        return connection if isinstance(connection, DuckDBConnection) else None
    return None


def _register_tabular_input(
    *,
    con: DuckDBConnection,
    table_key: str,
    value: TabularInput,
) -> None:
    """Register a tabular input as a schema-qualified DuckDB view."""
    try:
        schema, name = split_table_key(table_key)
    except ValueError:
        LOG.warning("profile gateway skipping invalid table key: %s", table_key)
        return
    try:
        table = to_table(value, batch_size=DEFAULT_ARROW_BATCH_SIZE)
    except (DuckDBError, PolarsError, TypeError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("profile gateway failed to coerce %s: %s", table_key, exc)
        return
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    temp_name = f"tmp_{schema}_{name}_{uuid.uuid4().hex}"
    con.register(temp_name, table)
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW {schema}.{name} AS SELECT * FROM {temp_name}"
        )
    finally:
        con.unregister(temp_name)


def _profile_gateway_from_inputs(
    *,
    table_inputs: Mapping[str, TabularInput],
) -> tuple[MinimalStorageGateway, bool]:
    """Create a gateway for profile computations from tabular inputs.

    Returns
    -------
    tuple[MinimalStorageGateway, bool]
        Gateway and a flag indicating connection ownership.
    """
    for value in table_inputs.values():
        connection = _duckdb_connection_from_input(value)
        if connection is not None:
            return MinimalStorageGateway(connection), False
    connection = duckdb.connect(database=":memory:")
    for table_key, value in table_inputs.items():
        _register_tabular_input(con=connection, table_key=table_key, value=value)
    return MinimalStorageGateway(connection), True


def _rows_to_dicts(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def _table_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.Table:
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_contract_for_table_schema(table_schema=schema)
    if not rows:
        return pa.Table.from_batches([], schema=arrow_schema)
    return pa.Table.from_pylist(rows, schema=arrow_schema)


def _function_profile_rows(
    *,
    snapshot: SnapshotRef,
    table_inputs: Mapping[str, TabularInput],
) -> list[dict[str, object]]:
    gateway, owns_connection = _profile_gateway_from_inputs(table_inputs=table_inputs)
    try:
        inputs = compute_function_profile_inputs(cast("StorageGateway", gateway), snapshot)
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
        return _rows_to_dicts(build_function_profile_rows(inputs, views=views))
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("function_profile build failed: %s", exc)
        return []
    finally:
        if owns_connection:
            gateway.con.close()


def _file_profile_rows(
    *,
    snapshot: SnapshotRef,
    table_inputs: Mapping[str, TabularInput],
) -> list[dict[str, object]]:
    gateway, owns_connection = _profile_gateway_from_inputs(table_inputs=table_inputs)
    try:
        inputs = compute_file_profile_inputs(cast("StorageGateway", gateway), snapshot)
        return _rows_to_dicts(build_file_profile_rows(inputs))
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("file_profile build failed: %s", exc)
        return []
    finally:
        if owns_connection:
            gateway.con.close()


def _test_profile_rows(
    *,
    snapshot: SnapshotRef,
    table_inputs: Mapping[str, TabularInput],
) -> list[dict[str, object]]:
    gateway, owns_connection = _profile_gateway_from_inputs(table_inputs=table_inputs)
    try:
        result = build_test_profile_result(cast("StorageGateway", gateway), snapshot)
    except (DuckDBError, RuntimeError, ValueError, TypeError) as exc:
        LOG.warning("test_profile build failed: %s", exc)
        return []
    finally:
        if owns_connection:
            gateway.con.close()
    return _rows_to_dicts(result.rows or [])


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
    """Build analytics.function_profile from stored inputs.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot context.
    q__core__goids
        Loader relation for ``core.goids``.
    q__analytics__function_metrics
        Loader relation for ``analytics.function_metrics``.
    q__analytics__coverage_functions
        Loader relation for ``analytics.coverage_functions``.
    q__analytics__goid_risk_factors
        Loader relation for ``analytics.goid_risk_factors``.

    Returns
    -------
    pa.Table
        Arrow table for analytics.function_profile.
    """
    table_inputs = {
        "core.goids": q__core__goids,
        "analytics.function_metrics": q__analytics__function_metrics,
        "analytics.coverage_functions": q__analytics__coverage_functions,
        "analytics.goid_risk_factors": q__analytics__goid_risk_factors,
    }
    return _table_from_rows(
        FUNCTION_PROFILE_TABLE_KEY,
        _function_profile_rows(snapshot=env.snapshot, table_inputs=table_inputs),
    )


@save_dataset(
    context=PROFILES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FILE_PROFILE_TABLE_KEY),
)
def file_profile__table(
    env: BuildEnv,
    q__core__ast_metrics: TabularInput,
    q__core__modules: TabularInput,
) -> pa.Table:
    """Build analytics.file_profile from stored inputs.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot context.
    q__core__ast_metrics
        Loader relation for ``core.ast_metrics``.
    q__core__modules
        Loader relation for ``core.modules``.

    Returns
    -------
    pa.Table
        Arrow table for analytics.file_profile.
    """
    table_inputs = {
        "core.ast_metrics": q__core__ast_metrics,
        "core.modules": q__core__modules,
    }
    return _table_from_rows(
        FILE_PROFILE_TABLE_KEY,
        _file_profile_rows(snapshot=env.snapshot, table_inputs=table_inputs),
    )


@save_dataset(
    context=TEST_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
)
def test_profile__table(
    env: BuildEnv,
    q__analytics__test_catalog: TabularInput,
    q__analytics__test_coverage_edges: TabularInput,
    q__analytics__test_graph_metrics_tests: TabularInput,
) -> pa.Table:
    """Build analytics.test_profile from stored inputs.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot context.
    q__analytics__test_catalog
        Loader relation for ``analytics.test_catalog``.
    q__analytics__test_coverage_edges
        Loader relation for ``analytics.test_coverage_edges``.
    q__analytics__test_graph_metrics_tests
        Loader relation for ``analytics.test_graph_metrics_tests``.

    Returns
    -------
    pa.Table
        Arrow table for analytics.test_profile.
    """
    table_inputs = {
        "analytics.test_catalog": q__analytics__test_catalog,
        "analytics.test_coverage_edges": q__analytics__test_coverage_edges,
        "analytics.test_graph_metrics_tests": q__analytics__test_graph_metrics_tests,
    }
    return _table_from_rows(
        TEST_PROFILE_TABLE_KEY,
        _test_profile_rows(snapshot=env.snapshot, table_inputs=table_inputs),
    )


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
    """Finalize profiles target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the profiles target.
    """
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
    """Finalize test_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the test_profile target.
    """
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
