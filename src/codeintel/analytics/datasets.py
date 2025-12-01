"""Analytics-facing dataset contracts and row insertion helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    BehavioralCoverageRowModel,
    DatasetContract,
    FunctionAstFeaturesRow,
    FunctionMetricsRow,
    FunctionProfileRowModel,
    FunctionTypesRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    ProfileRowModel,
    TableSchema,
    behavioral_coverage_row_to_tuple,
    function_ast_features_row_to_tuple,
    function_metrics_row_to_tuple,
    function_profile_row_to_tuple,
    function_types_row_to_tuple,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.ingestion.common import run_batch
from codeintel.storage.datasets import DatasetRegistry, load_dataset_registry
from codeintel.storage.gateway import StorageGateway

# Aliases for backward compatibility
FunctionGraphMetricsRow = GraphMetricsFunctionsRow
ModuleGraphMetricsRow = GraphMetricsModulesRow
FunctionGraphMetricsExtRow = GraphMetricsFunctionsExtRow
ModuleGraphMetricsExtRow = GraphMetricsModulesExtRow
function_graph_metrics_row_to_tuple = graph_metrics_functions_row_to_tuple
module_graph_metrics_row_to_tuple = graph_metrics_modules_row_to_tuple
function_graph_metrics_ext_row_to_tuple = graph_metrics_functions_ext_row_to_tuple
module_graph_metrics_ext_row_to_tuple = graph_metrics_modules_ext_row_to_tuple
TestProfileRow = ProfileRowModel
BehavioralCoverageRow = BehavioralCoverageRowModel

type RowType = Mapping[str, object]
RowT = TypeVar("RowT", bound=RowType)
ToTuple = Callable[[RowT], tuple[object, ...]]

REPO_COMMIT_ARITY = 2


def _build_delete_sql_by_table() -> dict[str, str]:
    """
    Build a mapping of table keys to DELETE SQL statements from contracts.

    Automatically generates delete SQL for all datasets that have both 'repo'
    and 'commit' columns in their schema, enabling scoped deletion by
    repository and commit.

    Returns
    -------
    dict[str, str]
        Mapping from table key to DELETE SQL statement.
    """
    result: dict[str, str] = {}
    for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items():
        if contract.schema is None or contract.is_view:
            continue
        col_names = contract.schema.column_names()
        if "repo" in col_names and "commit" in col_names:
            # S608: table_key is derived from trusted contract definitions, not user input
            result[table_key] = f"DELETE FROM {table_key} WHERE repo = ? AND commit = ?"  # noqa: S608
    return result


DELETE_SQL_BY_TABLE: dict[str, str] = _build_delete_sql_by_table()


@dataclass(frozen=True)
class DeleteScope:
    """Optional delete configuration applied before insert."""

    params: Sequence[object]
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AnalyticsDatasetContract[RowT: RowType]:
    """
    Analytics-facing dataset contract for a DuckDB table or view.

    Attributes
    ----------
    name
        Logical dataset name, e.g. "analytics.function_metrics".
    table_key
        Fully-qualified DuckDB identifier (usually the same as name).
    schema
        TableSchema entry for the dataset, when available.
    row_type
        Typed row model (usually a TypedDict).
    to_tuple
        Serializer from row dict -> tuple in INSERT column order.
    primary_key
        Primary key columns (if known).
    indexes
        Index column definitions for the dataset.
    dataset_meta
        Dataset registry entry when present.
    """

    name: str
    table_key: str
    schema: TableSchema | None
    row_type: type[RowT]
    to_tuple: ToTuple
    primary_key: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...]
    dataset_meta: DatasetContract | None = None


def _build_registry(gateway: StorageGateway) -> DatasetRegistry:
    return load_dataset_registry(gateway.con)


def build_analytics_dataset_contracts(
    gateway: StorageGateway,
) -> dict[str, AnalyticsDatasetContract[RowType]]:
    """
    Build dataset contracts for analytics tables using registry metadata.

    Returns
    -------
    dict[str, AnalyticsDatasetContract[RowType]]
        Contracts keyed by dataset name.
    """
    registry = _build_registry(gateway)

    def _contract(
        name: str,
        *,
        row_type: type[RowT],
        to_tuple: ToTuple,
    ) -> AnalyticsDatasetContract[RowT]:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY[name]
        schema = contract.schema
        primary_key = schema.primary_key if schema is not None else ()
        indexes = tuple(index.columns for index in schema.indexes) if schema else ()
        dataset_meta = registry.by_table_key.get(contract.table_key)
        return AnalyticsDatasetContract(
            name=contract.table_key,
            table_key=contract.table_key,
            schema=schema,
            row_type=row_type,
            to_tuple=to_tuple,
            primary_key=primary_key,
            indexes=indexes,
            dataset_meta=dataset_meta,
        )

    return {
        "analytics.function_metrics": _contract(
            "analytics.function_metrics",
            row_type=FunctionMetricsRow,
            to_tuple=function_metrics_row_to_tuple,
        ),
        "analytics.function_types": _contract(
            "analytics.function_types",
            row_type=FunctionTypesRow,
            to_tuple=function_types_row_to_tuple,
        ),
        "analytics.function_profile": _contract(
            "analytics.function_profile",
            row_type=FunctionProfileRowModel,
            to_tuple=function_profile_row_to_tuple,
        ),
        "analytics.function_ast_features": _contract(
            "analytics.function_ast_features",
            row_type=FunctionAstFeaturesRow,
            to_tuple=function_ast_features_row_to_tuple,
        ),
        "analytics.test_profile": _contract(
            "analytics.test_profile",
            row_type=TestProfileRow,
            to_tuple=serialize_test_profile_row,
        ),
        "analytics.behavioral_coverage": _contract(
            "analytics.behavioral_coverage",
            row_type=BehavioralCoverageRow,
            to_tuple=behavioral_coverage_row_to_tuple,
        ),
        "analytics.graph_metrics_functions": _contract(
            "analytics.graph_metrics_functions",
            row_type=FunctionGraphMetricsRow,
            to_tuple=function_graph_metrics_row_to_tuple,
        ),
        "analytics.graph_metrics_modules": _contract(
            "analytics.graph_metrics_modules",
            row_type=ModuleGraphMetricsRow,
            to_tuple=module_graph_metrics_row_to_tuple,
        ),
        "analytics.graph_metrics_functions_ext": _contract(
            "analytics.graph_metrics_functions_ext",
            row_type=FunctionGraphMetricsExtRow,
            to_tuple=function_graph_metrics_ext_row_to_tuple,
        ),
        "analytics.graph_metrics_modules_ext": _contract(
            "analytics.graph_metrics_modules_ext",
            row_type=ModuleGraphMetricsExtRow,
            to_tuple=module_graph_metrics_ext_row_to_tuple,
        ),
    }


def get_analytics_dataset_contract(
    gateway: StorageGateway,
    name: str,
) -> AnalyticsDatasetContract[RowType]:
    """
    Return the dataset contract for a named analytics dataset.

    Returns
    -------
    AnalyticsDatasetContract[RowType]
        Contract for the requested dataset.

    Raises
    ------
    KeyError
        If the dataset name is unknown.
    """
    contracts = build_analytics_dataset_contracts(gateway)
    if name not in contracts:
        message = f"Unknown analytics dataset: {name}"
        raise KeyError(message)
    return contracts[name]


def get_function_ast_features_contract(
    gateway: StorageGateway,
) -> AnalyticsDatasetContract[FunctionAstFeaturesRow]:
    """
    Return the dataset contract for function AST features.

    Returns
    -------
    AnalyticsDatasetContract[FunctionAstFeaturesRow]
        Contract describing analytics.function_ast_features.
    """
    contract = get_analytics_dataset_contract(gateway, "analytics.function_ast_features")
    return cast("AnalyticsDatasetContract[FunctionAstFeaturesRow]", contract)


def insert_analytics_rows(
    gateway: StorageGateway,
    contract: AnalyticsDatasetContract[RowT],
    rows: Sequence[RowT],
    *,
    delete_scope: DeleteScope | None = None,
    scope: str | None = None,
) -> None:
    """
    Insert rows for a dataset contract using run_batch.

    Raises
    ------
    ValueError
        If delete columns cannot be determined for the requested dataset.
    """
    schema_columns = (
        tuple(column.name for column in contract.schema.columns) if contract.schema else ()
    )
    if delete_scope is not None:
        columns_for_delete = delete_scope.columns
        if (
            columns_for_delete is None
            and contract.primary_key
            and len(delete_scope.params) == len(contract.primary_key)
        ):
            columns_for_delete = contract.primary_key
        elif (
            columns_for_delete is None
            and len(delete_scope.params) == REPO_COMMIT_ARITY
            and "repo" in schema_columns
            and "commit" in schema_columns
        ):
            columns_for_delete = ("repo", "commit")

        if columns_for_delete is None:
            message = f"Delete columns unknown for {contract.table_key}"
            raise ValueError(message)

        delete_sql = DELETE_SQL_BY_TABLE.get(contract.table_key)
        if delete_sql is None:
            message = f"Unsupported delete target: {contract.table_key}"
            raise ValueError(message)
        gateway.con.execute(delete_sql, list(delete_scope.params))

    if rows:
        tuple_rows = [contract.to_tuple(row) for row in rows]
        run_batch(
            gateway,
            contract.table_key,
            tuple_rows,
            delete_params=None,
            scope=scope,
        )
