"""Analytics-facing dataset contracts and row insertion helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import pandas as pd
from sqlglot import exp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.analytics.utilities.persistence import DeleteScope
    from codeintel.config.datasets import (
        DatasetContract,
        TableSchema,
    )
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway import StorageGateway

from codeintel.config.datasets import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    BehavioralCoverageRowModel,
    FunctionAstFeaturesRow,
    FunctionContractsRow,
    FunctionEffectsRow,
    FunctionMetricsRow,
    FunctionProfileRowModel,
    FunctionTypesRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    ProfileRowModel,
    behavioral_coverage_row_to_tuple,
    function_ast_features_row_to_tuple,
    function_contracts_row_to_tuple,
    function_effects_row_to_tuple,
    function_metrics_row_to_tuple,
    function_profile_row_to_tuple,
    function_types_row_to_tuple,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.config.datasets.validation import validate_df
from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

type RowType = Mapping[str, object]
RowT = TypeVar("RowT", bound=RowType)
ToTuple = Callable[[RowT], tuple[object, ...]]

REPO_COMMIT_ARITY = 2


def _table_supports_snapshot_delete(table_key: str) -> bool:
    """Check if a table supports repo/commit scoped deletion.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., 'analytics.function_metrics').

    Returns
    -------
    bool
        True if the table has repo and commit columns.
    """
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    if contract is None or contract.schema is None or contract.is_view:
        return False
    col_names = contract.schema.column_names()
    return "repo" in col_names and "commit" in col_names


def _delete_sql_for_table(table_key: str) -> str:
    schema, table = table_key.split(".", 1)
    table_expr = exp.Table(this=exp.to_identifier(table), db=exp.to_identifier(schema))
    condition = exp.and_(
        exp.EQ(this=exp.column("repo"), expression=exp.Parameter()),
        exp.EQ(this=exp.column("commit"), expression=exp.Parameter()),
    )
    statement = exp.Delete(this=table_expr, where=condition)
    return statement.sql(dialect="duckdb")


DELETE_SQL_BY_TABLE: dict[str, str] = {
    table_key: _delete_sql_for_table(table_key)
    for table_key in DATASET_CONTRACTS_BY_TABLE_KEY
    if _table_supports_snapshot_delete(table_key)
}


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
        "analytics.function_effects": _contract(
            "analytics.function_effects",
            row_type=FunctionEffectsRow,
            to_tuple=function_effects_row_to_tuple,
        ),
        "analytics.function_contracts": _contract(
            "analytics.function_contracts",
            row_type=FunctionContractsRow,
            to_tuple=function_contracts_row_to_tuple,
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
            row_type=ProfileRowModel,
            to_tuple=serialize_test_profile_row,
        ),
        "analytics.behavioral_coverage": _contract(
            "analytics.behavioral_coverage",
            row_type=BehavioralCoverageRowModel,
            to_tuple=behavioral_coverage_row_to_tuple,
        ),
        "analytics.graph_metrics_functions": _contract(
            "analytics.graph_metrics_functions",
            row_type=GraphMetricsFunctionsRow,
            to_tuple=graph_metrics_functions_row_to_tuple,
        ),
        "analytics.graph_metrics_modules": _contract(
            "analytics.graph_metrics_modules",
            row_type=GraphMetricsModulesRow,
            to_tuple=graph_metrics_modules_row_to_tuple,
        ),
        "analytics.graph_metrics_functions_ext": _contract(
            "analytics.graph_metrics_functions_ext",
            row_type=GraphMetricsFunctionsExtRow,
            to_tuple=graph_metrics_functions_ext_row_to_tuple,
        ),
        "analytics.graph_metrics_modules_ext": _contract(
            "analytics.graph_metrics_modules_ext",
            row_type=GraphMetricsModulesExtRow,
            to_tuple=graph_metrics_modules_ext_row_to_tuple,
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
    """Insert rows for a dataset contract using DuckDBPolicyBackend.

    Deletions are routed through DuckDBPolicyBackend for centralized SQL
    generation.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    contract
        Dataset contract describing the target table.
    rows
        Rows to insert.
    delete_scope
        Optional deletion scope for clearing existing data.
    scope
        Optional scope label for logging.

    Raises
    ------
    ValueError
        If delete columns cannot be determined for the requested dataset.
    """
    _ = scope
    backend = DuckDBPolicyBackend(gateway)
    backend.ensure_table(contract.table_key)

    if delete_scope is not None:
        if not _table_supports_snapshot_delete(contract.table_key):
            message = f"Unsupported delete target: {contract.table_key}"
            raise ValueError(message)

        backend.delete_for_snapshot(
            contract.table_key,
            repo=delete_scope.repo,
            commit=delete_scope.commit,
        )

    if rows:
        tuple_rows = [contract.to_tuple(row) for row in rows]
        backend.bulk_insert(contract.table_key, tuple_rows)


def validate_contract_rows(
    table_key: str, rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """
    Validate rows for a dataset using Pandera and return normalized dicts.

    Missing values are normalized to ``None`` for safe DuckDB insertion.

    Returns
    -------
    list[dict[str, object]]
        Pandera-validated rows coerced to serializable dictionaries.
    """
    if not rows:
        return []
    df = validate_df(table_key, pd.DataFrame(rows))
    normalized = df.where(pd.notna(df), None)
    return normalized.to_dict(orient="records")


__all__ = [
    "AnalyticsDatasetContract",
    "build_analytics_dataset_contracts",
    "get_analytics_dataset_contract",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "validate_contract_rows",
    "validate_tuple_rows",
]


def validate_tuple_rows(
    table_key: str,
    rows: Sequence[Mapping[str, object] | Sequence[object]],
    *,
    columns: Sequence[str] | None = None,
    schema: TableSchema | None = None,
) -> list[tuple[object, ...]]:
    """
    Validate tuple rows for a dataset and return normalized tuples.

    Parameters
    ----------
    table_key
        Fully qualified dataset key.
    rows
        Iterable of rows as mappings or positional sequences.
    columns
        Column order corresponding to the tuples.
    schema
        Optional TableSchema used to derive column order.

    Returns
    -------
    list[tuple[object, ...]]
        Pandera-validated rows with ``None`` for missing values.

    Raises
    ------
    ValueError
        If both ``columns`` and ``schema`` are provided or if column names cannot be
        determined.
    """
    if not rows:
        return []
    if columns is not None and schema is not None:
        message = "Specify either schema or columns, not both"
        raise ValueError(message)
    column_names = tuple(columns or (schema.column_names() if schema else ()))
    if not column_names:
        message = f"Column names required to validate rows for {table_key}"
        raise ValueError(message)

    first = rows[0]
    columns_index = pd.Index(column_names)
    if isinstance(first, Mapping):
        mapping_rows = cast("Sequence[Mapping[str, object]]", rows)
        df = pd.DataFrame(mapping_rows, columns=columns_index)
    else:
        tuple_rows = cast("Sequence[Sequence[object]]", rows)
        df = pd.DataFrame(tuple_rows, columns=columns_index)

    validated = validate_df(table_key, df)
    normalized = validated.where(pd.notna(validated), None)
    ordered = normalized.loc[:, columns_index]
    return [tuple(row) for row in ordered.itertuples(index=False, name=None)]
