from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from codeintel.analytics.rows.function_metrics import (
    FunctionMetricsRow,
    function_metrics_row_to_tuple,
)
from codeintel.analytics.rows.function_types import (
    FunctionTypesRow,
    function_types_row_to_tuple,
)
from codeintel.analytics.rows.graph_metrics import (
    FunctionGraphMetricsRow,
    ModuleGraphMetricsRow,
    function_graph_metrics_row_to_tuple,
    module_graph_metrics_row_to_tuple,
)
from codeintel.analytics.rows.graph_metrics_ext import (
    FunctionGraphMetricsExtRow,
    ModuleGraphMetricsExtRow,
    function_graph_metrics_ext_row_to_tuple,
    module_graph_metrics_ext_row_to_tuple,
)
from codeintel.analytics.rows.test_profiles import (
    BehavioralCoverageRow,
    TestProfileRow,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.ingestion.common import run_batch
from codeintel.storage import rows as row_models
from codeintel.storage.datasets import Dataset, DatasetRegistry, load_dataset_registry
from codeintel.storage.gateway import StorageGateway

RowType: TypeAlias = Mapping[str, object]
ToTuple = Callable[[RowType], tuple[object, ...]]


@dataclass(frozen=True)
class AnalyticsDatasetContract:
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
    row_type: type[RowType]
    to_tuple: ToTuple
    primary_key: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...]
    dataset_meta: Dataset | None = None


def _build_registry(gateway: StorageGateway) -> DatasetRegistry:
    return load_dataset_registry(gateway.con)


def build_analytics_dataset_contracts(
    gateway: StorageGateway,
) -> dict[str, AnalyticsDatasetContract]:
    """
    Build dataset contracts for analytics tables using registry metadata.
    """
    registry = _build_registry(gateway)

    def _contract(
        name: str,
        *,
        row_type: type[RowType],
        to_tuple: ToTuple,
    ) -> AnalyticsDatasetContract:
        dataset = registry.by_name.get(name)
        table_key = dataset.table_key if dataset is not None else name
        schema = TABLE_SCHEMAS.get(table_key)
        primary_key = schema.primary_key if schema is not None else ()
        indexes = (
            tuple(index.columns for index in schema.indexes)
            if schema is not None and schema.indexes
            else ()
        )
        return AnalyticsDatasetContract(
            name=name,
            table_key=table_key,
            schema=schema,
            row_type=row_type,
            to_tuple=to_tuple,
            primary_key=primary_key,
            indexes=indexes,
            dataset_meta=dataset,
        )

    return {
        "analytics.function_metrics": _contract(
            "analytics.function_metrics",
            row_type=FunctionMetricsRow,  # type: ignore[arg-type]
            to_tuple=function_metrics_row_to_tuple,
        ),
        "analytics.function_types": _contract(
            "analytics.function_types",
            row_type=FunctionTypesRow,  # type: ignore[arg-type]
            to_tuple=function_types_row_to_tuple,
        ),
        "analytics.function_profile": _contract(
            "analytics.function_profile",
            row_type=row_models.FunctionProfileRowModel,  # type: ignore[arg-type]
            to_tuple=row_models.function_profile_row_to_tuple,
        ),
        "analytics.test_profile": _contract(
            "analytics.test_profile",
            row_type=TestProfileRow,  # type: ignore[arg-type]
            to_tuple=serialize_test_profile_row,
        ),
        "analytics.behavioral_coverage": _contract(
            "analytics.behavioral_coverage",
            row_type=BehavioralCoverageRow,  # type: ignore[arg-type]
            to_tuple=behavioral_coverage_row_to_tuple,
        ),
        "analytics.graph_metrics_functions": _contract(
            "analytics.graph_metrics_functions",
            row_type=FunctionGraphMetricsRow,  # type: ignore[arg-type]
            to_tuple=function_graph_metrics_row_to_tuple,
        ),
        "analytics.graph_metrics_modules": _contract(
            "analytics.graph_metrics_modules",
            row_type=ModuleGraphMetricsRow,  # type: ignore[arg-type]
            to_tuple=module_graph_metrics_row_to_tuple,
        ),
        "analytics.graph_metrics_functions_ext": _contract(
            "analytics.graph_metrics_functions_ext",
            row_type=FunctionGraphMetricsExtRow,  # type: ignore[arg-type]
            to_tuple=function_graph_metrics_ext_row_to_tuple,
        ),
        "analytics.graph_metrics_modules_ext": _contract(
            "analytics.graph_metrics_modules_ext",
            row_type=ModuleGraphMetricsExtRow,  # type: ignore[arg-type]
            to_tuple=module_graph_metrics_ext_row_to_tuple,
        ),
    }


def get_analytics_dataset_contract(
    gateway: StorageGateway,
    name: str,
) -> AnalyticsDatasetContract:
    """
    Return the dataset contract for a named analytics dataset.
    """
    contracts = build_analytics_dataset_contracts(gateway)
    if name not in contracts:
        message = f"Unknown analytics dataset: {name}"
        raise KeyError(message)
    return contracts[name]


def insert_analytics_rows(
    gateway: StorageGateway,
    contract: AnalyticsDatasetContract,
    rows: list[RowType],
    *,
    delete_params: list[object] | None = None,
    scope: str | None = None,
) -> None:
    """
    Insert rows for a dataset contract using run_batch.
    """
    if not rows:
        return
    tuple_rows = [contract.to_tuple(row) for row in rows]
    run_batch(
        gateway,
        contract.table_key,
        tuple_rows,
        delete_params=delete_params,
        scope=scope,
    )
