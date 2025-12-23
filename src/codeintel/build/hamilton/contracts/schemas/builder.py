"""Schema builder to bridge existing contracts to DatasetSchema.

This module provides functions to create DatasetSchema instances from existing
DatasetContract and Pandera DataFrameSchema objects, enabling a smooth transition
to the unified schema layer.

Examples
--------
>>> from codeintel.build.hamilton.contracts.schemas.builder import build_all_schemas
>>> schemas = build_all_schemas()
>>> "analytics.function_metrics" in schemas
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.contracts.schemas.operation_contracts_dataset import (
    build_operation_contract_schema,
)
from codeintel.build.hamilton.contracts.schemas.pandera_schemas import _get_dataset_schemas
from codeintel.build.hamilton.contracts.schemas.schema import DatasetMetadata, DatasetSchema
from codeintel.build.schemas import get_schema_service
from codeintel.core.schemas.contract_service import get_enriched_contract_service
from codeintel.core.schemas.row_models import row_binding_for_table_schema

if TYPE_CHECKING:
    from pandera import DataFrameSchema

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import TableSchema

__all__ = [
    "build_all_schemas",
    "build_dataset_schema",
]


def build_dataset_schema(
    *,
    table_key: str,
    contract: DatasetContract,
    pandera_schema: DataFrameSchema,
    table_schema: TableSchema | None,
) -> DatasetSchema:
    """Create a DatasetSchema from a DatasetContract and Pandera schema.

    This bridges the existing contract infrastructure to the new unified
    schema layer.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    contract
        Existing DatasetContract with metadata.
    pandera_schema
        Pandera DataFrameSchema defining structure and constraints.
    table_schema
        TableSchema resolved from the canonical schema provider.

    Returns
    -------
    DatasetSchema
        Unified schema combining both sources.

    Examples
    --------
    >>> from codeintel.build.schemas import get_contract_for_table_key
    >>> from codeintel.build.hamilton.contracts.schemas.pandera_schemas import _get_dataset_schemas
    >>> contract = get_contract_for_table_key("analytics.function_metrics")
    >>> table_schema = get_schema_provider().require_table_schema("analytics.function_metrics")
    >>> pa_schema = _get_dataset_schemas()["analytics.function_metrics"]
    >>> ds = build_dataset_schema(
    ...     table_key="analytics.function_metrics",
    ...     contract=contract,
    ...     pandera_schema=pa_schema,
    ...     table_schema=table_schema,
    ... )
    >>> ds.name
    'analytics.function_metrics'
    """
    metadata = DatasetMetadata(
        description=contract.description,
        owner=contract.owner,
        family=contract.family,
        freshness_sla=contract.freshness_sla,
        retention_policy=contract.retention_policy,
        upstream_dependencies=contract.upstream_dependencies,
        downstream_consumers=(),
        tags=contract.tags,
        deprecated=contract.deprecated,
        deprecation_message=contract.deprecation_message,
    )

    row_model = None
    if table_schema is not None:
        row_model = row_binding_for_table_schema(table_schema=table_schema).row_model
    elif contract.row_binding is not None:
        row_model = contract.row_binding.row_model

    return DatasetSchema(
        name=table_key,
        pandera_schema=pandera_schema,
        row_model=row_model,
        ddl_schema=table_schema,
        metadata=metadata,
        composition=contract.composition,
    )


def build_all_schemas() -> dict[str, DatasetSchema]:
    """Build DatasetSchema instances for all registered datasets.

    This reads from DATASET_CONTRACTS_BY_TABLE_KEY and calls _get_dataset_schemas()
    to create a complete mapping of DatasetSchema objects.

    Returns
    -------
    dict[str, DatasetSchema]
        Mapping from table key to DatasetSchema for all datasets that have
        both a contract and a Pandera schema.

    Notes
    -----
    Datasets without a Pandera schema are skipped (views without explicit
    Pandera definitions, for example).

    Raises
    ------
    KeyError
        If a table schema is missing for a non-view contract.

    Examples
    --------
    >>> schemas = build_all_schemas()
    >>> len(schemas) > 0
    True
    >>> all(isinstance(s, DatasetSchema) for s in schemas.values())
    True
    """
    dataset_schemas = _get_dataset_schemas()
    schema_service = get_schema_service()
    schema_provider = schema_service.table_provider
    schemas: dict[str, DatasetSchema] = {}

    service = get_enriched_contract_service()
    for table_key, contract in service.iter_contracts_by_table_key():
        pandera_schema = dataset_schemas.get(table_key)

        if pandera_schema is None:
            continue

        table_schema = None if contract.is_view else schema_provider.get_table_schema(table_key)
        if table_schema is None and not contract.is_view:
            msg = f"Missing TableSchema for {table_key}"
            raise KeyError(msg)

        schemas[table_key] = build_dataset_schema(
            table_key=table_key,
            contract=contract,
            pandera_schema=pandera_schema,
            table_schema=table_schema,
        )

    schemas.update(_build_additional_schemas())

    return schemas


def _build_additional_schemas() -> dict[str, DatasetSchema]:
    """Return DatasetSchema objects that are not backed by contracts.

    These schemas are used for internal validation datasets that leverage
    the unified DatasetSchema abstraction but do not have physical tables.

    Returns
    -------
    dict[str, DatasetSchema]
        Mapping of additional schema name to DatasetSchema instance.
    """
    operation_contract_schema = build_operation_contract_schema()

    return {operation_contract_schema.name: operation_contract_schema}
