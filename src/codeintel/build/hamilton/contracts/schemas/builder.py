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
from codeintel.build.schemas import ContractResolutionSettings, iter_contracts_by_table_key

if TYPE_CHECKING:
    from pandera import DataFrameSchema

    from codeintel.core.schemas.contract_primitives import DatasetContract

__all__ = [
    "build_all_schemas",
    "build_dataset_schema",
]


def build_dataset_schema(
    contract: DatasetContract,
    pandera_schema: DataFrameSchema,
) -> DatasetSchema:
    """Create a DatasetSchema from a DatasetContract and Pandera schema.

    This bridges the existing contract infrastructure to the new unified
    schema layer.

    Parameters
    ----------
    contract
        Existing DatasetContract with metadata.
    pandera_schema
        Pandera DataFrameSchema defining structure and constraints.

    Returns
    -------
    DatasetSchema
        Unified schema combining both sources.

    Examples
    --------
    >>> from codeintel.build.schemas import get_contract_for_table_key
    >>> from codeintel.build.hamilton.contracts.schemas.pandera_schemas import _get_dataset_schemas
    >>> contract = get_contract_for_table_key("analytics.function_metrics")
    >>> pa_schema = _get_dataset_schemas()["analytics.function_metrics"]
    >>> ds = build_dataset_schema(contract, pa_schema)
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
    if contract.row_binding is not None:
        row_model = contract.row_binding.row_type

    return DatasetSchema(
        name=contract.table_key,
        pandera_schema=pandera_schema,
        row_model=row_model,
        ddl_schema=contract.schema,
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

    Examples
    --------
    >>> schemas = build_all_schemas()
    >>> len(schemas) > 0
    True
    >>> all(isinstance(s, DatasetSchema) for s in schemas.values())
    True
    """
    dataset_schemas = _get_dataset_schemas()
    schemas: dict[str, DatasetSchema] = {}

    for table_key, contract in iter_contracts_by_table_key(
        settings=ContractResolutionSettings(include_target_metadata=True)
    ):
        pandera_schema = dataset_schemas.get(table_key)

        if pandera_schema is None:
            continue

        schemas[table_key] = build_dataset_schema(contract, pandera_schema)

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
