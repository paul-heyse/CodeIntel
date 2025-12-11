"""Schema builder to bridge existing contracts to DatasetSchema.

This module provides functions to create DatasetSchema instances from existing
DatasetContract and Pandera DataFrameSchema objects, enabling a smooth transition
to the unified schema layer.

Examples
--------
>>> from codeintel.config.datasets.schema_builder import build_all_schemas
>>> schemas = build_all_schemas()
>>> "analytics.function_metrics" in schemas
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets.contracts import get_dataset_contracts_by_table_key
from codeintel.config.datasets.schema import DatasetMetadata, DatasetSchema
from codeintel.storage.pandera_schemas import _get_dataset_schemas

if TYPE_CHECKING:
    from pandera import DataFrameSchema

    from codeintel.config.datasets.contracts import DatasetContract

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
    >>> from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
    >>> from codeintel.storage.pandera_schemas import DATASET_SCHEMAS
    >>> contract = DATASET_CONTRACTS_BY_TABLE_KEY["analytics.function_metrics"]
    >>> pa_schema = DATASET_SCHEMAS["analytics.function_metrics"]
    >>> ds = build_dataset_schema(contract, pa_schema)
    >>> ds.name
    'analytics.function_metrics'
    """
    # Extract metadata from contract
    metadata = DatasetMetadata(
        description=contract.description,
        owner=contract.owner,
        family=contract.family,
        freshness_sla=contract.freshness_sla,
        retention_policy=contract.retention_policy,
        upstream_dependencies=contract.upstream_dependencies,
        downstream_consumers=(),  # Computed later by registry
        tags=contract.tags,
        deprecated=contract.deprecated,
        deprecation_message=contract.deprecation_message,
    )

    # Get pre-existing row model if available
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

    This reads from DATASET_CONTRACTS_BY_TABLE_KEY and DATASET_SCHEMAS to
    create a complete mapping of DatasetSchema objects.

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

    for table_key, contract in get_dataset_contracts_by_table_key().items():
        pandera_schema = dataset_schemas.get(table_key)

        if pandera_schema is None:
            # Skip datasets without Pandera schemas
            continue

        schemas[table_key] = build_dataset_schema(contract, pandera_schema)

    return schemas
