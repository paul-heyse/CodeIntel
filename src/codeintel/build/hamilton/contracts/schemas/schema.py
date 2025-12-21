"""Unified schema abstraction for CodeIntel datasets.

This module provides the `DatasetSchema` dataclass, which serves as the single
source of truth for dataset structure. All other representations (row models,
DDL, JSON schemas) can be derived from it.

Examples
--------
>>> from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
>>> schema = SCHEMA_REGISTRY.require("analytics.function_metrics")
>>> schema.column_names()
('function_goid_h128', 'urn', 'repo', ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.core.schemas.json_schema_gen import pandera_to_json_schema
from codeintel.core.schemas.row_models import typed_dict_from_pandera

if TYPE_CHECKING:
    import pandas as pd
    from pandera import DataFrameSchema

    from codeintel.config.datasets.primitives import CompositeSchema
    from codeintel.core.schemas.primitives import TableSchema

__all__ = [
    "DatasetMetadata",
    "DatasetSchema",
]


_ROW_MODEL_CACHE: dict[str, type[Any]] = {}


@dataclass(frozen=True)
class DatasetMetadata:
    """Governance and operational metadata for a dataset.

    Parameters
    ----------
    description
        Human-readable description of the dataset.
    owner
        Team or individual responsible for the dataset.
    family
        Dataset family derived from schema prefix (core, analytics, graph, docs).
    freshness_sla
        Expected refresh frequency (e.g., "daily", "hourly").
    retention_policy
        Data retention policy descriptor (e.g., "90d").
    upstream_dependencies
        Tuple of dataset names this dataset depends on.
    downstream_consumers
        Tuple of dataset names that consume this dataset.
    tags
        Classification tags applied to the dataset.
    deprecated
        Whether this dataset is deprecated.
    deprecation_message
        Migration guidance if the dataset is deprecated.
    """

    description: str | None = None
    owner: str | None = None
    family: str | None = None
    freshness_sla: str | None = None
    retention_policy: str | None = None
    upstream_dependencies: tuple[str, ...] = ()
    downstream_consumers: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    deprecated: bool = False
    deprecation_message: str | None = None


def _default_metadata() -> DatasetMetadata:
    """Create a default DatasetMetadata instance.

    Returns
    -------
    DatasetMetadata
        Empty metadata with default values.
    """
    return DatasetMetadata()


def _to_class_name(table_key: str) -> str:
    """Convert a table key to a PascalCase class name.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    str
        PascalCase name suitable for a class (e.g., "AnalyticsFunctionMetrics").

    Examples
    --------
    >>> _to_class_name("analytics.function_metrics")
    'AnalyticsFunctionMetrics'
    """
    parts = table_key.replace(".", "_").split("_")
    return "".join(part.capitalize() for part in parts)


@dataclass(frozen=True)
class DatasetSchema:
    """Unified schema abstraction for a dataset.

    This is THE single source of truth for dataset structure. Row models,
    DDL, and JSON schemas can all be derived from this abstraction.

    Parameters
    ----------
    name
        Fully qualified table name (e.g., "analytics.function_metrics").
    pandera_schema
        Pandera DataFrameSchema defining structure and validation constraints.
    row_model
        Optional pre-computed TypedDict for row-level typing.
    ddl_schema
        Optional TableSchema for DDL generation.
    metadata
        Governance and operational metadata.
    composition
        Optional CompositeSchema for profile datasets.

    Examples
    --------
    >>> from pandera import DataFrameSchema, Column
    >>> pa_schema = DataFrameSchema({"repo": Column(str), "loc": Column(int)})
    >>> ds = DatasetSchema(name="test.example", pandera_schema=pa_schema)
    >>> ds.column_names()
    ('repo', 'loc')
    """

    name: str
    pandera_schema: DataFrameSchema
    row_model: type[Any] | None = None
    ddl_schema: TableSchema | None = None
    metadata: DatasetMetadata = field(default_factory=_default_metadata)
    composition: CompositeSchema | None = None

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a DataFrame against this schema.

        Parameters
        ----------
        df
            DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Validated (and possibly coerced) DataFrame.
        """
        return self.pandera_schema.validate(df, lazy=True)

    def column_names(self) -> tuple[str, ...]:
        """Return ordered column names from the Pandera schema.

        Returns
        -------
        tuple[str, ...]
            Column names in definition order.
        """
        return tuple(self.pandera_schema.columns.keys())

    def json_schema(self) -> dict[str, Any]:
        """Generate JSON Schema 2020-12 from the Pandera schema.

        Returns
        -------
        dict[str, Any]
            JSON Schema representation of the dataset structure.
        """
        return pandera_to_json_schema(self.pandera_schema)

    def get_row_model(self) -> type[Any]:
        """Return or generate the row model class.

        If a pre-computed row model was provided at construction, returns it.
        Otherwise, generates one from the Pandera schema. Generated models
        are cached in a module-level registry for performance.

        Returns
        -------
        type[Any]
            Row model class for this dataset's rows.

        Notes
        -----
        The generated model is derived directly from the Pandera schema to
        ensure the row model always matches the validation schema.
        """
        if self.row_model is not None:
            return self.row_model

        if self.name in _ROW_MODEL_CACHE:
            return _ROW_MODEL_CACHE[self.name]

        model = typed_dict_from_pandera(
            f"{_to_class_name(self.name)}Row",
            self.pandera_schema,
        )
        _ROW_MODEL_CACHE[self.name] = model
        return model

    @property
    def table_key(self) -> str:
        """Return the fully qualified table name.

        Returns
        -------
        str
            The dataset name (alias for name property).
        """
        return self.name

    def has_composition(self) -> bool:
        """Check if this dataset is a composite profile.

        Returns
        -------
        bool
            True if composition metadata is present.
        """
        return self.composition is not None
