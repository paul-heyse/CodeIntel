"""Hamilton-native data validators.

This package provides custom validators that integrate with Hamilton's
@check_output_custom decorator to provide framework-driven data validation.

The validators in this package make Hamilton's DAG engine the authoritative
source for data contracts, enforcing validation at execution time.

Examples
--------
Basic usage with individual validators:

>>> from hamilton.function_modifiers import check_output_custom
>>> from codeintel.build.hamilton.validators import (
...     ColumnsExistValidator,
...     NoNullsInColumnsValidator,
... )
>>>
>>> @check_output_custom(
...     ColumnsExistValidator(["id", "name"]),
...     NoNullsInColumnsValidator(["id"]),
... )
>>> def my_node(...) -> pd.DataFrame:
...     ...

Using contract builders for common patterns:

>>> from codeintel.build.hamilton.validators import build_table_contract
>>>
>>> validators = build_table_contract(
...     required_columns=["id", "name", "value"],
...     column_types={"value": "float"},
...     no_nulls=["id"],
...     unique=["id"],
... )
>>>
>>> @check_output_custom(*validators)
>>> def my_table(...) -> pd.DataFrame:
...     ...

Migrating from SCHEMA_REGISTRY:

>>> from codeintel.build.hamilton.validators import validators_from_schema_registry
>>> validators = validators_from_schema_registry("analytics.function_metrics")
>>> @check_output_custom(*validators)
>>> def my_migrated_node(...) -> pd.DataFrame:
...     ...
"""
from __future__ import annotations

from codeintel.build.hamilton.validators.contracts import (
    build_enum_column_contract,
    build_key_column_contract,
    build_metrics_contract,
    build_table_contract,
)
from codeintel.build.hamilton.validators.dataframe import (
    ColumnsExistValidator,
    ColumnTypesValidator,
    ColumnValuesInSetValidator,
    NoNullsInColumnsValidator,
    RowCountRangeValidator,
    RowCountValidator,
    UniqueColumnsValidator,
)
from codeintel.build.hamilton.validators.migration import (
    MigrationReport,
    generate_migration_code,
    schema_output_from_registry,
    validators_from_pandera_schema,
    validators_from_schema_registry,
)

__all__ = [
    # DataFrame validators
    "ColumnsExistValidator",
    "ColumnTypesValidator",
    "ColumnValuesInSetValidator",
    "NoNullsInColumnsValidator",
    "RowCountRangeValidator",
    "RowCountValidator",
    "UniqueColumnsValidator",
    # Contract builders
    "build_enum_column_contract",
    "build_key_column_contract",
    "build_metrics_contract",
    "build_table_contract",
    # Migration utilities
    "MigrationReport",
    "generate_migration_code",
    "schema_output_from_registry",
    "validators_from_pandera_schema",
    "validators_from_schema_registry",
]
