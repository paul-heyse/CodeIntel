"""Migration bridge from SCHEMA_REGISTRY to Hamilton-native validators.

This module provides utilities to generate Hamilton validators from existing
Pandera schemas in the SCHEMA_REGISTRY. It enables gradual migration to
Hamilton-native validation.

The migration bridge allows:
1. Converting Pandera schemas to Hamilton validators
2. Generating @schema.output arguments from existing schemas
3. Creating validator sets that mirror existing schema validation

Examples
--------
Generate Hamilton validators from an existing Pandera schema:

>>> from codeintel.build.hamilton.validators.migration import (
...     validators_from_schema_registry,
... )
>>> validators = validators_from_schema_registry("analytics.function_metrics")
>>> # Use with @check_output_custom(*validators)

Generate @schema.output args from registry:

>>> from codeintel.build.hamilton.validators.migration import (
...     schema_output_from_registry,
... )
>>> columns = schema_output_from_registry("analytics.function_metrics")
>>> # Use with @schema.output(*columns)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from codeintel.build.hamilton.validators.dataframe import (
    ColumnsExistValidator,
    ColumnTypesValidator,
    NoNullsInColumnsValidator,
    UniqueColumnsValidator,
)

if TYPE_CHECKING:
    from hamilton.data_quality.base import BaseDefaultValidator

__all__ = [
    "MigrationReport",
    "schema_output_from_registry",
    "validators_from_pandera_schema",
    "validators_from_schema_registry",
]

log = logging.getLogger(__name__)


class MigrationReport:
    """Report of migration from Pandera to Hamilton validators.

    Tracks what was migrated and any issues encountered.

    Parameters
    ----------
    table_key
        The table being migrated.

    Attributes
    ----------
    columns_migrated
        Number of columns successfully migrated.
    validators_created
        Number of validators created.
    warnings
        List of warning messages.
    errors
        List of error messages.
    """

    def __init__(self, table_key: str) -> None:
        """Initialize migration report."""
        self.table_key = table_key
        self.columns_migrated: int = 0
        self.validators_created: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def success(self) -> bool:
        """Check if migration was successful (no errors)."""
        return len(self.errors) == 0

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        log.warning("Migration warning for %s: %s", self.table_key, message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        log.error("Migration error for %s: %s", self.table_key, message)

    def summary(self) -> str:
        """Generate summary string.

        Returns
        -------
        str
            Human-readable migration summary for this table.
        """
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"Migration {status} for {self.table_key}: "
            f"{self.columns_migrated} columns, "
            f"{self.validators_created} validators, "
            f"{len(self.warnings)} warnings, "
            f"{len(self.errors)} errors"
        )


def _pandera_dtype_to_hamilton_type(dtype: Any) -> str:
    """Convert Pandera dtype to Hamilton type string.

    Parameters
    ----------
    dtype
        Pandera column dtype.

    Returns
    -------
    str
        Hamilton type string.
    """
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "int"
    if "float" in dtype_str:
        return "float"
    if "bool" in dtype_str:
        return "bool"
    if "datetime" in dtype_str:
        return "datetime"
    if "string" in dtype_str or "object" in dtype_str:
        return "string"
    return "object"


def validators_from_schema_registry(
    table_key: str,
    *,
    strict: bool = False,
) -> list[BaseDefaultValidator]:
    """Generate Hamilton validators from existing Pandera schema.

    This is the primary migration bridge - it reads a schema from
    SCHEMA_REGISTRY and creates equivalent Hamilton validators.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    strict
        If True, raise an exception if the table is not found.

    Returns
    -------
    list[BaseDefaultValidator]
        List of validators equivalent to the Pandera schema.
        Returns empty list if table not found (unless strict=True).

    Notes
    -----
    When strict=True and the table is not found in SCHEMA_REGISTRY,
    a KeyError is raised.

    Raises
    ------
    ImportError
        If the schema registry cannot be imported when strict=True.
    KeyError
        If the table is missing from the registry and strict=True.

    Examples
    --------
    >>> validators = validators_from_schema_registry("analytics.function_metrics")
    >>> @check_output_custom(*validators)
    >>> def t__function_metrics__compute(...) -> pd.DataFrame:
    ...     ...
    """
    try:
        from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY

        dataset_schema = SCHEMA_REGISTRY.get(table_key)
        if dataset_schema is None:
            if strict:
                msg = f"Table {table_key} not found in SCHEMA_REGISTRY"
                raise KeyError(msg)
            log.warning("Table %s not found in SCHEMA_REGISTRY", table_key)
            return []

        return validators_from_pandera_schema(dataset_schema.pandera_schema)

    except ImportError:
        if strict:
            msg = "SCHEMA_REGISTRY not available"
            raise ImportError(msg) from None
        log.debug("SCHEMA_REGISTRY not available")
        return []
    except KeyError:
        raise
    except Exception as exc:
        if strict:
            raise
        log.warning("Error creating validators for %s: %s", table_key, exc)
        return []


def validators_from_pandera_schema(
    pandera_schema: Any,
) -> list[BaseDefaultValidator]:
    """Generate Hamilton validators from a Pandera DataFrameSchema.

    Parameters
    ----------
    pandera_schema
        A Pandera DataFrameSchema instance.

    Returns
    -------
    list[BaseDefaultValidator]
        List of Hamilton validators that implement the same validation.

    Examples
    --------
    >>> import pandera as pa
    >>> schema = pa.DataFrameSchema(
    ...     {
    ...         "id": pa.Column(int, nullable=False, unique=True),
    ...         "name": pa.Column(str),
    ...     }
    ... )
    >>> validators = validators_from_pandera_schema(schema)
    """
    validators: list[BaseDefaultValidator] = []

    # Get column information
    columns = list(pandera_schema.columns.keys())
    column_types: dict[str, str] = {}
    non_nullable: list[str] = []
    unique_cols: list[str] = []

    for col_name, col_spec in pandera_schema.columns.items():
        # Extract dtype
        dtype_str = _pandera_dtype_to_hamilton_type(col_spec.dtype)
        column_types[col_name] = dtype_str

        # Check nullable
        if hasattr(col_spec, "nullable") and not col_spec.nullable:
            non_nullable.append(col_name)

        # Check unique
        if hasattr(col_spec, "unique") and col_spec.unique:
            unique_cols.append(col_name)

    # Create validators
    if columns:
        validators.append(ColumnsExistValidator(columns))

    if column_types:
        validators.append(ColumnTypesValidator(column_types))

    if non_nullable:
        validators.append(NoNullsInColumnsValidator(non_nullable))

    if unique_cols:
        validators.append(UniqueColumnsValidator(unique_cols))

    return validators


def schema_output_from_registry(
    table_key: str,
    *,
    strict: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Generate @schema.output arguments from existing Pandera schema.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    strict
        If True, raise an exception if the table is not found.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Tuple of (column_name, dtype) tuples for @schema.output.
        Returns empty tuple if table not found (unless strict=True).

    Notes
    -----
    When strict=True and the table is not found, KeyError is raised.

    Raises
    ------
    ImportError
        If the schema registry cannot be imported when strict=True.
    KeyError
        If the table is missing from the registry and strict=True.

    Examples
    --------
    >>> columns = schema_output_from_registry("analytics.function_metrics")
    >>> @schema.output(*columns)
    >>> def t__function_metrics__compute(...) -> pd.DataFrame:
    ...     ...
    """
    try:
        from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY

        dataset_schema = SCHEMA_REGISTRY.get(table_key)
        if dataset_schema is None:
            if strict:
                msg = f"Table {table_key} not found in SCHEMA_REGISTRY"
                raise KeyError(msg)
            log.warning("Table %s not found in SCHEMA_REGISTRY", table_key)
            return ()

        pandera_schema = dataset_schema.pandera_schema
        columns: list[tuple[str, str]] = []

        for col_name, col_spec in pandera_schema.columns.items():
            dtype_str = _pandera_dtype_to_hamilton_type(col_spec.dtype)
            columns.append((col_name, dtype_str))

        return tuple(columns)

    except ImportError:
        if strict:
            msg = "SCHEMA_REGISTRY not available"
            raise ImportError(msg) from None
        log.debug("SCHEMA_REGISTRY not available")
        return ()
    except KeyError:
        raise
    except Exception as exc:
        if strict:
            raise
        log.warning("Error getting schema for %s: %s", table_key, exc)
        return ()


def generate_migration_code(
    table_key: str,
    node_name: str | None = None,
) -> str:
    """Generate code snippet for migrating a table to Hamilton-native validation.

    This utility helps developers by generating the code they need to
    add Hamilton-native validation to their nodes.

    Parameters
    ----------
    table_key
        Fully-qualified table name.
    node_name
        Optional Hamilton node name (defaults to generated name).

    Returns
    -------
    str
        Python code snippet ready to paste into a Hamilton module.

    Examples
    --------
    >>> code = generate_migration_code("analytics.function_metrics")
    >>> print(code)
    """
    # Generate node name if not provided
    if node_name is None:
        parts = table_key.replace(".", "__")
        node_name = f"t__{parts}__compute"

    # Get schema info
    columns = schema_output_from_registry(table_key)
    validators = validators_from_schema_registry(table_key)

    # Build decorator strings
    schema_args = ",\n    ".join(f'("{col}", "{dtype}")' for col, dtype in columns)

    validator_lines = []
    for v in validators:
        validator_name = type(v).__name__
        if isinstance(v, ColumnsExistValidator):
            cols = v.required_columns
            validator_lines.append(f"    {validator_name}({cols}),")
        elif isinstance(v, ColumnTypesValidator):
            types = v.column_types
            validator_lines.append(f"    {validator_name}({types}),")
        elif isinstance(v, NoNullsInColumnsValidator) or isinstance(v, UniqueColumnsValidator):
            cols = v.columns
            validator_lines.append(f"    {validator_name}({cols}),")

    validators_str = "\n".join(validator_lines)

    # Generate the code
    code = f'''
# Generated migration code for {table_key}
from hamilton.function_modifiers import check_output_custom, schema, tag

from codeintel.build.hamilton.validators import (
    ColumnsExistValidator,
    ColumnTypesValidator,
    NoNullsInColumnsValidator,
    UniqueColumnsValidator,
)

@tag(domain="{table_key.split(".", maxsplit=1)[0]}", target="{table_key.split(".")[1]}", node_type="compute")
@schema.output(
    {schema_args}
)
@check_output_custom(
{validators_str}
)
def {node_name}(...) -> pd.DataFrame:
    """Compute {table_key} with Hamilton-native validation."""
    ...
'''
    return code.strip()
