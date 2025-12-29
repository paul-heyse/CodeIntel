"""Core schema primitives.

These dataclasses define the minimal schema language used across the project.
They intentionally avoid importing higher-level configuration modules so that
storage and build layers can depend on them without cycles.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Literal

ColumnType = str

_DECIMAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^DECIMAL\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)$"
)
_ADVANCED_TYPE_PREFIXES: Final[tuple[str, ...]] = ("STRUCT", "LIST", "MAP", "UNION")
_ALIAS_CANONICAL: Final[dict[str, str]] = {
    "BOOL": "BOOLEAN",
    "BOOLEAN": "BOOLEAN",
    "INT": "INTEGER",
    "INT4": "INTEGER",
    "INT32": "INTEGER",
    "INTEGER": "INTEGER",
    "SMALLINT": "INTEGER",
    "TINYINT": "INTEGER",
    "BIGINT": "BIGINT",
    "INT64": "BIGINT",
    "LONG": "BIGINT",
    "DOUBLE": "DOUBLE",
    "FLOAT": "DOUBLE",
    "FLOAT4": "DOUBLE",
    "FLOAT8": "DOUBLE",
    "FLOAT64": "DOUBLE",
    "REAL": "DOUBLE",
    "DECIMAL": "DECIMAL",
    "NUMERIC": "DECIMAL",
    "VARCHAR": "VARCHAR",
    "TEXT": "VARCHAR",
    "STRING": "VARCHAR",
    "CHAR": "VARCHAR",
    "JSON": "JSON",
    "TIMESTAMP": "TIMESTAMP",
    "TIMESTAMP_TZ": "TIMESTAMPTZ",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
    "TIMESTAMPTZ": "TIMESTAMPTZ",
}
COLUMN_TYPE_BASE_VALUES: Final[frozenset[str]] = frozenset(
    {
        "BOOLEAN",
        "INTEGER",
        "BIGINT",
        "DOUBLE",
        "DECIMAL",
        "DECIMAL(38,0)",
        "VARCHAR",
        "JSON",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        *_ADVANCED_TYPE_PREFIXES,
    }
)

_PYTHON_TYPE_MAP: Final[dict[str, type[object]]] = {
    "INTEGER": int,
    "BIGINT": int,
    "DOUBLE": float,
    "BOOLEAN": bool,
    "VARCHAR": str,
    "JSON": object,
    "TIMESTAMP": dt.datetime,
    "TIMESTAMPTZ": dt.datetime,
    "STRUCT": dict,
    "MAP": dict,
    "LIST": list,
    "UNION": object,
}


def _clean_column_type(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        msg = "Column type must be a non-empty string"
        raise ValueError(msg)
    return cleaned


def _compact_upper(value: str) -> str:
    return " ".join(value.upper().split())


def _normalize_list_suffix(value: str, registry: ColumnTypeRegistry) -> ColumnType | None:
    if not value.endswith("[]"):
        return None
    inner = value[:-2].strip()
    if not inner:
        msg = "Column type list syntax requires a base type"
        raise ValueError(msg)
    return f"LIST({registry.normalize(inner)})"


def _normalize_decimal(compact: str) -> ColumnType | None:
    match = _DECIMAL_PATTERN.fullmatch(compact)
    if match is None:
        return None
    precision = match.group(1)
    scale = match.group(2)
    return f"DECIMAL({precision},{scale})"


def _normalize_array(value: str, compact: str) -> ColumnType | None:
    if not compact.startswith("ARRAY"):
        return None
    suffix = value[len("ARRAY") :].strip()
    if not suffix:
        msg = "ARRAY column type requires inner type"
        raise ValueError(msg)
    return f"LIST{suffix}"


def _normalize_advanced(value: str, upper: str) -> ColumnType | None:
    for prefix in _ADVANCED_TYPE_PREFIXES:
        if upper.startswith(prefix):
            return f"{prefix}{value[len(prefix) :]}"
    return None


def _decimal_scale(value: ColumnType) -> int | None:
    compact = str(value).upper().replace(" ", "")
    match = _DECIMAL_PATTERN.match(compact)
    if match is None:
        return None
    return int(match.group(2))


class ColumnTypeRegistry:
    """Normalize and interpret column type strings."""

    def normalize(self, value: str) -> ColumnType:
        """Normalize a column type string.

        Parameters
        ----------
        value
            Raw column type string.

        Returns
        -------
        ColumnType
            Normalized column type string.

        Raises
        ------
        TypeError
            If ``value`` is not a string.
        ValueError
            If ``value`` is empty or unsupported.
        """
        if not isinstance(value, str):
            msg = f"Column type must be a string, got {type(value)}"
            raise TypeError(msg)
        cleaned = _clean_column_type(value)
        list_type = _normalize_list_suffix(cleaned, self)
        if list_type is not None:
            return list_type
        upper = cleaned.upper()
        compact = _compact_upper(cleaned)
        alias = _ALIAS_CANONICAL.get(compact)
        if alias is not None:
            return alias
        if compact in COLUMN_TYPE_BASE_VALUES:
            return compact
        decimal = _normalize_decimal(compact)
        if decimal is not None:
            return decimal
        array_type = _normalize_array(cleaned, compact)
        if array_type is not None:
            return array_type
        advanced = _normalize_advanced(cleaned, upper)
        if advanced is not None:
            return advanced
        msg = f"Unsupported column type: {value}"
        raise ValueError(msg)

    @staticmethod
    def base_type(value: ColumnType) -> str:
        """Return the normalized base type for a column type string.

        Returns
        -------
        str
            Normalized base type.
        """
        upper = str(value).strip().upper()
        if upper.startswith("DECIMAL("):
            return "DECIMAL"
        if upper.startswith("ARRAY"):
            return "LIST"
        for prefix in _ADVANCED_TYPE_PREFIXES:
            if upper.startswith(prefix):
                return prefix
        return upper

    def is_nested(self, value: ColumnType) -> bool:
        """Return True when the column type is a nested/complex type.

        Returns
        -------
        bool
            True if the base type is nested.
        """
        return self.base_type(value) in _ADVANCED_TYPE_PREFIXES

    def python_type_for(self, value: ColumnType) -> type[object]:
        """Return the Python type for a column type.

        Returns
        -------
        type[object]
            Python runtime type representing the column values.

        Raises
        ------
        ValueError
            If the column type cannot be mapped.
        """
        base = self.base_type(value)
        if base == "DECIMAL":
            scale = _decimal_scale(value)
            if scale == 0:
                return int
            return Decimal
        python_type = _PYTHON_TYPE_MAP.get(base)
        if python_type is None:
            msg = f"Unsupported ColumnType for Python mapping: {value}"
            raise ValueError(msg)
        return python_type


COLUMN_TYPE_REGISTRY = ColumnTypeRegistry()


def normalize_column_type(value: str) -> ColumnType:
    """Normalize a column type string.

    Parameters
    ----------
    value
        Raw column type string.

    Returns
    -------
    ColumnType
        Normalized column type string.
    """
    return COLUMN_TYPE_REGISTRY.normalize(value)


def column_type_base(value: ColumnType) -> str:
    """Return the normalized base type for a column type string.

    Returns
    -------
    str
        Normalized base type.
    """
    return COLUMN_TYPE_REGISTRY.base_type(value)


def column_type_is_nested(value: ColumnType) -> bool:
    """Return True when the column type is a nested/complex type.

    Returns
    -------
    bool
        True if the base type is nested.
    """
    return COLUMN_TYPE_REGISTRY.is_nested(value)


WriteMode = Literal["append", "replace", "upsert"]
ReplaceScope = Literal["snapshot", "table"]


@dataclass(frozen=True)
class TableWritePolicy:
    """Table-level write policy for materialization operations.

    Parameters
    ----------
    mode
        Write mode applied during materialization.
    replace_scope
        Scope for replace operations ("snapshot" deletes repo+commit rows,
        "table" deletes all rows).
    conflict_columns
        Columns used for upsert conflict detection.
    update_columns
        Columns updated on conflict (defaults to all non-conflict columns).
    hash_column
        Optional column used to gate updates (only update when hash differs).
    use_staging
        Whether to use a staging relation for writes.
    """

    mode: WriteMode = "replace"
    replace_scope: ReplaceScope = "snapshot"
    conflict_columns: tuple[str, ...] | None = None
    update_columns: tuple[str, ...] | None = None
    hash_column: str | None = None
    use_staging: bool = False


@dataclass(frozen=True)
class Column:
    """Definition of a single table column.

    Parameters
    ----------
    name
        Column name.
    type
        DuckDB column type.
    nullable
        Whether the column allows NULL values.
    description
        Optional column description for documentation.
    """

    name: str
    type: ColumnType
    nullable: bool = True
    description: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this column.

        Returns
        -------
        dict[str, object]
            JSON-serializable mapping for this column.
        """
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "description": self.description,
        }


@dataclass(frozen=True)
class Index:
    """Secondary index definition.

    Parameters
    ----------
    name
        Index name.
    columns
        Column names included in the index.
    unique
        Whether the index enforces uniqueness.
    """

    name: str
    columns: tuple[str, ...]
    unique: bool = False

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this index.

        Returns
        -------
        dict[str, object]
            JSON-serializable mapping for this index.
        """
        return {
            "name": self.name,
            "columns": list(self.columns),
            "unique": self.unique,
        }


@dataclass(frozen=True)
class TableSchema:
    """Schema definition for a DuckDB table.

    Parameters
    ----------
    schema
        Database schema name (e.g., "core", "analytics").
    name
        Table name.
    columns
        List of column definitions.
    primary_key
        Tuple of column names forming the primary key.
    indexes
        Secondary index definitions.
    description
        Optional table description for documentation.
    """

    schema: str
    name: str
    columns: list[Column]
    primary_key: tuple[str, ...] = ()
    indexes: tuple[Index, ...] = ()
    description: str | None = None
    write_policy: TableWritePolicy | None = None

    @property
    def table_key(self) -> str:
        """Return the fully qualified table key.

        Returns
        -------
        str
            Fully qualified table key in "schema.name" format.
        """
        return f"{self.schema}.{self.name}"

    @property
    def fq_name(self) -> str:
        """Return the fully qualified table name.

        Returns
        -------
        str
            Fully qualified table name (alias for table_key).
        """
        return self.table_key

    def column_names(self) -> list[str]:
        """Return ordered column names.

        Returns
        -------
        list[str]
            Column names in schema order.
        """
        return [col.name for col in self.columns]

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this table schema.

        Returns
        -------
        dict[str, object]
            JSON-serializable mapping for this table schema.
        """
        payload: dict[str, object] = {
            "schema": self.schema,
            "name": self.name,
            "table_key": self.table_key,
            "description": self.description,
            "primary_key": list(self.primary_key),
            "indexes": [idx.to_json_obj() for idx in self.indexes],
            "columns": [col.to_json_obj() for col in self.columns],
        }
        if self.write_policy is None:
            return payload
        write_policy_payload: dict[str, object] = {
            "mode": self.write_policy.mode,
            "replace_scope": self.write_policy.replace_scope,
            "conflict_columns": list(self.write_policy.conflict_columns or ()),
            "update_columns": list(self.write_policy.update_columns or ()),
            "hash_column": self.write_policy.hash_column,
            "use_staging": self.write_policy.use_staging,
        }
        payload["write_policy"] = write_policy_payload
        return payload


__all__ = [
    "COLUMN_TYPE_BASE_VALUES",
    "COLUMN_TYPE_REGISTRY",
    "Column",
    "ColumnType",
    "ColumnTypeRegistry",
    "Index",
    "ReplaceScope",
    "TableSchema",
    "TableWritePolicy",
    "WriteMode",
    "column_type_base",
    "column_type_is_nested",
    "normalize_column_type",
]
