"""Core schema primitives.

These dataclasses define the minimal schema language used across the project.
They intentionally avoid importing higher-level configuration modules so that
storage and build layers can depend on them without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnType = Literal[
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
]

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
    "Column",
    "ColumnType",
    "Index",
    "ReplaceScope",
    "TableSchema",
    "TableWritePolicy",
    "WriteMode",
]
