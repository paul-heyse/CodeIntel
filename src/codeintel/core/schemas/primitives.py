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

_DECIMAL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^DECIMAL\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_ADVANCED_TYPE_PREFIXES: Final[tuple[str, ...]] = ("STRUCT", "LIST", "MAP", "UNION")
COMPLEX_TYPE_BASES: Final[frozenset[str]] = frozenset({*_ADVANCED_TYPE_PREFIXES, "JSON"})
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
    "BLOB": "BLOB",
    "BINARY": "BLOB",
    "BYTEA": "BLOB",
    "VARBINARY": "BLOB",
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
        "BLOB",
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
    "BLOB": bytes,
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
_ALLOWED_COLUMN_PROMOTIONS: Final[dict[str, tuple[str, ...]]] = {
    "INTEGER": ("BIGINT",),
}


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


def is_allowed_column_promotion(source: ColumnType, target: ColumnType) -> bool:
    """Return True when a column type promotion is explicitly allowed.

    Returns
    -------
    bool
        True when the promotion is allowed.
    """
    source_norm = normalize_column_type(source)
    target_norm = normalize_column_type(target)
    if source_norm == target_norm:
        return True
    source_base = column_type_base(source_norm)
    target_base = column_type_base(target_norm)
    if source_base in COMPLEX_TYPE_BASES or target_base in COMPLEX_TYPE_BASES:
        return False
    allowed = _ALLOWED_COLUMN_PROMOTIONS.get(source_base, ())
    return target_base in allowed


WriteMode = Literal["append", "replace", "upsert"]
ReplaceScope = Literal["snapshot", "table"]
FinalizeNullListPolicy = Literal["error", "empty"]
FinalizeInvariantKind = Literal["list_alignment", "struct_required"]
FinalizeDedupeTier = Literal["canonical", "stable_set", "best_effort", "throughput"]
FinalizeDedupeStrategy = Literal[
    "order_independent",
    "first",
    "keep_best_by_score",
    "keep_arbitrary",
]
SortDirection = Literal["ascending", "descending"]


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
    stable_sort_keys
        Optional stable sort keys for deterministic on-disk ordering.
    combine_chunks
        Whether to consolidate chunks before writes when supported.
    """

    mode: WriteMode = "replace"
    replace_scope: ReplaceScope = "snapshot"
    conflict_columns: tuple[str, ...] | None = None
    update_columns: tuple[str, ...] | None = None
    hash_column: str | None = None
    use_staging: bool = False
    stable_sort_keys: tuple[str, ...] | None = None
    combine_chunks: bool | None = None


@dataclass(frozen=True)
class FinalizeListPolicySpec:
    """Finalize policy for list null handling."""

    column: str
    null_policy: FinalizeNullListPolicy = "error"

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this policy.

        Returns
        -------
        dict[str, object]
            JSON-serializable policy payload.
        """
        return {"column": self.column, "null_policy": self.null_policy}


@dataclass(frozen=True)
class FinalizeInvariantSpec:
    """Finalize invariant policy for table validation."""

    kind: FinalizeInvariantKind
    column: str
    related: tuple[str, ...]

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this invariant.

        Returns
        -------
        dict[str, object]
            JSON-serializable invariant payload.
        """
        return {
            "kind": self.kind,
            "column": self.column,
            "related": list(self.related),
        }


@dataclass(frozen=True)
class FinalizeDedupeSpec:
    """Finalize policy for dedupe behavior."""

    enabled: bool = True
    keys: tuple[str, ...] = ()
    prefer_columns: tuple[str, ...] = ()
    tie_breakers: tuple[tuple[str, SortDirection], ...] = ()
    tier: FinalizeDedupeTier | None = None
    strategy: FinalizeDedupeStrategy | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this dedupe policy.

        Returns
        -------
        dict[str, object]
            JSON-serializable dedupe payload.
        """
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "keys": list(self.keys),
            "prefer_columns": list(self.prefer_columns),
            "tie_breakers": [list(item) for item in self.tie_breakers],
        }
        if self.tier is not None:
            payload["tier"] = self.tier
        if self.strategy is not None:
            payload["strategy"] = self.strategy
        return payload


@dataclass(frozen=True)
class FinalizePolicy:
    """Finalize policy defaults for a table schema."""

    required_non_null: tuple[str, ...] = ()
    list_policies: tuple[FinalizeListPolicySpec, ...] = ()
    invariants: tuple[FinalizeInvariantSpec, ...] = ()
    dedupe: FinalizeDedupeSpec | None = None
    canonical_sort_keys: tuple[str, ...] | None = None

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this finalize policy.

        Returns
        -------
        dict[str, object]
            JSON-serializable finalize policy payload.
        """
        payload: dict[str, object] = {
            "required_non_null": list(self.required_non_null),
            "list_policies": [policy.to_json_obj() for policy in self.list_policies],
            "invariants": [inv.to_json_obj() for inv in self.invariants],
        }
        if self.dedupe is not None:
            payload["dedupe"] = self.dedupe.to_json_obj()
        if self.canonical_sort_keys is not None:
            payload["canonical_sort_keys"] = list(self.canonical_sort_keys)
        return payload


@dataclass(frozen=True)
class PlanPolicy:
    """Plan defaults for query/scan behavior."""

    default_projection: tuple[str, ...] | None = None
    join_safe_columns: tuple[str, ...] = ()

    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this plan policy.

        Returns
        -------
        dict[str, object]
            JSON-serializable plan policy payload.
        """
        payload: dict[str, object] = {}
        if self.default_projection is not None:
            payload["default_projection"] = list(self.default_projection)
        if self.join_safe_columns:
            payload["join_safe_columns"] = list(self.join_safe_columns)
        return payload


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
    finalize_policy: FinalizePolicy | None = None
    plan_policy: PlanPolicy | None = None

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
            if self.finalize_policy is None:
                if self.plan_policy is None:
                    return payload
                payload["plan_policy"] = self.plan_policy.to_json_obj()
                return payload
            payload["finalize_policy"] = self.finalize_policy.to_json_obj()
            if self.plan_policy is not None:
                payload["plan_policy"] = self.plan_policy.to_json_obj()
            return payload
        write_policy_payload: dict[str, object] = {
            "mode": self.write_policy.mode,
            "replace_scope": self.write_policy.replace_scope,
            "conflict_columns": list(self.write_policy.conflict_columns or ()),
            "update_columns": list(self.write_policy.update_columns or ()),
            "hash_column": self.write_policy.hash_column,
            "use_staging": self.write_policy.use_staging,
        }
        if self.write_policy.stable_sort_keys is not None:
            write_policy_payload["stable_sort_keys"] = list(self.write_policy.stable_sort_keys)
        payload["write_policy"] = write_policy_payload
        if self.finalize_policy is not None:
            payload["finalize_policy"] = self.finalize_policy.to_json_obj()
        if self.plan_policy is not None:
            payload["plan_policy"] = self.plan_policy.to_json_obj()
        return payload


def resolve_stable_sort_keys(table_schema: TableSchema | None) -> tuple[str, ...] | None:
    """Resolve stable sort keys from a table schema policy.

    Returns
    -------
    tuple[str, ...] | None
        Stable sort keys. An empty tuple explicitly disables canonical ordering, and
        None indicates no explicit ordering policy is configured.
    """
    if table_schema is None:
        return None
    policy = table_schema.write_policy
    if policy is not None and policy.stable_sort_keys is not None:
        return policy.stable_sort_keys
    return table_schema.primary_key or None


def resolve_canonical_sort_keys(table_schema: TableSchema | None) -> tuple[str, ...] | None:
    """Resolve canonical sort keys for finalization/ordering metadata.

    Returns
    -------
    tuple[str, ...] | None
        Canonical sort keys. An empty tuple explicitly disables canonical ordering,
        and None indicates no ordering policy is configured.
    """
    if table_schema is None:
        return None
    write_policy = table_schema.write_policy
    if write_policy is not None and write_policy.stable_sort_keys is not None:
        return write_policy.stable_sort_keys
    finalize_policy = table_schema.finalize_policy
    if finalize_policy is not None and finalize_policy.canonical_sort_keys is not None:
        return finalize_policy.canonical_sort_keys
    return resolve_stable_sort_keys(table_schema)


def resolve_default_projection(table_schema: TableSchema | None) -> tuple[str, ...] | None:
    """Resolve default projection columns from a table schema plan policy.

    Returns
    -------
    tuple[str, ...] | None
        Default projection columns, or None when no policy is configured.
    """
    if table_schema is None:
        return None
    plan_policy = table_schema.plan_policy
    if plan_policy is None:
        return None
    return plan_policy.default_projection


def resolve_join_safe_columns(table_schema: TableSchema | None) -> tuple[str, ...]:
    """Resolve join-safe allowlist columns from a table schema plan policy.

    Returns
    -------
    tuple[str, ...]
        Join-safe allowlist column names (empty when not configured).
    """
    if table_schema is None:
        return ()
    plan_policy = table_schema.plan_policy
    if plan_policy is None:
        return ()
    return plan_policy.join_safe_columns


__all__ = [
    "COLUMN_TYPE_BASE_VALUES",
    "COLUMN_TYPE_REGISTRY",
    "COMPLEX_TYPE_BASES",
    "Column",
    "ColumnType",
    "ColumnTypeRegistry",
    "FinalizeDedupeSpec",
    "FinalizeDedupeStrategy",
    "FinalizeDedupeTier",
    "FinalizeInvariantKind",
    "FinalizeInvariantSpec",
    "FinalizeListPolicySpec",
    "FinalizeNullListPolicy",
    "FinalizePolicy",
    "Index",
    "PlanPolicy",
    "ReplaceScope",
    "SortDirection",
    "TableSchema",
    "TableWritePolicy",
    "WriteMode",
    "column_type_base",
    "column_type_is_nested",
    "is_allowed_column_promotion",
    "normalize_column_type",
    "resolve_canonical_sort_keys",
    "resolve_default_projection",
    "resolve_join_safe_columns",
    "resolve_stable_sort_keys",
]
