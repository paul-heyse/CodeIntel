"""Primitive types and column fragments for dataset schemas.

This module provides the foundational types used to define dataset schemas:
- ColumnType literal for DuckDB column types
- Column, Index, TableSchema dataclasses for schema definitions
- RowToTuple callable type for row serialization
- Reusable column fragments for schema composition
- CompositeSchema for profile table composition metadata
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Schema Definition Types
# ---------------------------------------------------------------------------

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
COLUMN_TYPE = ColumnType


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

    @property
    def fq_name(self) -> str:
        """Return fully qualified table name.

        Returns
        -------
        str
            Table name in "schema.name" format.
        """
        return f"{self.schema}.{self.name}"

    def column_names(self) -> list[str]:
        """Return ordered column names.

        Returns
        -------
        list[str]
            Column names in definition order.
        """
        return [col.name for col in self.columns]


# Type aliases for row serialization
RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


# ---------------------------------------------------------------------------
# Reusable Column Fragments
# ---------------------------------------------------------------------------
# These fragments provide composable building blocks for TABLE_SCHEMAS.
# Use tuple unpacking (*FRAGMENT) to include them in column lists.

# Versioning context (repo + commit)
REPO_COMMIT_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
)

# Function entity identification (GOID only, nullable=False)
FUNCTION_GOID_COL: Final[tuple[Column, ...]] = (
    Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
)

# Function entity identification (GOID only, nullable)
FUNCTION_GOID_COL_NULLABLE: Final[tuple[Column, ...]] = (
    Column("function_goid_h128", "DECIMAL(38,0)"),
)

# Function entity with full context (nullable columns for analytics tables)
FUNCTION_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("function_goid_h128", "DECIMAL(38,0)"),
    Column("urn", "VARCHAR"),
    Column("repo", "VARCHAR"),
    Column("commit", "VARCHAR"),
    Column("rel_path", "VARCHAR"),
    Column("language", "VARCHAR"),
    Column("kind", "VARCHAR"),
    Column("qualname", "VARCHAR"),
    Column("start_line", "INTEGER"),
    Column("end_line", "INTEGER"),
)

# Module entity identification
MODULE_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
    Column("module", "VARCHAR", nullable=False),
)

# Subsystem entity identification
SUBSYSTEM_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
    Column("subsystem_id", "VARCHAR", nullable=False),
)

# Test entity identification
TEST_ENTITY_COLS: Final[tuple[Column, ...]] = (
    Column("test_id", "VARCHAR", nullable=False),
    Column("test_goid_h128", "DECIMAL(38,0)"),
    Column("repo", "VARCHAR", nullable=False),
    Column("commit", "VARCHAR", nullable=False),
)

# Timestamp suffix (nullable=False)
CREATED_AT_COL: Final[tuple[Column, ...]] = (Column("created_at", "TIMESTAMP", nullable=False),)

# Timestamp suffix (nullable)
CREATED_AT_COL_NULLABLE: Final[tuple[Column, ...]] = (Column("created_at", "TIMESTAMP"),)

# Location columns (for entities with source spans)
SOURCE_SPAN_COLS: Final[tuple[Column, ...]] = (
    Column("rel_path", "VARCHAR"),
    Column("start_line", "INTEGER"),
    Column("end_line", "INTEGER"),
)

# Risk columns (used in risk factor tables)
RISK_COLS: Final[tuple[Column, ...]] = (
    Column("risk_score", "DOUBLE"),
    Column("risk_level", "VARCHAR"),
)

# Ownership columns (tags and owners)
OWNERSHIP_COLS: Final[tuple[Column, ...]] = (
    Column("tags", "JSON"),
    Column("owners", "JSON"),
)


# ---------------------------------------------------------------------------
# CompositeSchema - Profile Composition Metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompositeSchema:
    """Declare how a profile schema is composed from multiple source tables.

    This metadata enables:
    - Validation that profiles contain all expected columns from sources
    - Documentation of normalized-to-denormalized relationships
    - Detection of schema drift when source tables change

    Parameters
    ----------
    composed_of
        Tuple of source table keys (e.g., "analytics.function_metrics").
    shared_fragments
        Column fragments shared across all source tables (deduplicated).
    additional_columns
        Profile-specific columns not derived from any source table.
    column_mappings
        Dict mapping source column names to profile column names for renames.
    excluded_columns
        Set of source column names intentionally excluded from the profile.

    Examples
    --------
    >>> cs = CompositeSchema(
    ...     composed_of=("analytics.function_metrics", "analytics.function_types"),
    ...     shared_fragments=(FUNCTION_ENTITY_COLS,),
    ...     additional_columns=(Column("risk_score", "DOUBLE"),),
    ...     column_mappings={"keyword_only_params": "keyword_params"},
    ...     excluded_columns=frozenset({"created_at"}),
    ... )
    """

    composed_of: tuple[str, ...]
    shared_fragments: tuple[tuple[Column, ...], ...]
    additional_columns: tuple[Column, ...]
    column_mappings: dict[str, str]
    excluded_columns: frozenset[str]

    def _get_shared_column_names(self) -> set[str]:
        """Return column names from shared fragments.

        Returns
        -------
        set[str]
            Set of column names from all shared fragments.
        """
        names: set[str] = set()
        for fragment in self.shared_fragments:
            for col in fragment:
                names.add(col.name)
        return names

    def source_column_names(self, table_schemas: dict[str, TableSchema]) -> set[str]:
        """Return all column names expected from sources after mappings.

        Parameters
        ----------
        table_schemas
            The TABLE_SCHEMAS dict to look up source table definitions.

        Returns
        -------
        set[str]
            Set of column names expected in the profile from source tables.
        """
        shared_cols = self._get_shared_column_names()
        result: set[str] = set()

        for table_key in self.composed_of:
            schema = table_schemas.get(table_key)
            if schema is None:
                continue
            for col_name in schema.column_names():
                if col_name in shared_cols:
                    # Shared columns are added once, not per-source
                    continue
                if col_name in self.excluded_columns:
                    continue
                # Apply mapping if present
                mapped_name = self.column_mappings.get(col_name, col_name)
                result.add(mapped_name)

        # Add shared fragment columns (once)
        result.update(shared_cols)
        return result

    def validate_against_profile(
        self,
        profile_schema: TableSchema,
        table_schemas: dict[str, TableSchema],
    ) -> list[str]:
        """Validate that the profile schema matches expected composition.

        Parameters
        ----------
        profile_schema
            The TableSchema of the profile table to validate.
        table_schemas
            The TABLE_SCHEMAS dict to look up source table definitions.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        # Get expected columns from sources
        expected_from_sources = self.source_column_names(table_schemas)

        # Get actual profile columns
        actual_cols = set(profile_schema.column_names())

        # Check for missing columns (expected but not in profile)
        missing = expected_from_sources - actual_cols
        if missing:
            errors.append(f"Missing columns from sources: {sorted(missing)}")

        # Note: We don't flag extra columns as errors since profiles may have
        # computed columns or columns from sources not yet mapped

        return errors

    def get_source_for_column(
        self,
        column_name: str,
        table_schemas: dict[str, TableSchema],
    ) -> str | None:
        """Find which source table provides a given column.

        Parameters
        ----------
        column_name
            The column name to look up (profile-side name).
        table_schemas
            The TABLE_SCHEMAS dict to look up source table definitions.

        Returns
        -------
        str | None
            The table key of the source providing this column, or None.
        """
        # Check if it's a shared fragment column
        shared_cols = self._get_shared_column_names()
        if column_name in shared_cols:
            # Shared columns come from all sources; return first
            return self.composed_of[0] if self.composed_of else None

        # Check if it's an additional column
        for col in self.additional_columns:
            if col.name == column_name:
                return None  # Profile-specific, no source

        # Reverse the mapping to find source name
        reverse_mapping = {v: k for k, v in self.column_mappings.items()}
        source_name = reverse_mapping.get(column_name, column_name)

        # Search source tables
        for table_key in self.composed_of:
            schema = table_schemas.get(table_key)
            if schema is None:
                continue
            if source_name in schema.column_names():
                return table_key

        return None


__all__ = [
    "COLUMN_TYPE",
    "CREATED_AT_COL",
    "CREATED_AT_COL_NULLABLE",
    "FUNCTION_ENTITY_COLS",
    "FUNCTION_GOID_COL",
    "FUNCTION_GOID_COL_NULLABLE",
    "MODULE_ENTITY_COLS",
    "OWNERSHIP_COLS",
    "REPO_COMMIT_COLS",
    "RISK_COLS",
    "SOURCE_SPAN_COLS",
    "SUBSYSTEM_ENTITY_COLS",
    "TEST_ENTITY_COLS",
    "Column",
    "ColumnType",
    "CompositeSchema",
    "Index",
    "RowDictType",
    "RowToTuple",
    "TableSchema",
]
