"""Tests for codeintel.config.datasets.primitives module."""

from __future__ import annotations

import pytest

from codeintel.config.datasets.primitives import (
    CREATED_AT_COL,
    FUNCTION_ENTITY_COLS,
    FUNCTION_GOID_COL,
    MODULE_ENTITY_COLS,
    REPO_COMMIT_COLS,
    Column,
    CompositeSchema,
    Index,
    TableSchema,
)

CREATED_AT_COLUMN_COUNT = 1
FUNCTION_GOID_COLUMN_COUNT = 1
MIN_FUNCTION_ENTITY_COLUMNS = 6
MODULE_ENTITY_COLUMN_COUNT = 3
REPO_COMMIT_COLUMN_COUNT = 2


def require(condition: object, message: str) -> None:
    """Fail the current test with a descriptive message."""
    if not condition:
        pytest.fail(message)


def test_column_creation() -> None:
    """Verify Column dataclass behaves correctly."""
    col = Column(name="test_col", type="VARCHAR", nullable=True, description="Test column")
    require(col.name == "test_col", "name should store provided value")
    require(col.type == "VARCHAR", "type should store provided value")
    require(col.nullable is True, "nullable should store provided value")
    require(col.description == "Test column", "description should store provided value")


def test_column_defaults() -> None:
    """Verify Column has correct default values."""
    col = Column(name="simple", type="INTEGER")
    require(col.nullable is True, "nullable should default to True")
    require(col.description is None, "description should default to None")


def test_index_creation() -> None:
    """Verify Index dataclass behaves correctly."""
    idx = Index(name="idx_test", columns=("col1", "col2"), unique=True)
    require(idx.name == "idx_test", "name should store provided value")
    require(idx.columns == ("col1", "col2"), "columns should store provided tuple")
    require(idx.unique is True, "unique should store provided value")


def test_index_defaults() -> None:
    """Verify Index has correct default values."""
    idx = Index(name="idx_simple", columns=("col1",))
    require(idx.unique is False, "unique should default to False")


def test_table_schema_fq_name() -> None:
    """Verify TableSchema.fq_name property."""
    schema = TableSchema(
        schema="analytics",
        name="test_table",
        columns=[Column("id", "INTEGER")],
    )
    require(schema.fq_name == "analytics.test_table", "fq_name should match schema.name")


def test_table_schema_column_names() -> None:
    """Verify TableSchema.column_names method."""
    schema = TableSchema(
        schema="core",
        name="sample",
        columns=[
            Column("id", "INTEGER"),
            Column("name", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ],
    )
    require(
        schema.column_names() == ["id", "name", "created_at"],
        "column_names should list columns in order",
    )


def test_table_schema_with_indexes_and_primary_key() -> None:
    """Verify TableSchema with primary key and indexes."""
    schema = TableSchema(
        schema="test",
        name="indexed",
        columns=[
            Column("id", "INTEGER", nullable=False),
            Column("value", "VARCHAR"),
        ],
        primary_key=("id",),
        indexes=(Index("idx_value", ("value",)),),
    )
    require(schema.primary_key == ("id",), "primary_key should store provided tuple")
    require(len(schema.indexes) == 1, "indexes should contain provided index")
    require(schema.indexes[0].name == "idx_value", "index name should match provided value")


def test_column_fragments_exist() -> None:
    """Verify column fragment constants are defined and have expected structure."""
    require(
        len(REPO_COMMIT_COLS) == REPO_COMMIT_COLUMN_COUNT,
        "REPO_COMMIT_COLS should contain repo and commit",
    )
    require(REPO_COMMIT_COLS[0].name == "repo", "first repo commit column should be repo")
    require(REPO_COMMIT_COLS[1].name == "commit", "second repo commit column should be commit")

    require(
        len(FUNCTION_GOID_COL) == FUNCTION_GOID_COLUMN_COUNT,
        "FUNCTION_GOID_COL should contain one column",
    )
    require(
        FUNCTION_GOID_COL[0].name == "function_goid_h128",
        "FUNCTION_GOID_COL should expose function_goid_h128",
    )

    require(
        len(FUNCTION_ENTITY_COLS) >= MIN_FUNCTION_ENTITY_COLUMNS,
        "FUNCTION_ENTITY_COLS should be populated",
    )
    require(
        len(MODULE_ENTITY_COLS) == MODULE_ENTITY_COLUMN_COUNT,
        "MODULE_ENTITY_COLS should contain three entries",
    )

    require(
        len(CREATED_AT_COL) == CREATED_AT_COLUMN_COUNT,
        "CREATED_AT_COL should contain one column",
    )
    require(CREATED_AT_COL[0].name == "created_at", "created_at column should be present")


def test_composite_schema_source_column_names() -> None:
    """Verify CompositeSchema.source_column_names method."""
    cs = CompositeSchema(
        composed_of=("source1", "source2"),
        shared_fragments=(REPO_COMMIT_COLS,),
        additional_columns=(Column("extra", "VARCHAR"),),
        column_mappings={"old_name": "new_name"},
        excluded_columns=frozenset({"excluded_col"}),
    )

    # Create mock table schemas for testing
    mock_schemas: dict[str, TableSchema] = {
        "source1": TableSchema(
            schema="test",
            name="source1",
            columns=[
                Column("repo", "VARCHAR"),
                Column("commit", "VARCHAR"),
                Column("col_a", "INTEGER"),
            ],
        ),
        "source2": TableSchema(
            schema="test",
            name="source2",
            columns=[
                Column("repo", "VARCHAR"),
                Column("commit", "VARCHAR"),
                Column("old_name", "VARCHAR"),
                Column("excluded_col", "VARCHAR"),
            ],
        ),
    }

    result = cs.source_column_names(mock_schemas)

    # Should include shared columns
    require("repo" in result, "repo should be included from shared fragments")
    require("commit" in result, "commit should be included from shared fragments")

    # Should include source-specific columns
    require("col_a" in result, "col_a should be included from source1")

    # Should apply mapping
    require("new_name" in result, "column mapping should apply to old_name")
    require("old_name" not in result, "old_name should be replaced by new_name")

    # Should exclude specified columns
    require("excluded_col" not in result, "excluded_col should be removed")


def test_composite_schema_get_source_for_column() -> None:
    """Verify CompositeSchema.get_source_for_column method."""
    cs = CompositeSchema(
        composed_of=("source1",),
        shared_fragments=(REPO_COMMIT_COLS,),
        additional_columns=(Column("extra", "VARCHAR"),),
        column_mappings={},
        excluded_columns=frozenset(),
    )

    mock_schemas: dict[str, TableSchema] = {
        "source1": TableSchema(
            schema="test",
            name="source1",
            columns=[
                Column("repo", "VARCHAR"),
                Column("commit", "VARCHAR"),
                Column("unique_col", "INTEGER"),
            ],
        ),
    }

    # Shared column returns first source
    require(
        cs.get_source_for_column("repo", mock_schemas) == "source1",
        "repo should map to first composed source",
    )

    # Additional column returns None (profile-specific)
    require(
        cs.get_source_for_column("extra", mock_schemas) is None,
        "additional columns should return None source",
    )

    # Source-specific column returns that source
    require(
        cs.get_source_for_column("unique_col", mock_schemas) == "source1",
        "unique_col should map to source1",
    )
