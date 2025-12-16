"""Tests for safe database query helpers.

This module tests the query helpers from ``codeintel.storage.queries.safe``
that provide typed access to database operations with proper error handling.

Covers all safe_* functions for 80%+ coverage:
- safe_count, safe_count_with_scope, safe_table_exists
- safe_get_columns, safe_count_nulls
- safe_min_value, safe_max_value
- safe_count_non_positive, safe_count_duplicates
- safe_not_null_fraction, safe_count_orphan_refs
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.targets import OutputTarget
from codeintel.storage.queries.safe import (
    DUCKDB_QUERY_ERRORS,
    ColumnNotFoundError,
    ForeignKeyRef,
    QueryError,
    TableNotFoundError,
    safe_count,
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
    safe_table_exists,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_true,
)
from tests._helpers.factories import make_snapshot
from tests._helpers.ingestion import (
    SeedIngestionConfig,
    TargetContextConfig,
    build_target_context_for_target,
    seed_ingestion_tables,
)

if TYPE_CHECKING:
    from types import SimpleNamespace

    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway


EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_REPO_ROOT = Path("/opt/test")
EXPECTED_FRACTION_0_5 = 0.5
EXPECTED_FRACTION_1_0 = 1.0
EXPECTED_MIN_VALUE = 5.0
EXPECTED_MAX_VALUE = 20.0


def _make_test_target(name: str = "repo_scan") -> OutputTarget:
    """Create a minimal test target."""
    return OutputTarget(
        name=name,
        module="ingestion",
        plugin=name,
        contract=EMPTY_CONTRACT,
        dependencies=(),
        description="Test target",
    )


def _ctx_for_gateway(gateway: StorageGateway, tmp_path: Path) -> TargetExecutionContext:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return build_target_context_for_target(
        _make_test_target(),
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )


def test_query_error_attributes() -> None:
    """QueryError should store table and message."""
    error = QueryError("core.test", "Something went wrong")

    expect_equal(error.table, "core.test")


def test_table_not_found_error() -> None:
    """TableNotFoundError should indicate missing table."""
    error = TableNotFoundError("core.missing", "not found")

    expect_equal(error.table, "core.missing")


def test_column_not_found_error() -> None:
    """ColumnNotFoundError should store column name."""
    error = ColumnNotFoundError("core.test", "missing_col")

    expect_equal(error.table, "core.test")
    expect_equal(error.column, "missing_col")


def test_duckdb_query_errors_is_tuple() -> None:
    """DUCKDB_QUERY_ERRORS should be a tuple of exception types."""
    expect_is_instance(DUCKDB_QUERY_ERRORS, tuple)
    expect_true(len(DUCKDB_QUERY_ERRORS) > 0)


def test_safe_count_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_count should return row count for existing tables."""
    result = safe_count(fresh_gateway, "core.modules")

    if result is None:
        pytest.fail("safe_count returned None for existing table")
    expect_true(result >= 0)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "no-dot-separator",
        "",
    ],
)
def test_safe_count_invalid_or_missing_table(fresh_gateway: StorageGateway, table_key: str) -> None:
    """safe_count should return None for invalid or missing tables."""
    result = safe_count(fresh_gateway, table_key)

    expect_is_none(result)


def test_safe_count_returns_correct_count(ingestion_ctx_bundle: SimpleNamespace) -> None:
    """safe_count should return accurate row counts."""
    seed_ingestion_tables(
        ingestion_ctx_bundle.ctx,
        SeedIngestionConfig(module_paths=["a.py", "b.py"], include_defaults=False),
    )

    result = safe_count(ingestion_ctx_bundle.gateway, "core.modules")

    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_with_scope_filters_by_snapshot(
    ingestion_ctx_bundle: SimpleNamespace,
    tmp_path: Path,
) -> None:
    """safe_count_with_scope should count only matching repo/commit."""
    bundle = ingestion_ctx_bundle
    target_repo = bundle.ctx.snapshot.repo
    target_commit = bundle.ctx.snapshot.commit
    seed_ingestion_tables(
        bundle.ctx,
        SeedIngestionConfig(module_paths=["a.py", "b.py"], include_defaults=False),
    )
    other_ctx = build_target_context_for_target(
        _make_test_target(),
        tmp_path,
        config=TargetContextConfig(
            repo_root=bundle.repo_root,
            gateway=bundle.gateway,
            snapshot=("other/repo", "other-commit"),
        ),
    )
    seed_ingestion_tables(
        other_ctx, SeedIngestionConfig(module_paths=["c.py"], include_defaults=False)
    )

    snapshot = make_snapshot(repo=target_repo, commit=target_commit, repo_root=bundle.repo_root)
    result = safe_count_with_scope(bundle.gateway, "core.modules", snapshot)

    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_with_scope_nonexistent_table(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_count_with_scope should return None for nonexistent tables."""
    snapshot = make_snapshot(repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "nonexistent.table", snapshot)

    expect_is_none(result)


def test_safe_count_with_scope_no_matches(fresh_gateway: StorageGateway) -> None:
    """safe_count_with_scope should return 0 when no rows match."""
    snapshot = make_snapshot(
        repo="nonexistent_repo", commit="nonexistent", repo_root=TEST_REPO_ROOT
    )
    result = safe_count_with_scope(fresh_gateway, "core.modules", snapshot)

    expect_equal(result, 0)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "invalid-key",
    ],
)
def test_safe_table_exists_invalid_or_missing(
    fresh_gateway: StorageGateway, table_key: str
) -> None:
    """safe_table_exists should return False for invalid or missing tables."""
    result = safe_table_exists(fresh_gateway, table_key)

    expect_false(result)


def test_safe_count_sql_injection_protection(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle potential SQL injection attempts safely."""
    result = safe_count(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    expect_is_none(result)

    result = safe_count(fresh_gateway, "'; DROP TABLE core.modules;--")
    expect_is_none(result)


def test_safe_table_exists_sql_injection_protection(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_table_exists should handle potential SQL injection attempts safely."""
    result = safe_table_exists(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    expect_false(result)

    expect_true(safe_table_exists(fresh_gateway, "core.modules"))


def test_safe_count_with_special_characters(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle special characters in table keys."""
    result = safe_count(fresh_gateway, "core.table-with-dash")
    expect_is_none(result)

    result = safe_count(fresh_gateway, "core.table with space")
    expect_is_none(result)


def test_safe_count_with_unicode(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle unicode in table keys."""
    result = safe_count(fresh_gateway, "core.tableé")
    expect_is_none(result)


def test_safe_get_columns_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_get_columns should return column names for existing tables."""
    result = safe_get_columns(fresh_gateway, "core.modules")

    expect_is_instance(result, set)
    expect_true(len(result) > 0)
    expect_in("module", result)
    expect_in("path", result)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "invalid-key",
    ],
)
def test_safe_get_columns_nonexistent_or_invalid(
    fresh_gateway: StorageGateway, table_key: str
) -> None:
    """safe_get_columns should return empty set for nonexistent or invalid tables."""
    result = safe_get_columns(fresh_gateway, table_key)

    expect_equal(result, set())


def test_safe_count_nulls_no_nulls(ingestion_ctx_bundle: SimpleNamespace) -> None:
    """safe_count_nulls should return 0 when no NULL values exist."""
    seed_ingestion_tables(
        ingestion_ctx_bundle.ctx,
        SeedIngestionConfig(module_paths=["a.py", "b.py"], include_defaults=False),
    )

    result = safe_count_nulls(ingestion_ctx_bundle.gateway, "core.modules", "module")

    expect_equal(result, 0)


def test_safe_count_nulls_with_nulls(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_count_nulls should count NULL values correctly."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={
                "core.test_nulls": [
                    (1, "a"),
                    (2, None),
                    (3, None),
                    (4, "b"),
                ]
            },
            include_defaults=False,
        ),
    )

    result = safe_count_nulls(fresh_gateway, "core.test_nulls", "value")

    expect_equal(result, EXPECTED_COUNT_2)


@pytest.mark.parametrize(
    ("table_key", "column"),
    [
        ("core.modules", "nonexistent_col"),
        ("invalid.table", "column"),
    ],
)
def test_safe_count_nulls_invalid_inputs(
    fresh_gateway: StorageGateway, table_key: str, column: str
) -> None:
    """safe_count_nulls should return 0 for invalid table or column."""
    result = safe_count_nulls(fresh_gateway, table_key, column)

    expect_equal(result, 0)


def test_safe_min_value_with_data(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_min_value should return minimum value."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_numeric": [10.5, 5.0, 20.0]},
            include_defaults=False,
        ),
    )

    result = safe_min_value(fresh_gateway, "core.test_numeric", "value")

    expect_equal(result, EXPECTED_MIN_VALUE)


def test_safe_max_value_with_data(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_max_value should return maximum value."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_numeric2": [10.5, 5.0, 20.0]},
            include_defaults=False,
        ),
    )

    result = safe_max_value(fresh_gateway, "core.test_numeric2", "value")

    expect_equal(result, EXPECTED_MAX_VALUE)


def test_safe_min_value_empty_table(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_min_value should return None for empty table."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_empty_num": []},
            include_defaults=False,
        ),
    )

    result = safe_min_value(fresh_gateway, "core.test_empty_num", "value")

    expect_is_none(result)


def test_safe_max_value_empty_table(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_max_value should return None for empty table."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_empty_num2": []},
            include_defaults=False,
        ),
    )

    result = safe_max_value(fresh_gateway, "core.test_empty_num2", "value")

    expect_is_none(result)


def test_safe_min_value_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return None for invalid table."""
    result = safe_min_value(fresh_gateway, "invalid.table", "column")

    expect_is_none(result)


def test_safe_max_value_invalid_column(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return None for invalid column."""
    result = safe_max_value(fresh_gateway, "core.modules", "nonexistent")

    expect_is_none(result)


def test_safe_count_non_positive_with_negatives(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """safe_count_non_positive should count values <= 0."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_pos": [-5.0, 0.0, 10.0, -2.0]},
            include_defaults=False,
        ),
    )

    result = safe_count_non_positive(fresh_gateway, "core.test_pos", "value")

    expect_equal(result, EXPECTED_COUNT_3)


def test_safe_count_non_positive_all_positive(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """safe_count_non_positive should return 0 when all values are positive."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            numeric_tables={"core.test_all_pos": [5.0, 10.0]},
            include_defaults=False,
        ),
    )

    result = safe_count_non_positive(fresh_gateway, "core.test_all_pos", "value")

    expect_equal(result, 0)


def test_safe_count_non_positive_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should return 0 for invalid table."""
    result = safe_count_non_positive(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0)


def test_safe_count_duplicates_with_dupes(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_count_duplicates should count duplicate values."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={
                "core.test_dupes": [
                    (1, "alice"),
                    (2, "bob"),
                    (3, "alice"),
                    (4, "alice"),
                    (5, "charlie"),
                ]
            },
            include_defaults=False,
        ),
    )

    result = safe_count_duplicates(fresh_gateway, "core.test_dupes", "name")

    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_duplicates_no_dupes(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_count_duplicates should return 0 when all values are unique."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={
                "core.test_unique": [
                    (1, "a"),
                    (2, "b"),
                    (3, "c"),
                ]
            },
            include_defaults=False,
        ),
    )

    result = safe_count_duplicates(fresh_gateway, "core.test_unique", "name")

    expect_equal(result, 0)


def test_safe_count_duplicates_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should return 0 for invalid table."""
    result = safe_count_duplicates(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0)


def test_safe_not_null_fraction_all_not_null(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_not_null_fraction should return 1.0 when all values are non-null."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={"core.test_frac1": [(1, "a"), (2, "b")]},
            include_defaults=False,
        ),
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac1", "value")

    expect_equal(result, EXPECTED_FRACTION_1_0)


def test_safe_not_null_fraction_half_null(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_not_null_fraction should return correct fraction."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={
                "core.test_frac2": [
                    (1, "a"),
                    (2, None),
                    (3, "b"),
                    (4, None),
                ]
            },
            include_defaults=False,
        ),
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac2", "value")

    expect_equal(result, EXPECTED_FRACTION_0_5)


def test_safe_not_null_fraction_all_null(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_not_null_fraction should return 0.0 when all values are null."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={"core.test_frac3": [(1, None), (2, None)]},
            include_defaults=False,
        ),
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac3", "value")

    expect_equal(result, 0.0)


def test_safe_not_null_fraction_empty_table(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_not_null_fraction should return 0.0 for empty table."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            varchar_tables={"core.test_frac_empty": []},
            include_defaults=False,
        ),
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac_empty", "value")

    expect_equal(result, 0.0)


def test_safe_not_null_fraction_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 for invalid table."""
    result = safe_not_null_fraction(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0.0)


def test_safe_count_orphan_refs_no_orphans(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_count_orphan_refs should return 0 when all refs are valid."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            foreign_keys=[
                (
                    "core.test_parent",
                    "core.test_child",
                    [(1, "a"), (2, "b")],
                    [(1, 1), (2, 2)],
                )
            ],
            include_defaults=False,
        ),
    )

    fk = ForeignKeyRef(
        source_table="core.test_child",
        source_column="parent_id",
        ref_table="core.test_parent",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    expect_equal(result, 0)


def test_safe_count_orphan_refs_with_orphans(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """safe_count_orphan_refs should count orphaned references."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            foreign_keys=[
                (
                    "core.test_parent2",
                    "core.test_child2",
                    [(1, "a")],
                    [
                        (1, 1),
                        (2, 99),
                        (3, 100),
                    ],
                )
            ],
            include_defaults=False,
        ),
    )

    fk = ForeignKeyRef(
        source_table="core.test_child2",
        source_column="parent_id",
        ref_table="core.test_parent2",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_orphan_refs_with_nulls_allowed(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """safe_count_orphan_refs should handle NULL values when allow_null=True."""
    seed_ingestion_tables(
        _ctx_for_gateway(fresh_gateway, tmp_path),
        config=SeedIngestionConfig(
            foreign_keys=[
                (
                    "core.test_parent3",
                    "core.test_child3",
                    [(1, "a")],
                    [
                        (1, 1),
                        (2, None),
                        (3, 99),
                    ],
                )
            ],
            include_defaults=False,
        ),
    )

    fk = ForeignKeyRef(
        source_table="core.test_child3",
        source_column="parent_id",
        ref_table="core.test_parent3",
        ref_column="id",
        allow_null=True,
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    expect_true(result >= 1)


def test_safe_count_orphan_refs_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should return 0 for invalid tables."""
    fk = ForeignKeyRef(
        source_table="invalid.source",
        source_column="col",
        ref_table="invalid.target",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    expect_equal(result, 0)


def test_foreign_key_ref_dataclass() -> None:
    """ForeignKeyRef should have correct attributes."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
        allow_null=False,
    )

    expect_equal(fk.source_table, "core.child")
    expect_equal(fk.source_column, "parent_id")
    expect_equal(fk.ref_table, "core.parent")
    expect_equal(fk.ref_column, "id")
    expect_false(fk.allow_null)


def test_foreign_key_ref_default_allow_null() -> None:
    """ForeignKeyRef should default allow_null to True."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
    )

    expect_true(fk.allow_null)
