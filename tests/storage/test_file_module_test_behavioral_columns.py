"""Guard tests for profile tuple serialization lengths."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.schemas.service import get_schema_service
from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from tests._helpers.factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

get_schema_service()
_COLUMNS_BY_TABLE = load_columns_by_table()
FILE_PROFILE_COLUMNS = tuple(_COLUMNS_BY_TABLE["analytics.file_profile"])
MODULE_PROFILE_COLUMNS = tuple(_COLUMNS_BY_TABLE["analytics.module_profile"])
TEST_PROFILE_COLUMNS = tuple(_COLUMNS_BY_TABLE["analytics.test_profile"])
BEHAVIORAL_COVERAGE_COLUMNS = tuple(_COLUMNS_BY_TABLE["analytics.behavioral_coverage"])


def file_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a file profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    return serialize_row(row, FILE_PROFILE_COLUMNS)


def module_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a module profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    return serialize_row(row, MODULE_PROFILE_COLUMNS)


def serialize_test_profile_row(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a test profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    return serialize_row(row, TEST_PROFILE_COLUMNS)


def behavioral_coverage_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a behavioral coverage mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    return serialize_row(row, BEHAVIORAL_COVERAGE_COLUMNS)


def test_file_profile_tuple_length_matches_columns() -> None:
    """
    Ensure file_profile serializer aligns with declared columns.

    Raises
    ------
    AssertionError
        If tuple length diverges from column constant.
    """
    row = blank_file_profile_row()
    row["repo"] = "r"
    row["commit"] = "c"
    row["rel_path"] = "p.py"
    row["created_at"] = datetime.now(tz=UTC)
    length = len(file_profile_row_to_tuple(row))
    if length != len(FILE_PROFILE_COLUMNS):
        message = f"file_profile tuple length {length} != columns {len(FILE_PROFILE_COLUMNS)}"
        raise AssertionError(message)


def test_module_profile_tuple_length_matches_columns() -> None:
    """
    Ensure module_profile serializer aligns with declared columns.

    Raises
    ------
    AssertionError
        If tuple length diverges from column constant.
    """
    row = blank_module_profile_row()
    row["repo"] = "r"
    row["commit"] = "c"
    row["module"] = "pkg.mod"
    row["created_at"] = datetime.now(tz=UTC)
    length = len(module_profile_row_to_tuple(row))
    if length != len(MODULE_PROFILE_COLUMNS):
        message = f"module_profile tuple length {length} != columns {len(MODULE_PROFILE_COLUMNS)}"
        raise AssertionError(message)


def test_test_profile_tuple_length_matches_columns() -> None:
    """
    Ensure test_profile serializer aligns with declared columns.

    Raises
    ------
    AssertionError
        If tuple length diverges from column constant.
    """
    row = blank_test_profile_row()
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t"
    row["rel_path"] = "p.py"
    row["markers"] = []
    row["functions_covered"] = []
    row["primary_function_goids"] = []
    row["subsystems_covered"] = []
    row["created_at"] = datetime.now(tz=UTC)
    length = len(serialize_test_profile_row(row))
    if length != len(TEST_PROFILE_COLUMNS):
        message = f"test_profile tuple length {length} != columns {len(TEST_PROFILE_COLUMNS)}"
        raise AssertionError(message)


def test_behavioral_coverage_tuple_length_matches_columns() -> None:
    """
    Ensure behavioral_coverage serializer aligns with declared columns.

    Raises
    ------
    AssertionError
        If tuple length diverges from column constant.
    """
    row = blank_behavioral_coverage_row()
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t"
    row["rel_path"] = "p.py"
    row["behavior_tags"] = []
    row["tag_source"] = "heuristic"
    row["created_at"] = datetime.now(tz=UTC)
    length = len(behavioral_coverage_row_to_tuple(row))
    if length != len(BEHAVIORAL_COVERAGE_COLUMNS):
        message = (
            "behavioral_coverage tuple length "
            f"{length} != columns {len(BEHAVIORAL_COVERAGE_COLUMNS)}"
        )
        raise AssertionError(message)
