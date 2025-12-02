"""Guard tests for profile tuple serialization lengths."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    behavioral_coverage_row_to_tuple,
    file_profile_row_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_profile_row,
)
from tests._helpers.row_factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
)


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
