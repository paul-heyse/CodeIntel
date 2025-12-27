"""Guard tests for profile tuple serialization lengths."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import configure_schema_service
from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers.factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def _columns_by_table() -> dict[str, tuple[str, ...]]:
    columns = load_columns_by_table()
    return {key: tuple(value) for key, value in columns.items()}


def file_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a file profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.file_profile"]
    return serialize_row(row, columns)


def module_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a module profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.module_profile"]
    return serialize_row(row, columns)


def serialize_test_profile_row(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a test profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.test_profile"]
    return serialize_row(row, columns)


def behavioral_coverage_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a behavioral coverage mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.behavioral_coverage"]
    return serialize_row(row, columns)


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
    expected_len = len(_columns_by_table()["analytics.file_profile"])
    if length != expected_len:
        message = f"file_profile tuple length {length} != columns {expected_len}"
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
    expected_len = len(_columns_by_table()["analytics.module_profile"])
    if length != expected_len:
        message = f"module_profile tuple length {length} != columns {expected_len}"
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
    expected_len = len(_columns_by_table()["analytics.test_profile"])
    if length != expected_len:
        message = f"test_profile tuple length {length} != columns {expected_len}"
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
    expected_len = len(_columns_by_table()["analytics.behavioral_coverage"])
    if length != expected_len:
        message = f"behavioral_coverage tuple length {length} != columns {expected_len}"
        raise AssertionError(message)
