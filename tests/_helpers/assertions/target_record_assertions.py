"""Assertion helpers for Hamilton TargetRunRecord values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandera.errors import SchemaErrors

from codeintel.storage.validation.pandera_df import get_pandera_schema, validate_df
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.hamilton.run_records import TargetRunRecord
    from codeintel.storage.gateway import StorageGateway


def assert_target_ok(record: TargetRunRecord, *, expected_status: str = "succeeded") -> None:
    """Assert a TargetRunRecord has the expected status."""
    expect_equal(record.status, expected_status, label="target_status")
    if expected_status == "succeeded":
        expect_true(record.success, message="Expected record.success to be True.")


def assert_record_row_counts(
    record: TargetRunRecord,
    expected: Mapping[str, int],
) -> None:
    """Assert row counts contain expected table counts."""
    for table_key, count in expected.items():
        actual = record.row_counts.get(table_key)
        expect_equal(actual, count, label=table_key)


def assert_record_has_datasets(record: TargetRunRecord, keys: Iterable[str]) -> None:
    """Assert datasets contain the expected table keys."""
    dataset_keys = {ds.table_key for ds in record.datasets}
    for key in keys:
        expect_in(key, dataset_keys, label="dataset_key")


def assert_record_has_artifacts(record: TargetRunRecord, names: Iterable[str]) -> None:
    """Assert artifacts contain the expected names."""
    artifact_names = {artifact.name for artifact in record.artifacts}
    for name in names:
        expect_in(name, artifact_names, label="artifact_name")


def assert_table_schema_valid(gateway: StorageGateway, table_key: str) -> None:
    """Assert a table validates against the registered schema.

    Raises
    ------
    AssertionError
        If the table is missing a schema or fails validation.
    """
    schema = get_pandera_schema(table_key)
    if schema is None:
        message = f"No schema registered for {table_key}"
        raise AssertionError(message)
    df = gateway.con.table(table_key).df()
    try:
        validate_df(table_key, df, mode="strict")
    except SchemaErrors as exc:
        message = f"Schema validation failed for {table_key}: {exc}"
        raise AssertionError(message) from exc


def assert_record_schemas_valid(
    gateway: StorageGateway,
    record: TargetRunRecord,
    *,
    table_keys: Iterable[str] | None = None,
) -> None:
    """Assert schemas for datasets referenced by a TargetRunRecord.

    Raises
    ------
    AssertionError
        If a dataset is missing or fails schema validation.
    """
    dataset_keys = {dataset.table_key for dataset in record.datasets}
    selected = set(table_keys) if table_keys is not None else dataset_keys
    missing = selected - dataset_keys
    if missing:
        message = f"Record missing dataset refs for schema validation: {sorted(missing)}"
        raise AssertionError(message)
    for table_key in sorted(selected):
        assert_table_schema_valid(gateway, table_key)


__all__ = [
    "assert_record_has_artifacts",
    "assert_record_has_datasets",
    "assert_record_row_counts",
    "assert_record_schemas_valid",
    "assert_table_schema_valid",
    "assert_target_ok",
]
