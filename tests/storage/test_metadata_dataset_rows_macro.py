"""Ensure metadata.dataset_rows macro is usable for all registered datasets."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import StorageGateway


def test_dataset_rows_macro_handles_registry_datasets(fresh_gateway: StorageGateway) -> None:
    """
    Verify metadata.dataset_rows works for every dataset table_key in the registry.

    Uses a zero-row limit to avoid materializing data while exercising the macro.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for dataset_name, table_key in sorted(fresh_gateway.datasets.mapping.items()):
        try:
            con.execute(
                """
                SELECT 1
                FROM metadata.dataset_rows(?, 0, 0)
                LIMIT 0
                """,
                [table_key],
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{dataset_name} ({table_key}): {exc}")

    if failures:
        message = "dataset_rows macro failures: " + "; ".join(failures)
        pytest.fail(message)
