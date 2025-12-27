"""Tests for bulk insert normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from codeintel.core.data_models.ids import normalize_decimal_id
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.gateway import GatewayFactory
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from pathlib import Path


def test_bulk_insert_normalizes_numpy_scalars(tmp_path: Path) -> None:
    """DuckDB bulk insert should accept numpy scalar values."""
    db_path = tmp_path / "insert.duckdb"
    gateway = GatewayFactory().file_backed(db_path).with_schema().open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics.numpy_insert_test (
                id INTEGER,
                value DOUBLE,
                created_at TIMESTAMP
            )
            """
        )

        rows = [(np.int64(1), np.float64(2.5), datetime.now(UTC))]
        inserted = gateway.policy.bulk_insert(
            "analytics.numpy_insert_test",
            rows,
            columns=["id", "value", "created_at"],
        )

        expect_equal(inserted, 1)
        records = gateway.relation_from_table_key("analytics.numpy_insert_test").df()
        expect_equal(len(records), 1)
        expect_equal(normalize_decimal_id(records.loc[0, "id"]), 1)
    finally:
        gateway.close()
