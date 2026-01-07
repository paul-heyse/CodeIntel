"""Tests for insert helper utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.core.storage import StorageContext
from codeintel.storage.warehouse import Warehouse
from tests._helpers.assertions import expect_equal

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


def test_insert_rows_normalizes_mapping(fresh_gateway: StorageGateway) -> None:
    """Ensure mapping rows are normalized into schema order and inserted."""
    now = datetime.now(tz=UTC)
    row = {
        "repo": "r1",
        "commit": "c1",
        "modules": {},
        "overlays": None,
        "generated_at": now,
    }

    warehouse = Warehouse(context=StorageContext(gateway=fresh_gateway))
    warehouse.materialize_mappings("core.repo_map", [row])

    result = fresh_gateway.con.execute(
        "SELECT repo, commit FROM core.repo_map WHERE repo = ? AND commit = ?",
        ["r1", "c1"],
    ).fetchone()
    if result is None:
        pytest.fail("Expected inserted core.repo_map row")
    expect_equal(result[0], "r1")
    expect_equal(result[1], "c1")


def test_insert_rows_raises_on_missing_column(fresh_gateway: StorageGateway) -> None:
    """Validate missing required columns raise before insert."""
    warehouse = Warehouse(context=StorageContext(gateway=fresh_gateway))
    with pytest.raises(ValueError, match="Missing column generated_at"):
        warehouse.materialize_mappings(
            "core.repo_map",
            [
                {
                    "repo": "r1",
                    "commit": "c1",
                    "modules": {},
                    "overlays": None,
                },
            ],
        )
