"""Tests for insert helper utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.gateway import insert_helpers
from tests._helpers.assertions import expect_equal

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


def test_insert_rows_normalizes_mapping(fresh_gateway: StorageGateway) -> None:
    """Ensure mapping rows are normalized into schema order and inserted."""
    now = datetime.now(tz=UTC)
    row = {
        "repo": "r1",
        "commit": "c1",
        "modules": "[]",
        "overlays": None,
        "generated_at": now,
    }

    insert_helpers.insert_rows(
        fresh_gateway,
        "core.repo_map",
        [row],
    )

    result = fresh_gateway.con.execute(
        (
            "SELECT repo, commit, CAST(modules AS VARCHAR), overlays, generated_at "
            "FROM core.repo_map WHERE repo = ? AND commit = ?"
        ),
        ["r1", "c1"],
    ).fetchone()
    if result is None:
        pytest.fail("Expected inserted core.repo_map row")
    expect_equal(result[0], "r1")
    expect_equal(result[1], "c1")


def test_insert_rows_raises_on_missing_column(fresh_gateway: StorageGateway) -> None:
    """Validate missing required columns raise before insert."""
    with pytest.raises(ValueError, match="Missing column generated_at"):
        insert_helpers.insert_rows(
            fresh_gateway,
            "core.repo_map",
            [
                {
                    "repo": "r1",
                    "commit": "c1",
                    "modules": "[]",
                    "overlays": None,
                },
            ],
        )
