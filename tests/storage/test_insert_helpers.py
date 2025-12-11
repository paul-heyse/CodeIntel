"""Tests for registry-backed insert helper utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from codeintel.storage.gateway import insert_helpers
from tests._helpers.assertions import expect_equal
from tests._helpers.storage import capture_executor

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection


def test_insert_rows_normalizes_mapping() -> None:
    """Ensure mapping rows are normalized into schema order and dispatched."""
    executor, calls = capture_executor()

    row = {
        "repo": "r1",
        "commit": "c1",
        "modules": "[]",
        "overlays": None,
        "generated_at": "now",
    }

    insert_helpers.insert_rows(
        cast("DuckDBConnection", object()),
        "core.repo_map",
        [row],
        executor=executor,
    )

    expect_equal(len(calls), 1)
    expect_equal(calls[0].table, "core.repo_map")
    expect_equal(calls[0].rows, [("r1", "c1", "[]", None, "now")])


def test_insert_rows_raises_on_missing_column() -> None:
    """Validate missing required columns raise before hitting the macro."""
    executor, calls = capture_executor()

    with pytest.raises(ValueError, match="Missing column generated_at"):
        insert_helpers.insert_rows(
            cast("DuckDBConnection", object()),
            "core.repo_map",
            [
                {
                    "repo": "r1",
                    "commit": "c1",
                    "modules": "[]",
                    "overlays": None,
                },
            ],
            executor=executor,
        )
    expect_equal(len(calls), 0)
