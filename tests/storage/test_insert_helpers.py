"""Tests for registry-backed insert helper utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest

from codeintel.storage.gateway import insert_helpers
from codeintel.storage.gateway.protocol import DuckDBConnection
from tests._helpers.assertions import expect_equal


def test_insert_rows_normalizes_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mapping rows are normalized into schema order and dispatched."""
    captured: dict[str, object] = {}

    def fake_macro(
        _con: DuckDBConnection,
        table_key: str,
        rows: Iterable[tuple[object, ...]],
    ) -> None:
        captured["table_key"] = table_key
        captured["rows"] = list(rows)

    monkeypatch.setattr(insert_helpers, "macro_insert_rows", fake_macro)

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
    )

    expect_equal(captured["table_key"], "core.repo_map")
    expect_equal(captured["rows"], [("r1", "c1", "[]", None, "now")])


def test_insert_rows_raises_on_missing_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate missing required columns raise before hitting the macro."""

    def fake_macro(
        _con: DuckDBConnection,
        _table_key: str,
        _rows: Iterable[tuple[object, ...]],
    ) -> None:
        message = "macro_insert_rows should not be called"
        raise AssertionError(message)

    monkeypatch.setattr(insert_helpers, "macro_insert_rows", fake_macro)

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
        )
