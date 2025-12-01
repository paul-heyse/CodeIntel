"""Ensure normalized macros execute for all registered datasets."""

from __future__ import annotations

import pytest

from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.sql_helpers import safe_macro_call

pytestmark = pytest.mark.smoke


def test_normalized_macros_execute(fresh_gateway: StorageGateway) -> None:
    """
    Every normalized macro should execute with a zero-row limit.

    This guards against missing macro definitions or signature drift.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        try:
            sql, params = safe_macro_call(
                macro, [table_key, 0], allowed=set(NORMALIZED_MACROS.values())
            )
            con.execute(sql, params)
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{table_key} via {macro}: {exc}")
    if failures:
        message = "Normalized macro failures: " + "; ".join(failures)
        pytest.fail(message)
