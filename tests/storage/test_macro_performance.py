"""Lightweight performance guardrails for normalized macros."""

from __future__ import annotations

from time import perf_counter

import pytest

from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import StorageGateway

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("table_key", ["graph.call_graph_edges", "analytics.function_metrics"])
def test_normalized_macro_latency_smoke(fresh_gateway: StorageGateway, table_key: str) -> None:
    """
    Ensure representative normalized macros run quickly on empty data.

    The threshold is generous to avoid flakiness but still catches gross regressions.
    """
    con = fresh_gateway.con
    macro = NORMALIZED_MACROS[table_key]
    start = perf_counter()
    _ = con.sql(
        f"SELECT * FROM {macro}(?, ?, ?)",  # noqa: S608 - trusted macro name
        params=[table_key, 0, 0],
    ).fetchall()
    duration = perf_counter() - start
    if duration > 1.0:
        pytest.fail(f"Macro {macro} exceeded latency threshold: {duration:.3f}s")
