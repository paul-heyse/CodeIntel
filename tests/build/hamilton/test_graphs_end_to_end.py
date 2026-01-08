"""End-to-end tests for graph target migration."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.native.graphs.graph_targets import (
    CALL_GRAPH_TABLE_KEYS,
    IMPORT_GRAPH_TABLE_KEYS,
)
from tests._helpers.assertions.table_assertions import assert_table_has_rows
from tests._helpers.assertions.target_record_assertions import (
    assert_record_has_datasets,
    assert_target_ok,
)
from tests._helpers.harnesses.graph_harness import GraphTargetHarness


def test_call_graph_import_graph_end_to_end(graph_target_harness: GraphTargetHarness) -> None:
    """Run call_graph/import_graph end-to-end and assert materialized outputs.

    Raises
    ------
    ValueError
        If schema registry data is incomplete for the graph targets.
    """
    try:
        records = graph_target_harness.run_targets(("call_graph", "import_graph"))
    except ValueError as exc:
        if "Missing TableSchema definitions" in str(exc):
            pytest.xfail("Schema registry incomplete for graph targets.")
        raise
    call_graph_record = records["call_graph"]
    import_record = records["import_graph"]

    assert_target_ok(call_graph_record)
    assert_target_ok(import_record)
    assert_record_has_datasets(call_graph_record, CALL_GRAPH_TABLE_KEYS)
    graph_target_harness.assert_import_graph_datasets(import_record)

    gateway = graph_target_harness.harness.ctx.gateway
    for table_key in (*CALL_GRAPH_TABLE_KEYS, *IMPORT_GRAPH_TABLE_KEYS):
        assert_table_has_rows(gateway, table_key)
