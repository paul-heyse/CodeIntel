"""End-to-end tests for graph target migration."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.graph_targets import GOIDS_TABLE_KEYS
from codeintel.build.hamilton.native.graphs.import_graph import IMPORT_GRAPH_TABLE_KEYS
from tests._helpers.assertions.table_assertions import assert_table_has_rows
from tests._helpers.assertions.target_record_assertions import (
    assert_record_has_datasets,
    assert_target_ok,
)
from tests._helpers.harnesses.graph_harness import GraphTargetHarness


def test_goids_import_graph_end_to_end(graph_target_harness: GraphTargetHarness) -> None:
    """Run goids/import_graph end-to-end and assert materialized outputs."""
    records = graph_target_harness.run_targets(("goids", "import_graph"))
    goids_record = records["goids"]
    import_record = records["import_graph"]

    assert_target_ok(goids_record)
    assert_target_ok(import_record)
    assert_record_has_datasets(goids_record, GOIDS_TABLE_KEYS)
    graph_target_harness.assert_import_graph_datasets(import_record)

    gateway = graph_target_harness.harness.ctx.gateway
    for table_key in (*GOIDS_TABLE_KEYS, *IMPORT_GRAPH_TABLE_KEYS):
        assert_table_has_rows(gateway, table_key)
