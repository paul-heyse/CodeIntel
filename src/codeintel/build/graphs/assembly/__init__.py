"""Shared Arrow-first graph assembly helpers."""

from __future__ import annotations

from codeintel.build.graphs.assembly.collectors import (
    ColumnarBatchCollector,
    collector_for_table,
    empty_reader,
    reader_for_columnar_rows,
    reader_for_rows,
)
from codeintel.build.graphs.assembly.contracts import (
    align_reader_to_contract,
    align_table_to_contract,
    empty_contract_reader,
)
from codeintel.build.graphs.assembly.finalize import GraphFinalizeArtifacts, finalize_graph_plan
from codeintel.build.graphs.assembly.ids import payload_bytes, stable_decimal_id, stable_int_hash
from codeintel.build.graphs.assembly.kernels import (
    ExplodeEdgesResult,
    ExplodeEdgesSpec,
    explode_edges,
    explode_edges_with_aligned_lists,
    hash_struct_ordinal,
    stable_sort_for_contract,
    stable_sort_table,
)
from codeintel.build.graphs.assembly.plan_specs import (
    GraphJoinSpec,
    edge_projection,
    graph_join_spec,
    node_projection,
    ordering_keys,
    projection_for_columns,
)
from codeintel.build.graphs.assembly.plan_surface import GraphPlanSurface, graph_plan
from codeintel.build.graphs.assembly.readers import (
    drop_table_columns,
    ensure_table_columns,
    iter_normalized_tuples,
    reader_to_table,
    rename_table_columns,
    select_table_columns,
    table_rows,
    table_to_reader,
    tabular_to_reader,
    tabular_to_table,
)

__all__ = [
    "ColumnarBatchCollector",
    "ExplodeEdgesResult",
    "ExplodeEdgesSpec",
    "GraphFinalizeArtifacts",
    "GraphJoinSpec",
    "GraphPlanSurface",
    "align_reader_to_contract",
    "align_table_to_contract",
    "collector_for_table",
    "drop_table_columns",
    "edge_projection",
    "empty_contract_reader",
    "empty_reader",
    "ensure_table_columns",
    "explode_edges",
    "explode_edges_with_aligned_lists",
    "finalize_graph_plan",
    "graph_join_spec",
    "graph_plan",
    "hash_struct_ordinal",
    "iter_normalized_tuples",
    "node_projection",
    "ordering_keys",
    "payload_bytes",
    "projection_for_columns",
    "reader_for_columnar_rows",
    "reader_for_rows",
    "reader_to_table",
    "rename_table_columns",
    "select_table_columns",
    "stable_decimal_id",
    "stable_int_hash",
    "stable_sort_for_contract",
    "stable_sort_table",
    "table_rows",
    "table_to_reader",
    "tabular_to_reader",
    "tabular_to_table",
]
