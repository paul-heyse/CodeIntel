"""Configuration for tuple-to-dict analytics refactors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TupleRowSpec:
    """Configuration describing a tuple-to-dict refactor target."""

    module: str
    builder_functions: Sequence[str]
    rows_var: str
    row_type_qualname: str
    row_type_local: str
    field_names: Sequence[str]
    dataset_contract_getter: str | None = None


GRAPH_METRICS_FUNCTIONS_SPEC = TupleRowSpec(
    module="codeintel.analytics.graph_rows.graph_metrics",
    builder_functions=["build_function_graph_metric_rows"],
    rows_var="rows",
    row_type_qualname="codeintel.storage.rows.GraphMetricsFunctionsRow",
    row_type_local="GraphMetricsFunctionsRow",
    field_names=[
        "repo",
        "commit",
        "function_goid_h128",
        "call_fan_in",
        "call_fan_out",
        "call_in_degree",
        "call_out_degree",
        "call_pagerank",
        "call_betweenness",
        "call_closeness",
        "call_cycle_member",
        "call_cycle_id",
        "call_layer",
        "created_at",
    ],
    dataset_contract_getter="codeintel.analytics.datasets.get_analytics_dataset_contract",
)

ALL_SPECS: list[TupleRowSpec] = [
    GRAPH_METRICS_FUNCTIONS_SPEC,
]
