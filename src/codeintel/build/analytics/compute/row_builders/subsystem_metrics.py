"""Row builders for subsystem graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders.core import buffer_for_table
from codeintel.core.columnar.rows import ColumnarRowBuffer

if TYPE_CHECKING:
    from collections.abc import Mapping

SubsystemMetricRow = tuple[
    str,
    str,
    str,
    float,
    float,
    float,
    float,
    float,
    int,
    datetime,
]


@dataclass(frozen=True)
class SubsystemMetricInputs:
    """Inputs required to build subsystem graph metrics rows."""

    repo: str
    commit: str
    in_degree: Mapping[str, float]
    out_degree: Mapping[str, float]
    pagerank: Mapping[str, float]
    betweenness: Mapping[str, float]
    closeness: Mapping[str, float]
    layer: Mapping[str, int]
    created_at: datetime


def build_subsystem_graph_rows(inputs: SubsystemMetricInputs) -> ColumnarRowBuffer:
    """Construct rows for analytics.subsystem_graph_metrics.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for analytics.subsystem_graph_metrics.
    """
    buffer = buffer_for_table("analytics.subsystem_graph_metrics")
    for subsystem in inputs.pagerank:
        buffer.append(
            {
                "repo": inputs.repo,
                "commit": inputs.commit,
                "subsystem_id": subsystem,
                "import_in_degree": float(inputs.in_degree.get(subsystem, 0.0)),
                "import_out_degree": float(inputs.out_degree.get(subsystem, 0.0)),
                "import_pagerank": float(inputs.pagerank.get(subsystem, 0.0)),
                "import_betweenness": float(inputs.betweenness.get(subsystem, 0.0)),
                "import_closeness": float(inputs.closeness.get(subsystem, 0.0)),
                "import_layer": int(inputs.layer.get(subsystem, 0)),
                "created_at": inputs.created_at,
            }
        )
    return buffer


__all__ = ["SubsystemMetricInputs", "SubsystemMetricRow", "build_subsystem_graph_rows"]
