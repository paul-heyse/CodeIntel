"""Row builders for subsystem graph metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

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


def build_subsystem_graph_rows(inputs: SubsystemMetricInputs) -> list[SubsystemMetricRow]:
    """
    Construct rows for analytics.subsystem_graph_metrics.

    Returns
    -------
    list[SubsystemMetricRow]
        Rows ready for insertion into analytics.subsystem_graph_metrics.
    """
    return [
        (
            inputs.repo,
            inputs.commit,
            subsystem,
            float(inputs.in_degree.get(subsystem, 0.0)),
            float(inputs.out_degree.get(subsystem, 0.0)),
            float(inputs.pagerank.get(subsystem, 0.0)),
            float(inputs.betweenness.get(subsystem, 0.0)),
            float(inputs.closeness.get(subsystem, 0.0)),
            int(inputs.layer.get(subsystem, 0)),
            inputs.created_at,
        )
        for subsystem in inputs.pagerank
    ]


__all__ = ["SubsystemMetricInputs", "build_subsystem_graph_rows"]
