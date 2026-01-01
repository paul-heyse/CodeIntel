"""Risk aggregation for subsystems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.query_results import (
    coerce_optional_float,
    coerce_optional_str,
    coerce_str,
    iter_tuples_from_arrow_reader,
)

MEDIUM_RISK_THRESHOLD = 0.4


@dataclass(frozen=True)
class SubsystemRisk:
    """Aggregated risk signals for a subsystem."""

    function_count: int
    total_risk: float
    max_risk: float | None
    high_risk: int
    level: str

    @property
    def avg_risk(self) -> float | None:
        """Average risk score across subsystem functions."""
        if self.function_count == 0:
            return None
        return self.total_risk / self.function_count


@dataclass
class RiskTally:
    """Mutable accumulator for subsystem risk."""

    count: int = 0
    total: float = 0.0
    max_score: float | None = None
    high: int = 0

    def add(self, score: float, *, is_high: bool) -> None:
        """Update the tally with a new score."""
        self.count += 1
        self.total += score
        self.max_score = score if self.max_score is None else max(self.max_score, score)
        if is_high:
            self.high += 1


def aggregate_risk(
    gateway: StorageGateway, snapshot: SnapshotRef, labels: dict[str, str]
) -> dict[str, SubsystemRisk]:
    """
    Aggregate risk across subsystems based on function risk factors.

    Returns
    -------
    dict[str, SubsystemRisk]
        Risk summaries keyed by subsystem label.
    """
    con = gateway.con
    risk_by_label: dict[str, SubsystemRisk] = {}
    stats: dict[str, RiskTally] = defaultdict(RiskTally)
    reader = con.execute(
        """
        SELECT rf.risk_score, rf.risk_level, m.module
        FROM analytics.goid_risk_factors rf
        LEFT JOIN analytics.function_metrics fm
          ON fm.function_goid_h128 = rf.function_goid_h128
        LEFT JOIN core.modules m
          ON m.path = fm.rel_path
        WHERE rf.repo = ? AND rf.commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    for risk_score, risk_level, module in iter_tuples_from_arrow_reader(reader):
        if module is None:
            continue
        module_name = coerce_str(module, ctx="core.modules.module")
        label = labels.get(module_name)
        if label is None:
            continue
        score = coerce_optional_float(risk_score, ctx="risk_score") or 0.0
        entry = stats[label]
        level = coerce_optional_str(risk_level, ctx="risk_level")
        entry.add(score, is_high=level == "high")

    for label, entry in stats.items():
        count = entry.count
        total = entry.total
        max_score = entry.max_score
        high = entry.high
        risk_level = "low"
        if high > 0:
            risk_level = "high"
        elif count > 0 and (total / count) >= MEDIUM_RISK_THRESHOLD:
            risk_level = "medium"
        risk_by_label[label] = SubsystemRisk(
            function_count=count,
            total_risk=total,
            max_risk=max_score,
            high_risk=high,
            level=risk_level,
        )
    return risk_by_label
