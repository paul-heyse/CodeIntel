"""Risk aggregation for subsystems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_float, coerce_optional_str

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

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
    snapshot: SnapshotRef,
    labels: dict[str, str],
    *,
    risk_factors_frame: pl.DataFrame | None = None,
    function_metrics_frame: pl.DataFrame | None = None,
    modules_frame: pl.DataFrame | None = None,
) -> dict[str, SubsystemRisk]:
    """
    Aggregate risk across subsystems based on function risk factors.

    Returns
    -------
    dict[str, SubsystemRisk]
        Risk summaries keyed by subsystem label.
    """
    if not _has_required_frames(risk_factors_frame, function_metrics_frame, modules_frame):
        return {}
    module_by_path = _module_by_path_from_frame(
        modules_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    function_module = _function_module_map(
        function_metrics_frame,
        module_by_path,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    stats = _risk_stats_from_frames(
        risk_factors_frame,
        function_module,
        labels,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return _risk_by_label(stats)


def _filter_frame_by_snapshot(
    frame: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(pl.col("repo") == repo)
    if "commit" in filtered.columns:
        filtered = filtered.filter(pl.col("commit") == commit)
    return filtered


def _has_required_frames(*frames: pl.DataFrame | None) -> bool:
    return all(frame is not None and not frame.is_empty() for frame in frames)


def _module_by_path_from_frame(
    modules_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    if modules_frame is None or modules_frame.is_empty():
        return module_by_path
    modules_filtered = _filter_frame_by_snapshot(modules_frame, repo=repo, commit=commit)
    for row in modules_filtered.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and module is not None:
            module_by_path[path] = str(module)
    return module_by_path


def _function_module_map(
    function_metrics_frame: pl.DataFrame | None,
    module_by_path: dict[str, str],
    *,
    repo: str,
    commit: str,
) -> dict[int, str]:
    function_module: dict[int, str] = {}
    if function_metrics_frame is None or function_metrics_frame.is_empty():
        return function_module
    metrics_filtered = _filter_frame_by_snapshot(
        function_metrics_frame,
        repo=repo,
        commit=commit,
    )
    for row in metrics_filtered.iter_rows(named=True):
        goid = normalize_decimal_id(row.get("function_goid_h128"))
        rel_path = row.get("rel_path")
        if goid is None or not isinstance(rel_path, str):
            continue
        module_name = module_by_path.get(rel_path)
        if module_name is not None:
            function_module[goid] = module_name
    return function_module


def _risk_stats_from_frames(
    risk_factors_frame: pl.DataFrame | None,
    function_module: dict[int, str],
    labels: dict[str, str],
    *,
    repo: str,
    commit: str,
) -> dict[str, RiskTally]:
    stats: dict[str, RiskTally] = defaultdict(RiskTally)
    if risk_factors_frame is None or risk_factors_frame.is_empty():
        return stats
    risk_filtered = _filter_frame_by_snapshot(risk_factors_frame, repo=repo, commit=commit)
    for row in risk_filtered.iter_rows(named=True):
        goid = normalize_decimal_id(row.get("function_goid_h128"))
        if goid is None:
            continue
        module_name = function_module.get(goid)
        if module_name is None:
            continue
        label = labels.get(module_name)
        if label is None:
            continue
        score = coerce_optional_float(row.get("risk_score"), ctx="risk_score") or 0.0
        level = coerce_optional_str(row.get("risk_level"), ctx="risk_level")
        stats[label].add(score, is_high=level == "high")
    return stats


def _risk_level(entry: RiskTally) -> str:
    if entry.high > 0:
        return "high"
    if entry.count > 0 and (entry.total / entry.count) >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _risk_by_label(stats: dict[str, RiskTally]) -> dict[str, SubsystemRisk]:
    risk_by_label: dict[str, SubsystemRisk] = {}
    for label, entry in stats.items():
        risk_by_label[label] = SubsystemRisk(
            function_count=entry.count,
            total_risk=entry.total,
            max_risk=entry.max_score,
            high_risk=entry.high,
            level=_risk_level(entry),
        )
    return risk_by_label
