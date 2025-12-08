"""Typed config factories for tests.

These helpers centralize builder construction using ``SnapshotInit`` and the
typed ``analytics``/``graphs`` facets to avoid legacy delegation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, Unpack

from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.config.primitives import BuildLayoutOptions, SnapshotRef
from codeintel.config.steps_analytics import (
    CoverageAnalyticsStepConfig,
    DataModelUsageStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    ProfilesAnalyticsStepConfig,
    SubsystemsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)


def _snapshot_init(snapshot: SnapshotRef | SnapshotInit) -> SnapshotInit:
    if isinstance(snapshot, SnapshotInit):
        return snapshot
    return SnapshotInit(
        repo=snapshot.repo,
        commit=snapshot.commit,
        repo_root=snapshot.repo_root,
        branch=snapshot.branch,
    )


def _builder(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
) -> ConfigBuilder:
    return ConfigBuilder.from_snapshot(snapshot=_snapshot_init(snapshot), layout=layout)


def coverage_analytics_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
) -> CoverageAnalyticsStepConfig:
    return _builder(snapshot, layout=layout).analytics.coverage_analytics()


def build_test_coverage_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
    coverage_file: Path | None = None,
) -> TestCoverageStepConfig:
    return _builder(snapshot, layout=layout).analytics.test_coverage(coverage_file=coverage_file)


def build_test_profile_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
) -> TestProfileStepConfig:
    return _builder(snapshot, layout=layout).analytics.test_profile()


def profiles_analytics_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
) -> ProfilesAnalyticsStepConfig:
    return _builder(snapshot, layout=layout).analytics.profiles_analytics()


def data_model_usage_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
    max_examples_per_usage: int = 2,
) -> DataModelUsageStepConfig:
    return _builder(snapshot, layout=layout).analytics.data_model_usage(
        max_examples_per_usage=max_examples_per_usage
    )


def function_analytics_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
    fail_on_missing_spans: bool = False,
) -> FunctionAnalyticsStepConfig:
    return _builder(snapshot, layout=layout).analytics.function_analytics(
        fail_on_missing_spans=fail_on_missing_spans
    )


def function_history_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
    min_lines_threshold: int | None = None,
) -> FunctionHistoryStepConfig:
    cfg = _builder(snapshot, layout=layout).analytics.function_history()
    if min_lines_threshold is None:
        return cfg
    return replace(cfg, min_lines_threshold=min_lines_threshold)


def history_timeseries_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    commits: Iterable[str],
    entity_kind: str = "function",
    layout: BuildLayoutOptions | None = None,
) -> HistoryTimeseriesStepConfig:
    return _builder(snapshot, layout=layout).analytics.history_timeseries(
        commits=tuple(commits),
        entity_kind=entity_kind,
    )


class SubsystemsOverrides(TypedDict, total=False):
    min_modules: int
    max_subsystems: int
    import_weight: float
    symbol_weight: float
    config_weight: float


def subsystems_cfg(
    snapshot: SnapshotRef | SnapshotInit,
    *,
    layout: BuildLayoutOptions | None = None,
    **overrides: Unpack[SubsystemsOverrides],
) -> SubsystemsStepConfig:
    return _builder(snapshot, layout=layout).analytics.subsystems(**overrides)


__all__ = [
    "build_test_coverage_cfg",
    "build_test_profile_cfg",
    "coverage_analytics_cfg",
    "data_model_usage_cfg",
    "function_analytics_cfg",
    "function_history_cfg",
    "history_timeseries_cfg",
    "profiles_analytics_cfg",
    "subsystems_cfg",
]
