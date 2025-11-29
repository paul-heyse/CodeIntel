"""Coverage aggregation helpers for test profiles."""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.analytics.tests_profiles.legacy import legacy
from codeintel.analytics.tests_profiles.types import (
    FunctionCoverageEntryProtocol,
    SubsystemCoverageEntryProtocol,
    TestGraphMetricsProtocol,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.storage.gateway import DuckDBConnection


def aggregate_test_coverage_by_function(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig,
) -> Mapping[str, FunctionCoverageEntryProtocol]:
    """
    Aggregate coverage signals per test→function edge.

    Returns
    -------
    Mapping[str, object]
        Coverage entries keyed by ``test_id``.
    """
    return legacy.load_functions_covered(con, cfg.repo, cfg.commit)


def aggregate_test_coverage_by_subsystem(
    con: DuckDBConnection,
    cfg: BehavioralCoverageStepConfig,
) -> Mapping[str, SubsystemCoverageEntryProtocol]:
    """
    Aggregate subsystem coverage signals per test.

    Returns
    -------
    Mapping[str, object]
        Coverage entries keyed by ``test_id``.
    """
    return legacy.load_subsystems_covered(con, cfg.repo, cfg.commit)


def load_test_graph_metrics(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig,
) -> Mapping[str, TestGraphMetricsProtocol]:
    """
    Load test graph metrics used in importance calculations.

    Returns
    -------
    Mapping[str, object]
        Metrics keyed by ``test_id``.
    """
    return legacy.load_test_graph_metrics_public(con, cfg.repo, cfg.commit)


def load_test_records(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig | BehavioralCoverageStepConfig,
) -> list[TestRecord]:
    """
    Load test catalog records for the configured snapshot.

    Returns
    -------
    list[TestRecord]
        Test records for the snapshot.
    """
    return legacy.load_test_records_public(con, cfg.repo, cfg.commit)
