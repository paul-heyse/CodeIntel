"""Native Hamilton implementation for test_profile target.

This module provides the Hamilton native nodes for test profiles:
- `t__test_profile__compute`: Pure compute node for test profiles
- `test_profile__rows`: SaveToDecorator node for materialization

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.testing.profiles.builder import (
    TestProfileBuildResult,
    build_test_profile_result,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.targets import TargetGraph
from codeintel.hamilton.records import TargetRunRecord

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


# Column names in schema order for test_profile
TEST_PROFILE_COLS = (
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "urn",
    "rel_path",
    "module",
    "qualname",
    "language",
    "kind",
    "status",
    "duration_ms",
    "markers",
    "flaky",
    "last_run_at",
    "functions_covered",
    "functions_covered_count",
    "primary_function_goids",
    "subsystems_covered",
    "subsystems_covered_count",
    "primary_subsystem_id",
    "assert_count",
    "raise_count",
    "uses_parametrize",
    "uses_fixtures",
    "io_bound",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "flakiness_score",
    "importance_score",
    "notes",
    "tg_degree",
    "tg_weighted_degree",
    "tg_proj_degree",
    "tg_proj_weight",
    "tg_proj_clustering",
    "tg_proj_betweenness",
    "created_at",
)


@dataclass(frozen=True)
class TestProfileComputeResult:
    """Result from test profile computation.

    Attributes
    ----------
    result
        Build result containing row models, or None if failed.
    error
        Error message if computation failed.
    """

    result: TestProfileBuildResult | None
    error: str | None = None


def _row_to_tuple(row: Mapping[str, object], cols: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a dict row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping from column name to value.
    cols
        Column names in the desired order.

    Returns
    -------
    tuple[object, ...]
        Values in column order.
    """
    return tuple(row.get(col) for col in cols)


@tag(domain="analytics", target="test_profile", node_type="compute")
def t__test_profile__compute(
    env: BuildEnv,
    t__coverage_test_edges: TargetRunRecord,
) -> TestProfileComputeResult:
    """Build per-test profiles with coverage and subsystem context.

    Compute test profile rows for DAG-visible materialization.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__coverage_test_edges
        Upstream coverage_test_edges target result (for dependency).

    Returns
    -------
    TestProfileComputeResult
        Result containing row models for materialization.

    Notes
    -----
    The profiles include:
    - Coverage context for each test
    - Subsystem associations
    - Test metadata aggregation
    """
    if t__coverage_test_edges.status != "succeeded":
        return TestProfileComputeResult(
            result=None,
            error=f"Upstream coverage_test_edges target failed: {t__coverage_test_edges.error}",
        )

    try:
        # Build test profiles (pure compute, no persistence)
        build_result = build_test_profile_result(env.gateway, env.snapshot)

        return TestProfileComputeResult(result=build_result)

    except Exception as exc:
        log.exception("Test profile computation failed")
        return TestProfileComputeResult(
            result=None,
            error=str(exc),
        )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_profile"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_profile"),
    table_key=value("analytics.test_profile"),
    columns=value(TEST_PROFILE_COLS),
)
@tag(domain="analytics", target="test_profile", node_type="compute")
def test_profile__rows(
    t__test_profile__compute: TestProfileComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_profile table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the test_profile table, or None if skipped/failed.
    """
    if t__test_profile__compute.result is None:
        return None
    if t__test_profile__compute.result.rows is None:
        return None
    return tuple(
        _row_to_tuple(row, TEST_PROFILE_COLS) for row in t__test_profile__compute.result.rows
    )


@tag(domain="analytics", target="test_profile", node_type="materialize")
def t__test_profile(
    env: BuildEnv,
    graph: TargetGraph,
    t__test_profile__compute: TestProfileComputeResult,
    m__analytics__test_profile: dict[str, Any],
) -> TargetRunRecord:
    """Materialize test profile target.

    Converts materialization metadata into a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__test_profile__compute
        Computed test profile result from the compute node.
    m__analytics__test_profile
        Materialization metadata for test_profile table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__test_profile__compute.error:
        return TargetRunRecord(
            target="test_profile",
            plugin_name="native:test_profile",
            status="failed",
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=t__test_profile__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="test_profile",
        expected_table_key="analytics.test_profile",
        materialization=m__analytics__test_profile,
    )


__all__ = [
    "TEST_PROFILE_COLS",
    "TestProfileComputeResult",
    "t__test_profile",
    "t__test_profile__compute",
    "test_profile__rows",
]
