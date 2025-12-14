"""Test row builder functions for graph metrics.

Test the pure functions that construct typed row dictionaries from computed
metrics for insertion into DuckDB tables.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.compute import row_builders as row_builders_module
from codeintel.analytics.compute.row_builders import (
    FunctionGraphMetricInputs,
    SubsystemMetricInputs,
    SymbolModuleMetricInputs,
    build_function_graph_metric_rows,
    build_subsystem_graph_rows,
    build_symbol_module_rows,
    merge_component_metadata,
)

COMPONENT_ID_CACHE_VALUE = 10
COMPONENT_LAYER_CACHE_VALUE = 5
COMPONENT_ID_COMPUTED = 2
FUNCTION_ROW_COUNT = 2
CALL_FAN_IN_VALUE = 2
SECOND_FUNCTION_ID = 2
SUBSYSTEM_ROW_COUNT = 2
SUBSYSTEM_PAGERANK_API = 0.3
SUBSYSTEM_PAGERANK_CORE = 0.7
SYMBOL_MODULE_ROW_COUNT = 2


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


class FakeNeighborStats:
    """Fake NeighborStats for testing."""

    def __init__(self) -> None:
        """Initialize with empty mappings."""
        self.in_neighbors: dict[int, tuple[int, ...]] = {}
        self.out_neighbors: dict[int, tuple[int, ...]] = {}
        self.in_counts: dict[int, int] = {}
        self.out_counts: dict[int, int] = {}


class FakeComponentBundle:
    """Fake ComponentBundle for testing."""

    def __init__(self) -> None:
        """Initialize with empty mappings."""
        self.in_cycle: dict[int, bool] = {}
        self.scc_id: dict[int, int | None] = {}
        self.layer: dict[int, int | None] = {}


class FakeConfig:
    """Fake config for testing."""

    def __init__(self, repo: str = "test/repo", commit: str = "abc123") -> None:
        """Initialize with test values."""
        self.repo = repo
        self.commit = commit


class TestMergeComponentMetadata:
    """Test merge_component_metadata function."""

    @staticmethod
    def test_returns_computed_when_no_cache() -> None:
        """Verify computed values are returned when no cache is provided."""
        computed: dict[str, dict[str, int | bool]] = {
            "component_id": {"a": 1, "b": 2},
            "in_cycle": {"a": True, "b": False},
            "layer": {"a": 0, "b": 1},
        }

        result = merge_component_metadata({"a", "b"}, computed, None)

        _require(
            condition=result["component_id"] == {"a": 1, "b": 2},
            message="component_id mismatch when no cache",
        )
        _require(
            condition=result["in_cycle"] == {"a": True, "b": False},
            message="in_cycle mismatch when no cache",
        )
        _require(
            condition=result["layer"] == {"a": 0, "b": 1},
            message="layer mismatch when no cache",
        )

    @staticmethod
    def test_overlays_cached_values() -> None:
        """Verify cached values override computed values."""
        computed: dict[str, dict[str, int | bool]] = {
            "component_id": {"a": 1, "b": 2},
            "in_cycle": {"a": True, "b": False},
            "layer": {"a": 0, "b": 1},
        }
        cached: dict[str, dict[str, int | bool]] = {
            "component_id": {"a": 10},
            "in_cycle": {"a": False},
            "layer": {"a": 5},
        }

        result = merge_component_metadata({"a", "b"}, computed, cached)

        # 'a' should have cached values
        _require(
            condition=result["component_id"]["a"] == COMPONENT_ID_CACHE_VALUE,
            message="component_id cache not applied for 'a'",
        )
        _require(
            condition=result["in_cycle"]["a"] is False,
            message="in_cycle cache not applied for 'a'",
        )
        _require(
            condition=result["layer"]["a"] == COMPONENT_LAYER_CACHE_VALUE,
            message="layer cache not applied for 'a'",
        )
        # 'b' should have computed values
        _require(
            condition=result["component_id"]["b"] == COMPONENT_ID_COMPUTED,
            message="component_id mismatch for 'b'",
        )
        _require(
            condition=result["in_cycle"]["b"] is False,
            message="in_cycle mismatch for 'b'",
        )
        _require(condition=result["layer"]["b"] == 1, message="layer mismatch for 'b'")


class TestBuildFunctionGraphMetricRows:
    """Test build_function_graph_metric_rows function."""

    @staticmethod
    def test_builds_empty_list_for_no_nodes() -> None:
        """Verify empty list returned when no graph nodes."""
        inputs = FunctionGraphMetricInputs(
            repo="test/repo",
            commit="abc123",
            stats=FakeNeighborStats(),  # type: ignore[arg-type]
            centrality={"pagerank": {}, "betweenness": {}, "closeness": {}},
            components=FakeComponentBundle(),  # type: ignore[arg-type]
            graph_nodes=[],
            created_at=datetime.now(UTC),
        )

        result = build_function_graph_metric_rows(inputs)

        _require(condition=result == [], message="expected empty list for no graph nodes")

    @staticmethod
    def test_builds_rows_for_nodes() -> None:
        """Verify rows are built for each graph node."""
        stats = FakeNeighborStats()
        stats.in_neighbors = {1: (2, 3), 2: ()}
        stats.out_neighbors = {1: (4,), 2: (1,)}
        stats.in_counts = {1: 2, 2: 0}
        stats.out_counts = {1: 1, 2: 1}

        components = FakeComponentBundle()
        components.in_cycle = {1: False, 2: True}
        components.scc_id = {1: None, 2: 1}
        components.layer = {1: 0, 2: 1}

        created_at = datetime.now(UTC)

        inputs = FunctionGraphMetricInputs(
            repo="test/repo",
            commit="abc123",
            stats=stats,  # type: ignore[arg-type]
            centrality={
                "pagerank": {1: 0.5, 2: 0.3},
                "betweenness": {1: 0.1, 2: 0.2},
                "closeness": {1: 0.8, 2: 0.6},
            },
            components=components,  # type: ignore[arg-type]
            graph_nodes=[1, 2],
            created_at=created_at,
        )

        result = build_function_graph_metric_rows(inputs)

        _require(condition=len(result) == FUNCTION_ROW_COUNT, message="expected two rows")
        _require(
            condition=result[0]["function_goid_h128"] == 1,
            message="first row goid mismatch",
        )
        _require(
            condition=result[0]["call_fan_in"] == CALL_FAN_IN_VALUE,
            message="call_fan_in mismatch",
        )
        _require(
            condition=result[0]["call_fan_out"] == 1,
            message="call_fan_out mismatch",
        )
        _require(
            condition=result[0]["call_cycle_member"] is False,
            message="call_cycle_member mismatch for first row",
        )
        _require(
            condition=result[1]["function_goid_h128"] == SECOND_FUNCTION_ID,
            message="second row goid mismatch",
        )
        _require(
            condition=result[1]["call_cycle_member"] is True,
            message="call_cycle_member mismatch for second row",
        )


class TestBuildSubsystemGraphRows:
    """Test build_subsystem_graph_rows function."""

    @staticmethod
    def test_builds_rows_for_subsystems() -> None:
        """Verify rows are built for each subsystem."""
        created_at = datetime.now(UTC)

        inputs = SubsystemMetricInputs(
            repo="test/repo",
            commit="abc123",
            in_degree={"api": 5.0, "core": 3.0},
            out_degree={"api": 2.0, "core": 4.0},
            pagerank={"api": 0.3, "core": 0.7},
            betweenness={"api": 0.1, "core": 0.2},
            closeness={"api": 0.5, "core": 0.6},
            layer={"api": 0, "core": 1},
            created_at=created_at,
        )

        result = build_subsystem_graph_rows(inputs)

        _require(
            condition=len(result) == SUBSYSTEM_ROW_COUNT,
            message="expected two subsystem rows",
        )
        # Results are in pagerank iteration order (api, core)
        _require(condition=result[0][2] == "api", message="first subsystem name mismatch")
        _require(
            condition=result[0][5] == SUBSYSTEM_PAGERANK_API,
            message="first subsystem pagerank mismatch",
        )
        _require(condition=result[1][2] == "core", message="second subsystem name mismatch")
        _require(
            condition=result[1][5] == SUBSYSTEM_PAGERANK_CORE,
            message="second subsystem pagerank mismatch",
        )


class TestBuildSymbolRows:
    """Test build_symbol_module_rows and build_symbol_function_rows functions."""

    @staticmethod
    def test_builds_module_rows() -> None:
        """Verify symbol module rows are built correctly."""
        created_at = datetime.now(UTC)

        inputs = SymbolModuleMetricInputs(
            repo="test/repo",
            commit="abc123",
            centrality={
                "betweenness": {"mod_a": 0.1, "mod_b": 0.2},
                "closeness": {"mod_a": 0.5, "mod_b": 0.6},
                "eigenvector": {"mod_a": 0.3, "mod_b": 0.4},
                "harmonic": {"mod_a": 0.7, "mod_b": 0.8},
            },
            structure={
                "core_number": {"mod_a": 2, "mod_b": 3},
                "constraint": {"mod_a": 0.1, "mod_b": 0.2},
                "effective_size": {"mod_a": 1.5, "mod_b": 2.5},
                "community_id": {"mod_a": 0, "mod_b": 1},
            },
            comp_id={"mod_a": 0, "mod_b": 0},
            comp_size={"mod_a": 2, "mod_b": 2},
            created_at=created_at,
        )

        result = build_symbol_module_rows(inputs)

        _require(
            condition=len(result) == SYMBOL_MODULE_ROW_COUNT,
            message="expected two symbol module rows",
        )
        _require(condition=result[0][2] == "mod_a", message="first module name mismatch")
        _require(condition=result[1][2] == "mod_b", message="second module name mismatch")


class TestRowBuildersImport:
    """Test row builders can be imported from correct locations."""

    @staticmethod
    def test_import_from_row_builders_package() -> None:
        """Verify all exports are available from row_builders package."""
        _require(
            condition=callable(row_builders_module.build_function_graph_metric_rows),
            message="build_function_graph_metric_rows should be callable",
        )
        _require(
            condition=callable(row_builders_module.merge_component_metadata),
            message="merge_component_metadata should be callable",
        )
