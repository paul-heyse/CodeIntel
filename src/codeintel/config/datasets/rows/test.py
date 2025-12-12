"""Test-related TypedDict row models and serializers.

This module provides TypedDict definitions for test-related DuckDB tables:
- TestCatalogRowModel for analytics.test_catalog
- TestCoverageEdgeRow for analytics.test_coverage_edges
- ProfileRowModel (TestProfileRowModel) for analytics.test_profile
- BehavioralCoverageRowModel for analytics.behavioral_coverage
- SubsystemProfileCacheRow for analytics.subsystem_profile_cache
- SubsystemCoverageCacheRow for analytics.subsystem_coverage_cache
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Final, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DATETIME = datetime

_Column = TypeVar("_Column", bound=str)


def _serialize_row(
    row: Mapping[_Column, object],
    columns: Sequence[_Column],
) -> tuple[object, ...]:
    """Serialize a mapping using a stable column sequence.

    Parameters
    ----------
    row
        Row data as a mapping from column name to value.
    columns
        Ordered sequence of column names.

    Returns
    -------
    tuple[object, ...]
        Values ordered according to ``columns``.
    """
    return tuple(row[column] for column in columns)


# ---------------------------------------------------------------------------
# Test Catalog
# ---------------------------------------------------------------------------


class TestCatalogRowModel(TypedDict):
    """Row shape for analytics.test_catalog inserts.

    Parameters
    ----------
    test_id
        Test identifier.
    test_goid_h128
        Test GOID hash.
    urn
        Uniform Resource Name.
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    qualname
        Fully qualified name.
    kind
        Test kind.
    status
        Test status.
    duration_ms
        Test duration in milliseconds.
    markers
        Pytest markers.
    parametrized
        Whether test is parametrized.
    flaky
        Whether test is flaky.
    created_at
        Creation timestamp.
    """

    test_id: str
    test_goid_h128: int | None
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    qualname: str | None
    kind: str
    status: str
    duration_ms: float
    markers: list[str]
    parametrized: bool
    flaky: bool
    created_at: datetime


def serialize_test_catalog_row(row: TestCatalogRowModel) -> tuple[object, ...]:
    """Serialize a TestCatalogRowModel into the INSERT column order.

    Parameters
    ----------
    row
        The test catalog row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_catalog INSERTs.
    """
    return (
        row["test_id"],
        row["test_goid_h128"],
        row["urn"],
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["qualname"],
        row["kind"],
        row["status"],
        row["duration_ms"],
        row["markers"],
        row["parametrized"],
        row["flaky"],
        row["created_at"],
    )


# ---------------------------------------------------------------------------
# Test Coverage Edges
# ---------------------------------------------------------------------------

TEST_COVERAGE_EDGE_COLUMNS: Final[tuple[str, ...]] = (
    "test_id",
    "test_goid_h128",
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "qualname",
    "covered_lines",
    "executable_lines",
    "coverage_ratio",
    "last_status",
    "created_at",
)


class TestCoverageEdgeRow(TypedDict):
    """Row shape for analytics.test_coverage_edges inserts.

    Parameters
    ----------
    test_id
        Test identifier.
    test_goid_h128
        Test GOID hash.
    function_goid_h128
        Function GOID hash.
    urn
        Uniform Resource Name.
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    qualname
        Fully qualified name.
    covered_lines
        Number of covered lines.
    executable_lines
        Number of executable lines.
    coverage_ratio
        Coverage ratio.
    last_status
        Last test status.
    created_at
        Creation timestamp.
    """

    test_id: str
    test_goid_h128: int | None
    function_goid_h128: int
    urn: str | None
    repo: str
    commit: str
    rel_path: str
    qualname: str | None
    covered_lines: int
    executable_lines: int
    coverage_ratio: float
    last_status: str
    created_at: datetime


def serialize_test_coverage_edge(row: TestCoverageEdgeRow) -> tuple[object, ...]:
    """Serialize a TestCoverageEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The test coverage edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_coverage_edges INSERTs.
    """
    return _serialize_row(row, TEST_COVERAGE_EDGE_COLUMNS)


# ---------------------------------------------------------------------------
# Test Profile
# ---------------------------------------------------------------------------

TEST_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
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


class ProfileRowModel(TypedDict):
    """Row shape for ``analytics.test_profile`` inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    test_id
        Test identifier.
    test_goid_h128
        Test GOID hash.
    urn
        Uniform Resource Name.
    rel_path
        Relative file path.
    module
        Module name.
    qualname
        Fully qualified name.
    language
        Programming language.
    kind
        Test kind.
    status
        Test status.
    duration_ms
        Test duration in milliseconds.
    markers
        Pytest markers (JSON).
    flaky
        Whether test is flaky.
    last_run_at
        Last run timestamp.
    functions_covered
        Functions covered (JSON).
    functions_covered_count
        Count of functions covered.
    primary_function_goids
        Primary function GOIDs (JSON).
    subsystems_covered
        Subsystems covered (JSON).
    subsystems_covered_count
        Count of subsystems covered.
    primary_subsystem_id
        Primary subsystem ID.
    assert_count
        Number of assert statements.
    raise_count
        Number of raise statements.
    uses_parametrize
        Uses pytest.mark.parametrize.
    uses_fixtures
        Uses pytest fixtures.
    io_bound
        Is I/O bound.
    uses_network
        Uses network.
    uses_db
        Uses database.
    uses_filesystem
        Uses filesystem.
    uses_subprocess
        Uses subprocess.
    flakiness_score
        Flakiness score.
    importance_score
        Importance score.
    notes
        Additional notes.
    tg_degree
        Test graph degree.
    tg_weighted_degree
        Test graph weighted degree.
    tg_proj_degree
        Test graph projection degree.
    tg_proj_weight
        Test graph projection weight.
    tg_proj_clustering
        Test graph projection clustering.
    tg_proj_betweenness
        Test graph projection betweenness.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: object
    flaky: bool | None
    last_run_at: datetime | None
    functions_covered: object
    functions_covered_count: int | None
    primary_function_goids: object
    subsystems_covered: object
    subsystems_covered_count: int | None
    primary_subsystem_id: str | None
    assert_count: int | None
    raise_count: int | None
    uses_parametrize: bool | None
    uses_fixtures: bool | None
    io_bound: bool | None
    uses_network: bool | None
    uses_db: bool | None
    uses_filesystem: bool | None
    uses_subprocess: bool | None
    flakiness_score: float | None
    importance_score: float | None
    notes: str | None
    tg_degree: int | None
    tg_weighted_degree: float | None
    tg_proj_degree: int | None
    tg_proj_weight: float | None
    tg_proj_clustering: float | None
    tg_proj_betweenness: float | None
    created_at: datetime


def serialize_test_profile_row(row: ProfileRowModel) -> tuple[object, ...]:
    """Serialize a ProfileRowModel into INSERT column order.

    Parameters
    ----------
    row
        The test profile row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by test_profile INSERTs.
    """
    return _serialize_row(row, TEST_PROFILE_COLUMNS)


# ---------------------------------------------------------------------------
# Behavioral Coverage
# ---------------------------------------------------------------------------

BEHAVIORAL_COVERAGE_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "test_id",
    "test_goid_h128",
    "rel_path",
    "qualname",
    "behavior_tags",
    "tag_source",
    "heuristic_version",
    "llm_model",
    "llm_run_id",
    "created_at",
)


class BehavioralCoverageRowModel(TypedDict):
    """Row shape for ``analytics.behavioral_coverage`` inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    test_id
        Test identifier.
    test_goid_h128
        Test GOID hash.
    rel_path
        Relative file path.
    qualname
        Fully qualified name.
    behavior_tags
        Behavior tags (JSON).
    tag_source
        Source of tags.
    heuristic_version
        Heuristic version.
    llm_model
        LLM model used.
    llm_run_id
        LLM run identifier.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    test_id: str
    test_goid_h128: int | None
    rel_path: str
    qualname: str | None
    behavior_tags: object
    tag_source: str
    heuristic_version: str | None
    llm_model: str | None
    llm_run_id: str | None
    created_at: datetime


def behavioral_coverage_row_to_tuple(row: BehavioralCoverageRowModel) -> tuple[object, ...]:
    """Serialize a BehavioralCoverageRowModel into INSERT column order.

    Parameters
    ----------
    row
        The behavioral coverage row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by behavioral_coverage INSERTs.
    """
    return _serialize_row(row, BEHAVIORAL_COVERAGE_COLUMNS)


# ---------------------------------------------------------------------------
# Subsystem Profile Cache
# ---------------------------------------------------------------------------

SUBSYSTEM_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "modules_json",
    "entrypoints_json",
    "internal_edge_count",
    "external_edge_count",
    "fan_in",
    "fan_out",
    "function_count",
    "avg_risk_score",
    "max_risk_score",
    "high_risk_function_count",
    "risk_level",
    "import_in_degree",
    "import_out_degree",
    "import_pagerank",
    "import_betweenness",
    "import_closeness",
    "import_layer",
    "created_at",
)


class SubsystemProfileCacheRow(TypedDict):
    """Row shape for ``analytics.subsystem_profile_cache`` inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    subsystem_id
        Subsystem identifier.
    name
        Subsystem name.
    description
        Subsystem description.
    module_count
        Number of modules.
    modules_json
        Modules list (JSON).
    entrypoints_json
        Entrypoints (JSON).
    internal_edge_count
        Internal edge count.
    external_edge_count
        External edge count.
    fan_in
        Fan-in count.
    fan_out
        Fan-out count.
    function_count
        Function count.
    avg_risk_score
        Average risk score.
    max_risk_score
        Maximum risk score.
    high_risk_function_count
        High risk function count.
    risk_level
        Risk level classification.
    import_in_degree
        Import in-degree.
    import_out_degree
        Import out-degree.
    import_pagerank
        Import PageRank score.
    import_betweenness
        Import betweenness centrality.
    import_closeness
        Import closeness centrality.
    import_layer
        Import layer.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    modules_json: object | None
    entrypoints_json: list[object] | dict[str, object] | None
    internal_edge_count: int | None
    external_edge_count: int | None
    fan_in: int | None
    fan_out: int | None
    function_count: int | None
    avg_risk_score: float | None
    max_risk_score: float | None
    high_risk_function_count: int | None
    risk_level: str | None
    import_in_degree: float | None
    import_out_degree: float | None
    import_pagerank: float | None
    import_betweenness: float | None
    import_closeness: float | None
    import_layer: int | None
    created_at: datetime | None


def subsystem_profile_cache_to_tuple(row: SubsystemProfileCacheRow) -> tuple[object, ...]:
    """Serialize a SubsystemProfileCacheRow into INSERT column order.

    Parameters
    ----------
    row
        The subsystem profile cache row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by subsystem_profile_cache INSERTs.
    """
    return _serialize_row(row, SUBSYSTEM_PROFILE_COLUMNS)


# ---------------------------------------------------------------------------
# Subsystem Coverage Cache
# ---------------------------------------------------------------------------

SUBSYSTEM_COVERAGE_COLUMNS: Final[tuple[str, ...]] = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "function_count",
    "risk_level",
    "avg_risk_score",
    "max_risk_score",
    "test_count",
    "passed_test_count",
    "failed_test_count",
    "skipped_test_count",
    "xfail_test_count",
    "flaky_test_count",
    "total_functions_covered",
    "avg_functions_covered",
    "max_functions_covered",
    "min_functions_covered",
    "function_coverage_ratio",
    "created_at",
)


class SubsystemCoverageCacheRow(TypedDict):
    """Row shape for ``analytics.subsystem_coverage_cache`` inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    subsystem_id
        Subsystem identifier.
    name
        Subsystem name.
    description
        Subsystem description.
    module_count
        Number of modules.
    function_count
        Function count.
    risk_level
        Risk level classification.
    avg_risk_score
        Average risk score.
    max_risk_score
        Maximum risk score.
    test_count
        Total test count.
    passed_test_count
        Passed test count.
    failed_test_count
        Failed test count.
    skipped_test_count
        Skipped test count.
    xfail_test_count
        Expected failure test count.
    flaky_test_count
        Flaky test count.
    total_functions_covered
        Total functions covered.
    avg_functions_covered
        Average functions covered.
    max_functions_covered
        Maximum functions covered.
    min_functions_covered
        Minimum functions covered.
    function_coverage_ratio
        Function coverage ratio.
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    subsystem_id: str
    name: str | None
    description: str | None
    module_count: int | None
    function_count: int | None
    risk_level: str | None
    avg_risk_score: float | None
    max_risk_score: float | None
    test_count: int | None
    passed_test_count: int | None
    failed_test_count: int | None
    skipped_test_count: int | None
    xfail_test_count: int | None
    flaky_test_count: int | None
    total_functions_covered: int | None
    avg_functions_covered: float | None
    max_functions_covered: float | None
    min_functions_covered: float | None
    function_coverage_ratio: float | None
    created_at: datetime | None


def subsystem_coverage_cache_to_tuple(row: SubsystemCoverageCacheRow) -> tuple[object, ...]:
    """Serialize a SubsystemCoverageCacheRow into INSERT column order.

    Parameters
    ----------
    row
        The subsystem coverage cache row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by subsystem_coverage_cache INSERTs.
    """
    return _serialize_row(row, SUBSYSTEM_COVERAGE_COLUMNS)


__all__ = [
    "BEHAVIORAL_COVERAGE_COLUMNS",
    "SUBSYSTEM_COVERAGE_COLUMNS",
    "SUBSYSTEM_PROFILE_COLUMNS",
    "TEST_COVERAGE_EDGE_COLUMNS",
    "TEST_PROFILE_COLUMNS",
    "BehavioralCoverageRowModel",
    "ProfileRowModel",
    "SubsystemCoverageCacheRow",
    "SubsystemProfileCacheRow",
    "TestCatalogRowModel",
    "TestCoverageEdgeRow",
    "behavioral_coverage_row_to_tuple",
    "serialize_test_catalog_row",
    "serialize_test_coverage_edge",
    "serialize_test_profile_row",
    "subsystem_coverage_cache_to_tuple",
    "subsystem_profile_cache_to_tuple",
]
