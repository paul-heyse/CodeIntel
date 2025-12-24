"""Tests for AST metrics and git churn hotspots computation.

This module tests:
- FileChurn dataclass for aggregating per-file churn
- compute_hotspot_rows for computing hotspot scores
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.hotspots import FileChurn, compute_hotspot_rows, parse_git_log_lines
from codeintel.build.hamilton.native.analytics.hotspots import collect_git_file_stats
from codeintel.ingestion.engine.infrastructure import (
    ToolName,
    ToolRunner,
    ToolRunOptions,
    ToolRunResult,
)
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_false,
    expect_in,
    expect_is_not_none,
    expect_length,
    expect_not_in,
    expect_true,
)
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.factories import make_snapshot
from tests._helpers.graphs import canonical_ast_artifacts
from tests._helpers.fixtures.repos import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)
from tests._helpers.rows import AstMetricSeed
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds.ast_metrics import MEDIUM_COMPLEXITY

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext


EXPECTED_COMMIT_COUNT = 2
EXPECTED_AUTHOR_COUNT = 2
EXPECTED_AUTHOR_COUNT_MULTI = 3
EXPECTED_LINES_ADDED = 30
EXPECTED_LINES_DELETED = 10
PARSE_COMMITS_1 = 1
PARSE_COMMITS_2 = 2
PARSE_AUTHORS_1 = 1
PARSE_AUTHORS_2 = 2
PARSE_LINES_ADDED_10 = 10
PARSE_LINES_ADDED_15 = 15
PARSE_LINES_DELETED_0 = 0
PARSE_LINES_DELETED_3 = 3
PARSE_LINES_DELETED_5 = 5
PARSE_FILES_2 = 2
HOTSPOT_SCORE_THRESHOLD = 0.0
EXPECTED_FILE_COUNT_MULTI = 3
EXPECTED_COMPLEXITY = 10.0
EXPECTED_SUMMARY_KEYS = 4
FUNC_A_LINES = (1, 3)
FUNC_B_LINES = (1, 6)
FUNC_C_LINES = (1, 2)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    message = f"Expected numeric value, got {value!r}"
    raise AssertionError(message)


HELPER_LINES = (1, 2)


@pytest.fixture
def ast_metrics_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a seeded TestContext with AST metrics data.

    Yields
    ------
    Iterator[TestContext]
        Context seeded with core and AST metrics packs.
    """
    ctx = TestScenario.with_ast_metrics().build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def ast_lookup(ast_metrics_ctx: TestContext) -> dict[int, FunctionAst]:
    """Apply canonical AST artifacts and ensure GOIDs are registered.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping from GOID to FunctionAst for seeded functions.
    """
    artifacts = canonical_ast_artifacts(ast_metrics_ctx)
    if (
        ast_metrics_ctx.query_count(
            "core.goids", f"repo = '{ast_metrics_ctx.repo}' AND commit = '{ast_metrics_ctx.commit}'"
        )
        == 0
    ):
        ensure_catalog_with_goids(ast_metrics_ctx, artifacts.catalog)
    return artifacts.ast_map


def _call_fan_counts(ast_lookup: dict[int, FunctionAst]) -> tuple[dict[int, int], dict[int, int]]:
    """Compute simple fan-out and fan-in counts using call names.

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        Fan-out counts and fan-in counts keyed by GOID.
    """
    name_by_goid = {goid: ast_node.qualname.split(".")[-1] for goid, ast_node in ast_lookup.items()}
    canonical_names = set(name_by_goid.values())
    fan_out: dict[int, set[int]] = {goid: set() for goid in ast_lookup}
    for goid, ast_node in ast_lookup.items():
        target_names: set[str] = set()
        for node in ast.walk(ast_node.node):
            if isinstance(node, ast.Call):
                func = node.func
                target_name = None
                if isinstance(func, ast.Name):
                    target_name = func.id
                elif isinstance(func, ast.Attribute):
                    target_name = func.attr
                if target_name is not None and target_name in canonical_names:
                    target_names.add(target_name)
        fan_out[goid] = {
            candidate for candidate, name in name_by_goid.items() if name in target_names
        }
    fan_in: dict[int, int] = {}
    for goid in ast_lookup:
        fan_in[goid] = 0
    for targets in fan_out.values():
        for target in targets:
            fan_in[target] += 1
    fan_out_counts = {goid: len(targets) for goid, targets in fan_out.items()}
    return fan_out_counts, fan_in


AST_EXPECTATIONS = [
    (GOID_FUNC_A, MOD_A_PATH, 3, 0, 1, 0),
    (GOID_FUNC_B, MOD_B_PATH, 6, 0, 1, 1),
    (GOID_FUNC_C, MOD_C_PATH, 2, 0, 0, 1),
    (GOID_HELPER, MOD_UTIL_PATH, 2, 0, 0, 0),
]


def _ast_metrics_from_seeds(seeds: list[AstMetricSeed]) -> list[tuple[str, float]]:
    """Build (rel_path, complexity) tuples from AST metric seeds.

    Returns
    -------
    list[tuple[str, float]]
        Relative path and complexity tuples for assertions.
    """
    return [(seed.rel_path, seed.complexity) for seed in seeds]


def _ast_metrics_from_gateway(gateway: StorageGateway) -> list[tuple[str, float]]:
    """Load (rel_path, complexity) tuples from core.ast_metrics.

    Returns
    -------
    list[tuple[str, float]]
        Relative path and complexity tuples from the database.
    """
    rows = gateway.con.execute(
        "SELECT rel_path, complexity FROM core.ast_metrics",
    ).fetchall()
    return [(str(rel_path), float(complexity or 0.0)) for rel_path, complexity in rows]


def test_file_churn_creation() -> None:
    """Create a FileChurn with default values."""
    churn = FileChurn()

    expect_equal(churn.commits, set())
    expect_equal(churn.authors, set())
    expect_equal(churn.lines_added, 0)
    expect_equal(churn.lines_deleted, 0)


def test_file_churn_accumulation() -> None:
    """Accumulate churn data."""
    churn = FileChurn()
    churn.commits.add("abc123")
    churn.commits.add("def456")
    churn.authors.add("Alice")
    churn.authors.add("Bob")
    churn.lines_added = EXPECTED_LINES_ADDED
    churn.lines_deleted = EXPECTED_LINES_DELETED

    summary = churn.to_summary()

    expect_equal(summary["commit_count"], EXPECTED_COMMIT_COUNT)
    expect_equal(summary["author_count"], EXPECTED_AUTHOR_COUNT)
    expect_equal(summary["lines_added"], EXPECTED_LINES_ADDED)
    expect_equal(summary["lines_deleted"], EXPECTED_LINES_DELETED)


def test_file_churn_empty_summary() -> None:
    """Empty churn produces zero counts."""
    churn = FileChurn()
    summary = churn.to_summary()

    expect_equal(summary["commit_count"], 0)
    expect_equal(summary["author_count"], 0)
    expect_equal(summary["lines_added"], 0)
    expect_equal(summary["lines_deleted"], 0)


def test_file_churn_duplicate_commits() -> None:
    """Duplicate commits are deduplicated."""
    churn = FileChurn()
    churn.commits.add("abc123")
    churn.commits.add("abc123")

    summary = churn.to_summary()

    expect_equal(summary["commit_count"], 1)


def test_file_churn_multiple_authors() -> None:
    """Multiple authors counted uniquely in summary."""
    churn = FileChurn()
    churn.authors.update({"alice", "bob", "charlie"})

    summary = churn.to_summary()

    expect_equal(summary["author_count"], EXPECTED_AUTHOR_COUNT_MULTI)


def test_file_churn_to_summary_keys() -> None:
    """Verify to_summary returns expected keys."""
    churn = FileChurn()
    summary = churn.to_summary()

    expect_in("commit_count", summary)
    expect_in("author_count", summary)
    expect_in("lines_added", summary)
    expect_in("lines_deleted", summary)
    expect_equal(len(summary), EXPECTED_SUMMARY_KEYS)


def test_compute_hotspots_empty_ast_metrics() -> None:
    """Empty AST metrics produce no hotspot rows."""
    rows = compute_hotspot_rows([], churn_stats={})
    expect_equal(len(rows), 0)


def test_compute_hotspots_with_ast_data() -> None:
    """Compute hotspots with AST metrics data."""
    ast_metrics = _ast_metrics_from_seeds([AstMetricSeed(rel_path="test_file.py", complexity=5.0)])
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(len(rows), 1)
    row = rows[0]
    expect_equal(row["rel_path"], "test_file.py")
    expect_true(_as_float(row["score"]) > HOTSPOT_SCORE_THRESHOLD)


def test_compute_hotspots_multiple_files() -> None:
    """Compute hotspots with multiple files."""
    ast_metrics = _ast_metrics_from_seeds(
        [
            AstMetricSeed(rel_path="file1.py", complexity=3.0),
            AstMetricSeed(rel_path="file2.py", complexity=7.0),
            AstMetricSeed(rel_path="file3.py", complexity=12.0),
        ],
    )
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(len(rows), EXPECTED_FILE_COUNT_MULTI)


def test_compute_hotspots_score_calculation() -> None:
    """Verify hotspot score calculation components."""
    ast_metrics = _ast_metrics_from_seeds(
        [AstMetricSeed(rel_path="scored.py", complexity=EXPECTED_COMPLEXITY)]
    )
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(len(rows), 1)
    row = rows[0]
    expect_equal(_as_float(row["complexity"]), EXPECTED_COMPLEXITY)
    expect_equal(row["commit_count"], 0)
    expect_equal(row["author_count"], 0)
    expect_true(_as_float(row["score"]) > HOTSPOT_SCORE_THRESHOLD)


def test_compute_hotspots_high_complexity() -> None:
    """Compute hotspots handles high complexity values."""
    high_complexity = 100.0
    ast_metrics = _ast_metrics_from_seeds(
        [AstMetricSeed(rel_path="high_complexity.py", complexity=high_complexity)]
    )
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(len(rows), 1)
    row = rows[0]
    expect_equal(_as_float(row["complexity"]), high_complexity)
    expect_true(_as_float(row["score"]) > HOTSPOT_SCORE_THRESHOLD)


def test_compute_hotspots_idempotent() -> None:
    """Compute hotspots is deterministic across repeated runs."""
    ast_metrics = _ast_metrics_from_seeds([AstMetricSeed(rel_path="idempotent.py", complexity=5.0)])
    first = compute_hotspot_rows(ast_metrics, churn_stats={})
    second = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(first, second)


@pytest.mark.parametrize(
    ("lines_added", "lines_deleted"),
    [
        (0, 0),
        (100, 0),
        (0, 100),
        (50, 50),
        (1000, 500),
    ],
)
def test_file_churn_line_counts(lines_added: int, lines_deleted: int) -> None:
    """Test various line count combinations."""
    churn = FileChurn()
    churn.lines_added = lines_added
    churn.lines_deleted = lines_deleted

    summary = churn.to_summary()

    expect_equal(summary["lines_added"], lines_added)
    expect_equal(summary["lines_deleted"], lines_deleted)


def test_compute_hotspots_windows_path_handling() -> None:
    """Compute hotspots normalizes Windows-style paths."""
    ast_metrics = _ast_metrics_from_seeds(
        [AstMetricSeed(rel_path="path\\to\\file.py", complexity=5.0)]
    )
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    expect_equal(rows[0]["rel_path"], "path/to/file.py")


def test_compute_hotspots_zero_complexity() -> None:
    """Compute hotspots handles zero complexity."""
    ast_metrics = _ast_metrics_from_seeds(
        [AstMetricSeed(rel_path="zero_complexity.py", complexity=0.0)]
    )
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    row = rows[0]
    expect_equal(_as_float(row["complexity"]), 0.0)
    expect_true(_as_float(row["score"]) >= HOTSPOT_SCORE_THRESHOLD)


def test_compute_hotspots_negative_complexity() -> None:
    """Compute hotspots clamps negative complexity to zero."""
    ast_metrics = _ast_metrics_from_seeds([AstMetricSeed(rel_path="negative.py", complexity=-5.0)])
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})
    row = rows[0]
    expect_equal(_as_float(row["complexity"]), 0.0)
    expect_true(_as_float(row["score"]) >= HOTSPOT_SCORE_THRESHOLD)


def test_compute_hotspots_from_seeded_ast_metrics(ast_metrics_ctx: TestContext) -> None:
    """Compute hotspots using seeded AST metrics pack data."""
    ast_metrics = _ast_metrics_from_gateway(ast_metrics_ctx.gateway)
    rows = compute_hotspot_rows(ast_metrics, churn_stats={})

    row = next((r for r in rows if r["rel_path"] == MOD_A_PATH), None)
    expect_is_not_none(row)
    if row is None:
        pytest.fail("Expected hotspot row for seeded AST metrics")

    expect_equal(row["rel_path"], MOD_A_PATH)
    expect_equal(row["commit_count"], 0)
    expect_equal(row["author_count"], 0)
    expect_equal(_as_float(row["complexity"]), MEDIUM_COMPLEXITY)
    expect_true(_as_float(row["score"]) > HOTSPOT_SCORE_THRESHOLD)


def test_collect_git_file_stats_logs_git_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Git failures warn but return empty churn stats."""

    class _FailingRunner(ToolRunner):
        def __init__(self) -> None:
            super().__init__(cache_dir=Path.cwd())
            self.invocations = 0

        def run(
            self,
            tool: ToolName | str,
            args: Sequence[str],
            *,
            options: ToolRunOptions | None = None,
        ) -> ToolRunResult:
            _ = options
            self.invocations += 1
            resolved_tool = tool if isinstance(tool, ToolName) else ToolName(tool)
            return ToolRunResult(
                tool=resolved_tool,
                args=tuple(args),
                returncode=2,
                stdout="ok",
                stderr="fatal: not a git repo",
                duration_s=0.0,
            )

    snapshot = make_snapshot(repo_root=tmp_path)
    caplog.set_level("WARNING", logger="codeintel.build.hamilton.native.analytics.hotspots")

    stats = collect_git_file_stats(
        snapshot.repo_root,
        max_commits=5,
        runner=_FailingRunner(),
    )
    rows = compute_hotspot_rows(
        _ast_metrics_from_seeds([AstMetricSeed(rel_path="sample.py", complexity=2.0)]),
        stats,
    )
    expect_equal(rows[0]["commit_count"], 0)
    expect_equal(rows[0]["author_count"], 0)
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="git log exited with code 2",
    )


@pytest.mark.parametrize(
    "expected",
    AST_EXPECTATIONS,
)
def test_ast_metrics_loc_and_calls(
    ast_lookup: dict[int, FunctionAst],
    expected: tuple[int, str, int, int, int, int],
) -> None:
    """Validate AST spans, decorator counts, and simple call fan metrics."""
    goid, rel_path, expected_loc, decorators, fan_out, fan_in = expected
    fan_out_counts, fan_in_counts = _call_fan_counts(ast_lookup)
    node = ast_lookup[goid]
    expect_equal(node.rel_path, rel_path)
    expect_equal(node.end_line - node.start_line + 1, expected_loc)
    expect_equal(len(node.node.decorator_list), decorators)
    expect_equal(fan_out_counts[goid], fan_out)
    expect_equal(fan_in_counts[goid], fan_in)


@pytest.mark.parametrize(
    ("goid", "rel_path", "expected_start", "expected_end"),
    [
        (GOID_FUNC_A, MOD_A_PATH, FUNC_A_LINES[0], FUNC_A_LINES[1]),
        (GOID_FUNC_B, MOD_B_PATH, FUNC_B_LINES[0], FUNC_B_LINES[1]),
        (GOID_FUNC_C, MOD_C_PATH, FUNC_C_LINES[0], FUNC_C_LINES[1]),
        (GOID_HELPER, MOD_UTIL_PATH, HELPER_LINES[0], HELPER_LINES[1]),
    ],
)
def test_ast_lookup_matches_canonical_functions(
    ast_lookup: dict[int, FunctionAst],
    goid: int,
    rel_path: str,
    expected_start: int,
    expected_end: int,
) -> None:
    """AST lookup should map GOIDs to canonical function spans."""
    expect_in(goid, ast_lookup)
    node = ast_lookup[goid]
    expect_equal(node.rel_path, rel_path)
    expect_equal(node.start_line, expected_start)
    expect_equal(node.end_line, expected_end)


def test_parse_git_log_empty_lines() -> None:
    """Empty input returns empty dict."""
    result = parse_git_log_lines([])
    expect_false(result)


def test_parse_git_log_simple_commit() -> None:
    """Parse a single commit with one file."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t5\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/module.py", result)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], PARSE_COMMITS_1)
    expect_equal(stats["author_count"], PARSE_AUTHORS_1)
    expect_equal(stats["lines_added"], PARSE_LINES_ADDED_10)
    expect_equal(stats["lines_deleted"], PARSE_LINES_DELETED_5)


def test_parse_git_log_multiple_commits_same_file() -> None:
    """Aggregate stats across multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tBob",
        "5\t3\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], PARSE_COMMITS_2)
    expect_equal(stats["author_count"], PARSE_AUTHORS_2)
    expect_equal(stats["lines_added"], PARSE_LINES_ADDED_15)
    expect_equal(stats["lines_deleted"], PARSE_LINES_DELETED_3)


def test_parse_git_log_multiple_files() -> None:
    """Parse commits touching multiple files."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/a.py",
        "20\t5\tsrc/b.py",
    ]
    result = parse_git_log_lines(lines)
    expect_length(result, PARSE_FILES_2)
    expect_in("src/a.py", result)
    expect_in("src/b.py", result)


def test_parse_git_log_empty_line_handling() -> None:
    """Skip empty lines in input."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "",
        "10\t0\tsrc/module.py",
        "",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/module.py", result)


def test_parse_git_log_binary_file_numstat() -> None:
    """Handle binary files (- in numstat)."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "-\t-\timage.png",
        "10\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_in("image.png", result)
    stats = result["image.png"]
    expect_equal(stats["lines_added"], PARSE_LINES_DELETED_0)


def test_parse_git_log_normalize_path_separators() -> None:
    """Backslashes are normalized to forward slashes."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc\\windows\\module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/windows/module.py", result)


def test_parse_git_log_skip_lines_before_commit() -> None:
    """Skip numstat lines before any COMMIT header."""
    lines = [
        "10\t0\torphan.py",
        "COMMIT\tabc123\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_not_in("orphan.py", result)
    expect_in("src/module.py", result)


def test_parse_git_log_malformed_numstat_line() -> None:
    """Skip malformed numstat lines."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "not\ta\tvalid\tnumstat",
        "10\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_length(result, PARSE_COMMITS_1)


def test_parse_git_log_same_author_multiple_commits() -> None:
    """Same author counted once even with multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], PARSE_COMMITS_2)
    expect_equal(stats["author_count"], PARSE_AUTHORS_1)
