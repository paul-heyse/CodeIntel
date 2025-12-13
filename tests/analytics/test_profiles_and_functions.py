"""Consolidated tests for profiles analytics and function analysis helpers."""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from codeintel.analytics.compute.functions.complexity import (
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    ComplexityMetrics,
    compute_complexity,
)
from codeintel.analytics.compute.functions.loc import LinesOfCode, compute_loc, count_logical_lines
from codeintel.analytics.compute.functions.signatures import (
    FunctionSignature,
    ParameterInfo,
    extract_signature,
)
from codeintel.analytics.compute.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)
from codeintel.analytics.compute.profiles.aggregation import (
    FunctionMetricInput,
    ProfileAggregates,
    aggregate_function_metrics,
    compute_profile_stats,
)
from codeintel.analytics.compute.profiles.features import (
    COMPLEXITY_NORMALIZATION,
    COMPLEXITY_WEIGHT,
    HIGH_COMPLEXITY_THRESHOLD,
    HIGH_TYPED_RATIO,
    LARGE_MODULE_THRESHOLD,
    LOW_COMPLEXITY_THRESHOLD,
    LOW_TYPED_RATIO,
    SIZE_WEIGHT,
    SMALL_MODULE_THRESHOLD,
    TYPEDNESS_WEIGHT,
    ProfileFeatures,
    extract_profile_features,
)
from codeintel.analytics.profiles import (
    SLOW_TEST_THRESHOLD_MS,
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.analytics.profiles import files as profile_files
from codeintel.analytics.profiles import functions as profile_functions
from codeintel.analytics.profiles import modules as profile_modules
from codeintel.analytics.testing.profiles import rows as profile_rows
from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    behavioral_coverage_row_to_tuple,
    file_profile_row_to_tuple,
    function_profile_row_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_profile_row,
)
from tests._helpers import METRICS_PACK, assert_frozen
from tests._helpers.assertions import (
    assert_coverage_lines,
    assert_function_loc,
    assert_typedness_bucket,
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
    require_row,
)
from tests._helpers.factories.row_factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_function_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
    sample_behavioral_coverage_rows,
    sample_file_profile_rows,
    sample_function_profile_rows,
    sample_module_profile_rows,
    sample_test_profile_rows,
)
from tests._helpers.rows import list_public_exports
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds.core import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.datasets import (
        BehavioralCoverageRowModel,
        FileProfileRowModel,
        FunctionProfileRowModel,
        ModuleProfileRowModel,
        ProfileRowModel,
    )
    from codeintel.storage.gateway import DuckDBConnection
    from tests._helpers.context import TestContext

EPSILON = 1e-6
REL_PATH = "pkg/mod.py"
MODULE = "pkg.mod"
RowBuilder = Callable[[str, str], list[dict[str, object]]]
WriterFn = Callable[[Any, list[dict[str, object]]], int]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def profiles_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide seeded profile context using ProfilePack.

    Yields
    ------
    Iterator[TestContext]
        Seeded context configured with profile seeds.
    """
    ctx = TestScenario.with_profiles().build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def coverage_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide coverage context using coverage/metrics packs.

    Yields
    ------
    Iterator[TestContext]
        Context seeded with coverage and metrics packs.
    """
    ctx = TestScenario.with_coverage().with_seeds(METRICS_PACK).build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def function_types_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Context seeded with function types for typedness-related checks.

    Yields
    ------
    Iterator[TestContext]
        Context prepared with function types seeds.
    """
    ctx = TestScenario.with_function_types().build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


# =============================================================================
# Profile builder tests
# =============================================================================


def _assert_function_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT tests_touching, failing_tests, slow_tests, call_fan_in, call_fan_out,
               risk_component_coverage, doc_short, slow_test_threshold_ms
        FROM analytics.function_profile
        WHERE function_goid_h128 = 1
        """
    ).fetchone()
    if row is None:
        pytest.fail("function_profile row missing")
    expect_equal(row[0], 1)
    expect_equal(row[1], 1)
    expect_equal(row[2], 1)
    expect_equal(row[3], 1)
    expect_equal(row[4], 1)
    expect_true(abs(row[5] - 0.2) < EPSILON)
    expect_equal(row[6], "Short doc")
    expect_equal(row[7], SLOW_TEST_THRESHOLD_MS)


def _assert_file_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT file_coverage_ratio, high_risk_function_count, module
        FROM analytics.file_profile
        WHERE rel_path = ?
        """,
        [REL_PATH],
    ).fetchone()
    if row is None:
        pytest.fail("file_profile row missing")
    expect_true(abs(row[0] - 0.5) < EPSILON)
    expect_equal(row[1], 1)
    expect_equal(row[2], MODULE)


def _assert_module_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT module_coverage_ratio, import_fan_in, import_fan_out, in_cycle
        FROM analytics.module_profile
        WHERE module = ?
        """,
        [MODULE],
    ).fetchone()
    if row is None:
        pytest.fail("module_profile row missing")
    expect_true(abs(row[0] - 1.0) < EPSILON)
    expect_equal(row[1], 1)
    expect_equal(row[2], 1)
    expect_true(row[3] is True)


def test_profile_builders_aggregate_expected_fields(profiles_ctx: TestContext) -> None:
    """Ensure profile builders compose metrics, tests, coverage, and graph data."""
    gateway = profiles_ctx.gateway
    con = gateway.con
    snapshot = profiles_ctx.to_snapshot_ref()
    build_function_profile(gateway, snapshot)
    build_file_profile(gateway, snapshot)
    build_module_profile(gateway, snapshot)
    _assert_function_profile(con)
    _assert_file_profile(con)
    _assert_module_profile(con)


def test_coverage_aggregates_and_function_metrics(coverage_ctx: TestContext) -> None:
    """Coverage aggregates share seeds with function metrics and typedness helpers."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()
    assert_coverage_lines(
        con,
        snapshot=snapshot,
        rel_path=MOD_A_PATH,
        executable=10,
        covered=8,
    )
    assert_coverage_lines(
        con,
        snapshot=snapshot,
        rel_path=MOD_B_PATH,
        executable=15,
        covered=12,
    )
    assert_coverage_lines(
        con,
        snapshot=snapshot,
        rel_path=MOD_C_PATH,
        executable=8,
        covered=6,
    )
    row = require_row(
        con.execute(
            """
            SELECT COUNT(*) FROM analytics.coverage_functions
            WHERE repo = ? AND commit = ?
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchone()
    )
    expect_equal(row[0], 4)
    assert_function_loc(con, goid=GOID_FUNC_A, loc=10, logical_loc=8)
    assert_function_loc(con, goid=GOID_FUNC_B, loc=15, logical_loc=12)
    assert_function_loc(con, goid=GOID_FUNC_C, loc=8, logical_loc=6)
    assert_typedness_bucket(con, goid=GOID_FUNC_A, bucket="fully_typed")
    assert_typedness_bucket(con, goid=GOID_FUNC_B, bucket="partial_typed")
    assert_typedness_bucket(con, goid=GOID_FUNC_C, bucket="untyped")


# =============================================================================
# Profile contract tuple alignment tests
# =============================================================================


def _function_rows(repo: str, commit: str) -> list[FunctionProfileRowModel]:
    rows: list[FunctionProfileRowModel] = []
    for base in sample_function_profile_rows(repo, commit):
        module_name = base.get("module") or base.get("rel_path", "").replace("/", ".").removesuffix(
            ".py"
        )
        row = blank_function_profile_row()
        row.update(base)
        row.setdefault("module", module_name)
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _file_rows(repo: str, commit: str) -> list[FileProfileRowModel]:
    rows: list[FileProfileRowModel] = []
    for base in sample_file_profile_rows(repo, commit):
        row = blank_file_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _module_rows(repo: str, commit: str) -> list[ModuleProfileRowModel]:
    rows: list[ModuleProfileRowModel] = []
    for base in sample_module_profile_rows(repo, commit):
        row = blank_module_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _test_rows(repo: str, commit: str) -> list[ProfileRowModel]:
    rows: list[ProfileRowModel] = []
    for base in sample_test_profile_rows(repo, commit):
        module_name = base.get("module") or base.get("rel_path", "").replace("/", ".").removesuffix(
            ".py"
        )
        row = blank_test_profile_row()
        row.update(base)
        row.setdefault("module", module_name)
        row.setdefault("qualname", base.get("qualname", base["test_id"].split("::")[-1]))
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _behavior_rows(repo: str, commit: str) -> list[BehavioralCoverageRowModel]:
    rows: list[BehavioralCoverageRowModel] = []
    for base in sample_behavioral_coverage_rows(repo, commit):
        row = blank_behavioral_coverage_row()
        row.update(base)
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def test_profile_tuple_alignment() -> None:
    """Serializer alignment with column constants."""
    fn_row = _function_rows("r", "c")[0]
    file_row = _file_rows("r", "c")[0]
    mod_row = _module_rows("r", "c")[1]
    tst_row = _test_rows("r", "c")[0]
    beh_row = _behavior_rows("r", "c")[0]

    if len(function_profile_row_to_tuple(fn_row)) != len(FUNCTION_PROFILE_COLUMNS):
        pytest.fail("Function profile tuple length mismatch with column constants.")
    if len(file_profile_row_to_tuple(file_row)) != len(FILE_PROFILE_COLUMNS):
        pytest.fail("File profile tuple length mismatch with column constants.")
    if len(module_profile_row_to_tuple(mod_row)) != len(MODULE_PROFILE_COLUMNS):
        pytest.fail("Module profile tuple length mismatch with column constants.")
    if len(serialize_test_profile_row(tst_row)) != len(TEST_PROFILE_COLUMNS):
        pytest.fail("Test profile tuple length mismatch with column constants.")
    if len(behavioral_coverage_row_to_tuple(beh_row)) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        pytest.fail("Behavioral coverage tuple length mismatch with column constants.")


@pytest.mark.parametrize(
    ("rows_builder", "writer", "select_sql", "delete_sql"),
    [
        (
            _function_rows,
            profile_functions.write_function_profile_rows,
            "SELECT function_goid_h128, tags, owners "
            "FROM analytics.function_profile ORDER BY function_goid_h128",
            "DELETE FROM analytics.function_profile",
        ),
        (
            _file_rows,
            profile_files.write_file_profile_rows,
            "SELECT rel_path, module, tags FROM analytics.file_profile ORDER BY rel_path",
            "DELETE FROM analytics.file_profile",
        ),
        (
            _module_rows,
            profile_modules.write_module_profile_rows,
            "SELECT module, path, tags FROM analytics.module_profile ORDER BY module",
            "DELETE FROM analytics.module_profile",
        ),
    ],
)
def test_profile_writers_replace_existing_rows(
    tmp_path: Path,
    rows_builder: RowBuilder,
    writer: WriterFn,
    select_sql: str,
    delete_sql: str,
) -> None:
    """Profile writers should overwrite existing rows for the same snapshot."""
    ctx = TestScenario.minimal().build(tmp_path)
    try:
        ctx.gateway.con.execute(delete_sql)
        rows = rows_builder(ctx.repo, ctx.commit)
        inserted_first = writer(ctx.gateway, rows)
        expect_equal(inserted_first, len(rows))
        stored_first = ctx.gateway.con.execute(select_sql).fetchall()
        expect_equal(len(stored_first), len(rows))

        inserted_second = writer(ctx.gateway, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = ctx.gateway.con.execute(select_sql).fetchall()
        expect_equal(len(stored_second), 1)
        expect_equal(stored_second[0][0], stored_first[0][0])
    finally:
        ctx.close()


def test_test_and_behavioral_profile_writers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test profile and behavioral coverage writers respect snapshot alignment."""
    ctx = TestScenario.minimal().build(tmp_path)
    try:
        snapshot = ctx.to_snapshot_ref()
        caplog.set_level(logging.WARNING)

        test_rows = _test_rows(ctx.repo, ctx.commit)
        ctx.gateway.con.execute("DELETE FROM analytics.test_profile")
        inserted = profile_rows.write_test_profile_rows(ctx.gateway, snapshot, test_rows)
        expect_equal(inserted, len(test_rows))
        stored = ctx.gateway.con.execute(
            "SELECT test_id, rel_path FROM analytics.test_profile ORDER BY test_id"
        ).fetchall()
        expect_equal(len(stored), len(test_rows))

        replaced = profile_rows.write_test_profile_rows(ctx.gateway, snapshot, test_rows[:1])
        expect_equal(replaced, 1)
        remaining = ctx.gateway.con.execute(
            "SELECT COUNT(*) FROM analytics.test_profile"
        ).fetchone()
        if remaining is None:
            pytest.fail("test_profile rows missing after rewrite")
        expect_equal(int(remaining[0]), 1)

        ctx.gateway.con.execute("DELETE FROM analytics.behavioral_coverage")
        behavior_rows = _behavior_rows(ctx.repo, ctx.commit)
        inserted_behavior = profile_rows.write_behavioral_coverage_rows(
            ctx.gateway, snapshot, behavior_rows
        )
        expect_equal(inserted_behavior, len(behavior_rows))
        behavior_count = ctx.gateway.con.execute(
            "SELECT COUNT(*) FROM analytics.behavioral_coverage"
        ).fetchone()
        if behavior_count is None:
            pytest.fail("behavioral_coverage rows missing after insert")
        expect_equal(int(behavior_count[0]), len(behavior_rows))
        warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
        expect_equal(warnings, [])
    finally:
        ctx.close()


# =============================================================================
# Function analysis helpers (complexity, LOC, signatures)
# =============================================================================


def _parse_function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse source and return the function node.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef
        Parsed function node from the provided source.

    Raises
    ------
    TypeError
        If the parsed body does not start with a function definition.
    """
    tree = ast.parse(source.strip())
    node = tree.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node
    msg = f"Expected function node, got {type(node)}"
    raise TypeError(msg)


def test_complexity_metrics_and_buckets() -> None:
    """Compute complexity metrics across simple and medium functions."""
    simple = _parse_function(
        """
def simple():
    return 1
"""
    )
    medium = _parse_function(
        """
def medium(a, b, c, d, e, f):
    if a:
        return 1
    if b:
        return 2
    if c:
        return 3
    if d:
        return 4
    if e:
        return 5
    return 0
"""
    )
    simple_metrics = compute_complexity(simple)
    medium_metrics = compute_complexity(medium)
    expect_is_instance(simple_metrics, ComplexityMetrics)
    expect_equal(simple_metrics.cyclomatic, 1)
    expect_equal(simple_metrics.complexity_bucket, "low")
    expect_true(medium_metrics.cyclomatic > COMPLEXITY_LOW)
    expect_true(medium_metrics.cyclomatic <= COMPLEXITY_MEDIUM)
    expect_equal(medium_metrics.complexity_bucket, "medium")
    assert_frozen(simple_metrics, "cyclomatic", 5)


def test_complexity_counts_and_attributes() -> None:
    """Verify control-flow counts and attribute detection."""
    func = _parse_function(
        """
def generator(items):
    for item in items:
        yield item
        yield item * 2
"""
    )
    metrics = compute_complexity(func)
    expect_equal(metrics.return_count, 0)
    expect_equal(metrics.yield_count, 2)
    expect_true(metrics.is_generator)
    expect_equal(metrics.decorator_count, 0)
    expect_false(metrics.has_docstring)


def test_loc_and_logical_line_counts() -> None:
    """Count physical, logical, blank, and comment lines."""
    lines = [
        "def func():",
        "    # comment line",
        "    value = 1",
        "    ",
        "    return value",
    ]
    loc = compute_loc(lines, start_line=1, end_line=len(lines))
    expected = LinesOfCode(physical=5, logical=3, blank=1, comment=1)
    expect_equal(loc, expected, label="loc counts")
    expect_equal(count_logical_lines(lines), 3, label="logical lines")


def test_param_stats_and_typedness_flags() -> None:
    """Extract parameter stats and typedness flags from a typed function."""
    node = _parse_function(
        """
def demo(self, a: int, b, *, flag: bool = False, **kwargs) -> str:
    return str(a + (b or 0))
"""
    )
    stats = compute_param_stats(node)
    expected_types = {"a": "int", "b": None, "flag": "bool", "kwargs": None}
    expect_is_instance(stats, ParamStats)
    expect_equal(stats.param_count, 5, label="param_count")
    expect_equal(stats.positional_params, 3, label="positional_params")
    expect_equal(stats.keyword_only_params, 1, label="keyword_only_params")
    expect_true(stats.has_varargs is False, message="has_varargs")
    expect_true(stats.has_varkw is True, message="has_varkw")
    expect_equal(stats.total_params, 4, label="total_params")
    expect_equal(stats.annotated_params, 2, label="annotated_params")
    expect_equal(stats.param_types, expected_types, label="param_types")
    expect_true(stats.has_return_annotation is True, message="return annotation")
    expect_equal(stats.return_type, "str", label="return_type")

    fully_typed = compute_typedness_flags(
        total_params=2, annotated_params=2, has_return_annotation=True
    )
    expect_equal(
        fully_typed,
        TypednessFlags(
            param_typed_ratio=1.0,
            unannotated_params=0,
            fully_typed=True,
            partial_typed=False,
            untyped=False,
            typedness_bucket="typed",
            typedness_source="annotations",
        ),
    )


def test_signature_extraction_variants() -> None:
    """Extract signatures for different decorator styles and param shapes."""
    typed_func = _parse_function(
        """
def add(a: int, b: int) -> int:
    return a + b
"""
    )
    signature = extract_signature(typed_func)
    expect_is_instance(signature, FunctionSignature)
    expect_length(signature.parameters, 2)
    expect_equal(signature.parameters[0].name, "a")
    expect_equal(signature.docstring, None)

    decorated = _parse_function(
        """
@classmethod
def from_dict(cls, data: dict) -> "MyClass":
    return cls()
"""
    )
    deco_sig = extract_signature(decorated)
    expect_true(any(str(dec) == "classmethod" for dec in deco_sig.decorators))
    expect_equal(deco_sig.parameters[0].kind, "positional_or_keyword")

    property_func = _parse_function(
        """
@property
def count(self) -> int:
    return 1
"""
    )
    prop_sig = extract_signature(property_func)
    expect_true(any(str(dec) == "property" for dec in prop_sig.decorators))
    expect_true(prop_sig.is_property)
    expect_true(isinstance(prop_sig.parameters[0], ParameterInfo))
    assert_frozen(prop_sig, "name", "other")


def test_function_types_seed_alignment(function_types_ctx: TestContext) -> None:
    """Function types seed should populate typedness buckets consistently."""
    rows = function_types_ctx.gateway.con.execute(
        """
        SELECT typedness_bucket, COUNT(*) FROM analytics.function_types
        WHERE repo = ? AND commit = ?
        GROUP BY typedness_bucket
        """,
        [function_types_ctx.repo, function_types_ctx.commit],
    ).fetchall()
    buckets = {row[0]: int(row[1]) for row in rows}
    expect_in("fully_typed", buckets)
    expect_in("partial_typed", buckets)
    expect_in("untyped", buckets)


def test_function_metric_inputs_and_aggregates_defaults() -> None:
    """Validate defaults and simple construction for profile aggregation inputs."""
    aggregates = ProfileAggregates()
    expect_equal(aggregates.total_functions, 0)
    expect_equal(aggregates.total_loc, 0)
    metric = FunctionMetricInput(
        loc=25,
        complexity=3,
        typedness_ratio=0.75,
        typedness_bucket="partial",
        complexity_bucket="low",
    )
    expect_equal(metric.loc, 25)
    expect_equal(metric.typedness_ratio, 0.75)


def test_aggregate_function_metrics_and_stats() -> None:
    """Aggregate typed, partial, and untyped functions then compute ratios."""
    metrics = [
        FunctionMetricInput(10, 2, 1.0, "typed", "low"),
        FunctionMetricInput(20, 5, 0.5, "partial", "medium"),
        FunctionMetricInput(30, 8, 0.0, "untyped", "high"),
    ]
    aggregates = aggregate_function_metrics(metrics)
    expect_equal(aggregates.total_functions, 3)
    expect_equal(aggregates.total_loc, 60)
    expect_equal(aggregates.typed_count, 1)
    expect_equal(aggregates.partial_typed_count, 1)
    expect_equal(aggregates.untyped_count, 1)
    expect_equal(aggregates.complexity_buckets.get("high"), 1)

    stats = compute_profile_stats(aggregates)
    expect_equal(stats["typed_ratio"], pytest.approx(1 / 3))
    expect_equal(stats["avg_loc"], pytest.approx(20.0))


def test_extract_profile_features_classifies_boundaries() -> None:
    """Classify size, complexity, and typedness at threshold boundaries."""
    small = ProfileAggregates(
        total_functions=5,
        total_loc=SMALL_MODULE_THRESHOLD - 1,
        avg_complexity=LOW_COMPLEXITY_THRESHOLD - 0.1,
        avg_typedness=HIGH_TYPED_RATIO,
        typed_count=5,
    )
    small_features = extract_profile_features(small)
    expect_equal(small_features.size_category, "small")
    expect_equal(small_features.complexity_category, "simple")
    expect_equal(small_features.typedness_category, "typed")

    large = ProfileAggregates(
        total_functions=20,
        total_loc=LARGE_MODULE_THRESHOLD,
        avg_complexity=HIGH_COMPLEXITY_THRESHOLD,
        avg_typedness=LOW_TYPED_RATIO - 0.01,
        untyped_count=15,
    )
    large_features = extract_profile_features(large)
    expect_equal(large_features.size_category, "large")
    expect_equal(large_features.complexity_category, "complex")
    expect_equal(large_features.typedness_category, "untyped")


def test_profile_quality_score_bounds_and_weights() -> None:
    """Quality score stays bounded and weights sum to expected total."""
    total_weight = TYPEDNESS_WEIGHT + COMPLEXITY_WEIGHT + SIZE_WEIGHT
    expect_true(abs(total_weight - 1.0) < EPSILON)

    high_complexity = ProfileAggregates(
        total_functions=5,
        avg_complexity=COMPLEXITY_NORMALIZATION * 2,
        avg_typedness=1.0,
    )
    features = extract_profile_features(high_complexity)
    expect_true(0.0 <= features.quality_score <= 1.0)
    frozen_features = ProfileFeatures(
        size_category="medium",
        complexity_category="moderate",
        typedness_category="partial",
        quality_score=0.5,
    )
    assert_frozen(frozen_features, "quality_score", 0.6)


def test_exports_helper_lists_public_names() -> None:
    """list_public_exports should return sorted exports when __all__ absent."""
    module = type("Module", (), {"__all__": ("a", "b")})
    exports = list_public_exports(module)
    expect_equal(exports, ("a", "b"))
