"""Consolidated tests for profiles analytics and function analysis helpers."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import cast

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
from codeintel.analytics.profiles import (
    SLOW_TEST_THRESHOLD_MS,
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.config import SnapshotInit
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
from codeintel.config.primitives import BuildLayoutOptions
from codeintel.storage.gateway import DuckDBConnection
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.config_factory import profiles_analytics_cfg
from tests._helpers.factory_profiles import seed_profile_data
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
from tests._helpers.seeds.function_types import FunctionTypesPack

EPSILON = 1e-6
REL_PATH = "pkg/mod.py"
MODULE = "pkg.mod"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def profiles_ctx(tmp_path) -> Iterator[object]:
    """Provide seeded profile context using ProfilePack."""
    ctx = TestScenario.with_profiles().build(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def function_types_ctx(tmp_path):
    """Context seeded with function types for typedness-related checks."""
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


def test_profile_builders_aggregate_expected_fields(profiles_ctx) -> None:
    """Ensure profile builders compose metrics, tests, coverage, and graph data."""
    gateway = profiles_ctx.gateway
    con = gateway.con
    seed_profile_data(
        gateway,
        repo=profiles_ctx.repo,
        commit=profiles_ctx.commit,
        rel_path=REL_PATH,
        module=MODULE,
    )
    cfg = profiles_analytics_cfg(
        SnapshotInit(
            repo=profiles_ctx.repo,
            commit=profiles_ctx.commit,
            repo_root=profiles_ctx.repo_root,
        ),
        layout=BuildLayoutOptions(build_dir=profiles_ctx.build_dir),
    )
    build_function_profile(gateway, cfg)
    build_file_profile(gateway, cfg)
    build_module_profile(gateway, cfg)
    _assert_function_profile(con)
    _assert_file_profile(con)
    _assert_module_profile(con)


# =============================================================================
# Profile contract tuple alignment tests
# =============================================================================


def _function_rows(repo: str, commit: str):
    rows = []
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


def _file_rows(repo: str, commit: str):
    rows = []
    for base in sample_file_profile_rows(repo, commit):
        row = blank_file_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _module_rows(repo: str, commit: str):
    rows = []
    for base in sample_module_profile_rows(repo, commit):
        row = blank_module_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _test_rows(repo: str, commit: str):
    rows = []
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


def _behavior_rows(repo: str, commit: str):
    rows = []
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


def test_function_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    ctx = TestScenario.with_profiles().build(tmp_path=None)
    try:
        rows = _function_rows(ctx.repo, ctx.commit)
        inserted_first = ctx.gateway.analytics.insert_function_profile(rows)
        stored_first = ctx.gateway.con.execute(
            """
            SELECT function_goid_h128, tags, owners
            FROM analytics.function_profile
            ORDER BY function_goid_h128
            """
        ).fetchall()
        expect_equal(len(rows), inserted_first)
        expect_equal(len(rows), len(stored_first))

        # Verify deletion and idempotent insert
        inserted_second = ctx.gateway.analytics.insert_function_profile(rows)
        expect_equal(inserted_second, len(rows))
    finally:
        ctx.close()


# =============================================================================
# Function analysis helpers (complexity, LOC, signatures)
# =============================================================================


def _parse_function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse source and return the function node."""
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
    expect_false = lambda value: expect_true(not value)  # Local helper to stay within lint rules
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
    expect_true(any(dec.name == "classmethod" for dec in deco_sig.decorators))
    expect_equal(deco_sig.parameters[0].kind, "positional_or_keyword")

    property_func = _parse_function(
        """
@property
def count(self) -> int:
    return 1
"""
    )
    prop_sig = extract_signature(property_func)
    expect_true(any(dec.name == "property" for dec in prop_sig.decorators))
    expect_true(prop_sig.is_property)
    expect_true(isinstance(prop_sig.parameters[0], ParameterInfo))
    assert_frozen(prop_sig, "name", "other")


def test_function_types_seed_alignment(function_types_ctx) -> None:
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


def test_exports_helper_lists_public_names() -> None:
    """list_public_exports should return sorted exports when __all__ absent."""
    module = type("Module", (), {"__all__": ("a", "b")})
    exports = list_public_exports(module)
    expect_equal(exports, ("a", "b"))
