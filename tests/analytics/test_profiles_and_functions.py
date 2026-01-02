"""Consolidated tests for profiles analytics and function analysis helpers."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
import pytest

from codeintel.build.analytics.compute.functions.complexity import (
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    ComplexityMetrics,
    compute_complexity,
)
from codeintel.build.analytics.compute.functions.loc import (
    LinesOfCode,
    compute_loc,
    count_logical_lines,
)
from codeintel.build.analytics.compute.functions.signatures import (
    FunctionSignature,
    ParameterInfo,
    extract_signature,
)
from codeintel.build.analytics.compute.functions.typedness import (
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)
from codeintel.build.analytics.compute.profiles.aggregation import (
    FunctionMetricInput,
    ProfileAggregates,
    aggregate_function_metrics,
    compute_profile_stats,
)
from codeintel.build.analytics.compute.profiles.features import (
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
from codeintel.build.analytics.profiles.files import (
    build_file_profile_rows,
    compute_file_profile_inputs,
)
from codeintel.build.analytics.profiles.functions import (
    SLOW_TEST_THRESHOLD_MS,
    FunctionProfileViews,
    build_function_profile_rows,
    compute_function_profile_inputs,
    join_function_contracts,
    join_function_docs,
    join_function_effects,
    join_function_risk,
    join_function_roles,
    load_function_base_info,
)
from codeintel.build.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.build.analytics.profiles.modules import (
    build_module_profile_rows,
    compute_module_profile_inputs,
)
from codeintel.build.analytics.profiles.types import (
    FileProfileFrames,
    FunctionProfileFrames,
    ModuleProfileFrames,
)
from codeintel.build.analytics.profiles.utils import DEFAULT_MODULE_TABLE
from codeintel.build.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
    validate_contract_rows,
)
from codeintel.build.schemas import configure_schema_service
from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.fixtures.rows import (
    blank_file_profile_row,
    blank_function_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
    list_public_exports,
    sample_file_profile_rows,
    sample_function_profile_rows,
    sample_module_profile_rows,
    sample_test_profile_rows,
)
from tests._helpers.scenarios import TestScenario

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.build.analytics.profiles.types import FunctionProfileInputs
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFileProfileRow as FileProfileRowModel,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionProfileRow as FunctionProfileRowModel,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsModuleProfileRow as ModuleProfileRowModel,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsTestProfileRow as ProfileRowModel,
    )
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection
    from tests._helpers.context import TestContext


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def _columns_by_table() -> dict[str, tuple[str, ...]]:
    columns = load_columns_by_table()
    return {key: tuple(value) for key, value in columns.items()}


def function_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a function profile row using schema-derived column order.

    Parameters
    ----------
    row
        Row mapping keyed by column name.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.function_profile"]
    return serialize_row(row, columns)


def file_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a file profile row using schema-derived column order.

    Parameters
    ----------
    row
        Row mapping keyed by column name.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.file_profile"]
    return serialize_row(row, columns)


def module_profile_row_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a module profile row using schema-derived column order.

    Parameters
    ----------
    row
        Row mapping keyed by column name.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.module_profile"]
    return serialize_row(row, columns)


def serialize_test_profile_row(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a test profile row using schema-derived column order.

    Parameters
    ----------
    row
        Row mapping keyed by column name.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.test_profile"]
    return serialize_row(row, columns)


def _build_function_profile_views(
    inputs: FunctionProfileInputs,
    module_table: str,
) -> FunctionProfileViews:
    return FunctionProfileViews(
        base_by_func=load_function_base_info(inputs, module_table=module_table),
        risk_by_func=join_function_risk(inputs),
        graph_by_func=summarize_graph_for_function_profile(inputs),
        effects_by_func=join_function_effects(inputs),
        contracts_by_func=join_function_contracts(inputs),
        roles_by_func=join_function_roles(inputs),
        docs_by_func=join_function_docs(inputs),
    )


def _frame_from_table(gateway: StorageGateway, table_key: str) -> pl.DataFrame:
    try:
        relation = gateway.relation_from_table_key(table_key)
    except duckdb.Error:
        return pl.DataFrame()
    frame = pl.from_arrow(relation.arrow())
    return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame()


def _function_profile_frames(gateway: StorageGateway) -> FunctionProfileFrames:
    return FunctionProfileFrames(
        function_metrics=_frame_from_table(gateway, "analytics.function_metrics"),
        function_types=_frame_from_table(gateway, "analytics.function_types"),
        modules=_frame_from_table(gateway, DEFAULT_MODULE_TABLE),
        typedness=_frame_from_table(gateway, "analytics.typedness"),
        diagnostics=_frame_from_table(gateway, "analytics.static_diagnostics"),
        goid_risk_factors=_frame_from_table(gateway, "analytics.goid_risk_factors"),
        graph_metrics_functions=_frame_from_table(gateway, "analytics.graph_metrics_functions"),
        function_effects=_frame_from_table(gateway, "analytics.function_effects"),
        function_contracts=_frame_from_table(gateway, "analytics.function_contracts"),
        semantic_roles_functions=_frame_from_table(gateway, "analytics.semantic_roles_functions"),
        docstrings=_frame_from_table(gateway, "analytics.docstrings"),
        hotspots=_frame_from_table(gateway, "analytics.hotspots"),
        call_graph_edges=_frame_from_table(gateway, "graph.call_graph_edges"),
        call_graph_nodes=_frame_from_table(gateway, "graph.call_graph_nodes"),
    )


def _file_profile_frames(gateway: StorageGateway) -> FileProfileFrames:
    return FileProfileFrames(
        function_profile=_frame_from_table(gateway, FUNCTION_PROFILE_TABLE_KEY),
        ast_metrics=_frame_from_table(gateway, "analytics.ast_metrics"),
        hotspots=_frame_from_table(gateway, "analytics.hotspots"),
        typedness=_frame_from_table(gateway, "analytics.typedness"),
        static_diagnostics=_frame_from_table(gateway, "analytics.static_diagnostics"),
        modules=_frame_from_table(gateway, DEFAULT_MODULE_TABLE),
    )


def _module_profile_frames(gateway: StorageGateway) -> ModuleProfileFrames:
    return ModuleProfileFrames(
        modules=_frame_from_table(gateway, DEFAULT_MODULE_TABLE),
        function_profile=_frame_from_table(gateway, FUNCTION_PROFILE_TABLE_KEY),
        file_profile=_frame_from_table(gateway, FILE_PROFILE_TABLE_KEY),
        import_graph_edges=_frame_from_table(gateway, "graph.import_graph_edges"),
        semantic_roles_modules=_frame_from_table(gateway, "analytics.semantic_roles_modules"),
    )


def _write_snapshot_rows(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    snapshot: SnapshotRef,
) -> None:
    if not rows:
        return
    gateway.policy.delete_for_snapshot(table_key, repo=snapshot.repo, commit=snapshot.commit)
    gateway.policy.bulk_insert_mappings(table_key, rows)


def _populate_profile_tables(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    module_table = DEFAULT_MODULE_TABLE
    function_inputs = compute_function_profile_inputs(
        snapshot,
        _function_profile_frames(gateway),
    )
    views = _build_function_profile_views(function_inputs, module_table)
    function_rows = list(build_function_profile_rows(function_inputs, views=views))
    if not function_rows:
        pytest.fail("function_profile rows missing")
    _write_snapshot_rows(gateway, FUNCTION_PROFILE_TABLE_KEY, function_rows, snapshot)

    file_inputs = compute_file_profile_inputs(snapshot, _file_profile_frames(gateway))
    file_rows_iter = build_file_profile_rows(file_inputs, module_table=module_table)
    if file_rows_iter is None:
        pytest.fail("file_profile rows missing")
    file_rows = list(file_rows_iter)
    _write_snapshot_rows(gateway, FILE_PROFILE_TABLE_KEY, file_rows, snapshot)

    module_inputs = compute_module_profile_inputs(snapshot, _module_profile_frames(gateway))
    module_rows_iter = build_module_profile_rows(module_inputs, module_table=module_table)
    if module_rows_iter is None:
        pytest.fail("module_profile rows missing")
    module_rows = list(module_rows_iter)
    _write_snapshot_rows(gateway, MODULE_PROFILE_TABLE_KEY, module_rows, snapshot)


EPSILON = 1e-6
REL_PATH = "pkg/mod.py"
MODULE = "pkg.mod"
FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
FILE_PROFILE_TABLE_KEY = "analytics.file_profile"
MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
RowBuilder = Callable[[str, str], Sequence[Mapping[str, object]]]
WriterFn = Callable[[Any, Sequence[Mapping[str, object]]], int]


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
        SELECT high_risk_function_count, module
        FROM analytics.file_profile
        WHERE rel_path = ?
        """,
        [REL_PATH],
    ).fetchone()
    if row is None:
        pytest.fail("file_profile row missing")
    expect_equal(row[0], 1)
    expect_equal(row[1], MODULE)


def _assert_module_profile(con: DuckDBConnection) -> None:
    row = con.execute(
        """
        SELECT import_fan_in, import_fan_out, in_cycle
        FROM analytics.module_profile
        WHERE module = ?
        """,
        [MODULE],
    ).fetchone()
    if row is None:
        pytest.fail("module_profile row missing")
    expect_equal(row[0], 1)
    expect_equal(row[1], 1)
    expect_true(row[2] is True)


def test_profile_builders_aggregate_expected_fields(profiles_ctx: TestContext) -> None:
    """Ensure profile builders compose metrics, tests, and graph data."""
    gateway = profiles_ctx.gateway
    con = gateway.con
    snapshot = profiles_ctx.to_snapshot_ref()
    _populate_profile_tables(gateway, snapshot)
    _assert_function_profile(con)
    _assert_file_profile(con)
    _assert_module_profile(con)


# =============================================================================
# Profile contract tuple alignment tests
# =============================================================================


def _function_rows(repo: str, commit: str) -> list[FunctionProfileRowModel]:
    rows: list[FunctionProfileRowModel] = []
    for base in sample_function_profile_rows(repo, commit):
        rel_path = base.get("rel_path") or ""
        module_name = (base.get("module") or str(rel_path)).replace("/", ".").removesuffix(".py")
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
        rel_path = base.get("rel_path") or ""
        module_name = (base.get("module") or str(rel_path)).replace("/", ".").removesuffix(".py")
        row = blank_test_profile_row()
        row.update(base)
        row.setdefault("module", module_name)
        row.setdefault("qualname", base.get("qualname", base["test_id"].split("::")[-1]))
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def test_profile_tuple_alignment() -> None:
    """Serializer alignment with column constants."""
    columns = _columns_by_table()
    fn_row = _function_rows("r", "c")[0]
    file_row = _file_rows("r", "c")[0]
    mod_row = _module_rows("r", "c")[1]
    tst_row = _test_rows("r", "c")[0]

    if len(function_profile_row_to_tuple(fn_row)) != len(columns["analytics.function_profile"]):
        pytest.fail("Function profile tuple length mismatch with column constants.")
    if len(file_profile_row_to_tuple(file_row)) != len(columns["analytics.file_profile"]):
        pytest.fail("File profile tuple length mismatch with column constants.")
    if len(module_profile_row_to_tuple(mod_row)) != len(columns["analytics.module_profile"]):
        pytest.fail("Module profile tuple length mismatch with column constants.")
    if len(serialize_test_profile_row(tst_row)) != len(columns["analytics.test_profile"]):
        pytest.fail("Test profile tuple length mismatch with column constants.")


def test_test_profile_insertion(tmp_path: Path) -> None:
    """Insert test profile rows via dataset contracts."""
    ctx = TestScenario.minimal().build(tmp_path)
    try:
        test_rows = _test_rows(ctx.repo, ctx.commit)
        ctx.gateway.con.execute("DELETE FROM analytics.test_profile")
        test_contract = get_analytics_dataset_contract(ctx.gateway, "analytics.test_profile")
        validated_tests = validate_contract_rows(
            test_contract.table_key, test_rows, gateway=ctx.gateway
        )
        inserted = insert_analytics_rows(ctx.gateway, test_contract, validated_tests)
        expect_equal(inserted, len(test_rows))
        stored = ctx.gateway.con.execute(
            "SELECT test_id, rel_path FROM analytics.test_profile ORDER BY test_id"
        ).fetchall()
        expect_equal(len(stored), len(test_rows))

        ctx.gateway.con.execute("DELETE FROM analytics.test_profile")
        validated_subset = validate_contract_rows(
            test_contract.table_key, test_rows[:1], gateway=ctx.gateway
        )
        inserted_subset = insert_analytics_rows(ctx.gateway, test_contract, validated_subset)
        expect_equal(inserted_subset, 1)
        remaining = ctx.gateway.con.execute(
            "SELECT COUNT(*) FROM analytics.test_profile"
        ).fetchone()
        if remaining is None:
            pytest.fail("test_profile rows missing after rewrite")
        expect_equal(int(remaining[0]), 1)

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
