"""Testing analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.testing.behavioral.tags import build_behavior_rows
from codeintel.analytics.testing.compute import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.analytics.testing.coverage.edges import build_test_coverage_edges_rows
from codeintel.analytics.testing.profiles.builder import build_test_profile_result
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

TEST_COVERAGE_TARGET_NAME = "test_coverage_edges"
TEST_COVERAGE_TABLE_KEY = "analytics.test_coverage_edges"
TEST_COVERAGE_SAVE_CONTEXT = SaverContext(domain="analytics", target=TEST_COVERAGE_TARGET_NAME)
TEST_COVERAGE_CONTRACT = TableContractSpec(
    table_key=TEST_COVERAGE_TABLE_KEY,
    domain="analytics",
    target=TEST_COVERAGE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="test_coverage_edges__base",
)

TEST_GRAPH_TARGET_NAME = "test_graph_metrics"
TEST_GRAPH_TESTS_TABLE_KEY = "analytics.test_graph_metrics_tests"
TEST_GRAPH_FUNCTIONS_TABLE_KEY = "analytics.test_graph_metrics_functions"
TEST_GRAPH_TABLE_KEYS = (TEST_GRAPH_TESTS_TABLE_KEY, TEST_GRAPH_FUNCTIONS_TABLE_KEY)
TEST_GRAPH_SAVE_CONTEXT = SaverContext(domain="analytics", target=TEST_GRAPH_TARGET_NAME)
TEST_GRAPH_TESTS_CONTRACT = TableContractSpec(
    table_key=TEST_GRAPH_TESTS_TABLE_KEY,
    domain="analytics",
    target=TEST_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="test_graph_metrics_tests__base",
)
TEST_GRAPH_FUNCTIONS_CONTRACT = TableContractSpec(
    table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY,
    domain="analytics",
    target=TEST_GRAPH_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="test_graph_metrics_functions__base",
)

TEST_PROFILE_TARGET_NAME = "test_profile"
TEST_PROFILE_TABLE_KEY = "analytics.test_profile"
TEST_PROFILE_SAVE_CONTEXT = SaverContext(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
TEST_PROFILE_CONTRACT = TableContractSpec(
    table_key=TEST_PROFILE_TABLE_KEY,
    domain="analytics",
    target=TEST_PROFILE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="test_profile__base",
)

BEHAVIORAL_COVERAGE_TARGET_NAME = "behavioral_coverage"
BEHAVIORAL_COVERAGE_TABLE_KEY = "analytics.behavioral_coverage"
BEHAVIORAL_COVERAGE_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=BEHAVIORAL_COVERAGE_TARGET_NAME,
)
BEHAVIORAL_COVERAGE_CONTRACT = TableContractSpec(
    table_key=BEHAVIORAL_COVERAGE_TABLE_KEY,
    domain="analytics",
    target=BEHAVIORAL_COVERAGE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="behavioral_coverage__base",
)


def test_coverage_edges__base(
    env: BuildEnv,
    _q__analytics__coverage_lines: InferableTabularInput,
    _q__analytics__test_catalog: InferableTabularInput,
    _q__core__goids: InferableTabularInput,
) -> pl.LazyFrame:
    """Build test coverage edges rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing test coverage edges rows.
    """
    rows = build_test_coverage_edges_rows(env.gateway, env.snapshot)
    return rows_to_frame(TEST_COVERAGE_TABLE_KEY, rows)


@save_dataset(
    context=TEST_COVERAGE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_COVERAGE_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=TEST_COVERAGE_TARGET_NAME,
    table_key=TEST_COVERAGE_TABLE_KEY,
)
@table_contract(TEST_COVERAGE_CONTRACT)
def test_coverage_edges__table(test_coverage_edges__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist test coverage edges.

    Returns
    -------
    pl.LazyFrame
        Persisted test coverage edges frame.
    """
    return test_coverage_edges__base


@codeintel_target(domain="analytics", target=TEST_COVERAGE_TARGET_NAME)
def t__test_coverage_edges(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__test_coverage_edges: MaterializationResult,
) -> TargetRunRecord:
    """Finalize test_coverage_edges target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the test_coverage_edges target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=TEST_COVERAGE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            TEST_COVERAGE_TABLE_KEY: m__analytics__test_coverage_edges,
        },
    )


def test_graph_metrics_result(
    env: BuildEnv,
    _q__analytics__test_coverage_edges: InferableTabularInput,
    _q__analytics__goid_risk_factors: InferableTabularInput,
) -> TestGraphMetricsResult:
    """Compute test graph metrics rows.

    Returns
    -------
    TestGraphMetricsResult
        Computed test graph metrics result.
    """
    return compute_test_graph_metrics_pure(env.gateway, env.snapshot)


def test_graph_metrics_tests__base(
    test_graph_metrics_result: TestGraphMetricsResult,
) -> pl.LazyFrame:
    """Build test graph metrics rows for tests.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing test graph metrics rows for tests.
    """
    return rows_to_frame(
        TEST_GRAPH_TESTS_TABLE_KEY,
        test_graph_metrics_result.test_rows,
        columns=TEST_GRAPH_METRICS_TESTS_COLS,
    )


@save_dataset(
    context=TEST_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_GRAPH_TESTS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=TEST_GRAPH_TARGET_NAME,
    table_key=TEST_GRAPH_TESTS_TABLE_KEY,
)
@table_contract(TEST_GRAPH_TESTS_CONTRACT)
def test_graph_metrics_tests__table(
    test_graph_metrics_tests__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist test graph metrics rows for tests.

    Returns
    -------
    pl.LazyFrame
        Persisted test graph metrics frame for tests.
    """
    return test_graph_metrics_tests__base


def test_graph_metrics_functions__base(
    test_graph_metrics_result: TestGraphMetricsResult,
) -> pl.LazyFrame:
    """Build test graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing test graph metrics rows for functions.
    """
    return rows_to_frame(
        TEST_GRAPH_FUNCTIONS_TABLE_KEY,
        test_graph_metrics_result.function_rows,
        columns=TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    )


@save_dataset(
    context=TEST_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=TEST_GRAPH_TARGET_NAME,
    table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY,
)
@table_contract(TEST_GRAPH_FUNCTIONS_CONTRACT)
def test_graph_metrics_functions__table(
    test_graph_metrics_functions__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist test graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Persisted test graph metrics frame for functions.
    """
    return test_graph_metrics_functions__base


test_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=TEST_GRAPH_TARGET_NAME,
    table_keys=TEST_GRAPH_TABLE_KEYS,
    node_name="test_graph_metrics__table_materializations",
)


@codeintel_target(domain="analytics", target=TEST_GRAPH_TARGET_NAME)
def t__test_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    test_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize test_graph_metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the test_graph_metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=TEST_GRAPH_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=test_graph_metrics__table_materializations,
    )


def test_profile__base(
    env: BuildEnv,
    _q__analytics__test_catalog: InferableTabularInput,
    _q__analytics__test_coverage_edges: InferableTabularInput,
    _q__analytics__test_graph_metrics_tests: InferableTabularInput,
    _q__analytics__subsystem_coverage_cache: InferableTabularInput,
) -> pl.LazyFrame:
    """Build test profile rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing test profile rows.
    """
    result = build_test_profile_result(env.gateway, env.snapshot)
    if result.rows is None:
        return empty_frame_for_table(TEST_PROFILE_TABLE_KEY)
    return rows_to_frame(TEST_PROFILE_TABLE_KEY, result.rows)


@save_dataset(
    context=TEST_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
)
@table_contract(TEST_PROFILE_CONTRACT)
def test_profile__table(test_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist test profile rows.

    Returns
    -------
    pl.LazyFrame
        Persisted test profile frame.
    """
    return test_profile__base


@codeintel_target(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
def t__test_profile(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__test_profile: MaterializationResult,
) -> TargetRunRecord:
    """Finalize test_profile target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the test_profile target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=TEST_PROFILE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            TEST_PROFILE_TABLE_KEY: m__analytics__test_profile,
        },
    )


def behavioral_coverage__base(
    env: BuildEnv,
    _q__analytics__test_profile: InferableTabularInput,
) -> pl.LazyFrame:
    """Build behavioral coverage rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing behavioral coverage rows.
    """
    rows = build_behavior_rows(env.gateway, env.snapshot)
    return rows_to_frame(BEHAVIORAL_COVERAGE_TABLE_KEY, rows)


@save_dataset(
    context=BEHAVIORAL_COVERAGE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=BEHAVIORAL_COVERAGE_TABLE_KEY),
)
@table_contract(BEHAVIORAL_COVERAGE_CONTRACT)
def behavioral_coverage__table(behavioral_coverage__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist behavioral coverage rows.

    Returns
    -------
    pl.LazyFrame
        Persisted behavioral coverage frame.
    """
    return behavioral_coverage__base


@codeintel_target(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
def t__behavioral_coverage(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__behavioral_coverage: MaterializationResult,
) -> TargetRunRecord:
    """Finalize behavioral_coverage target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the behavioral_coverage target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=BEHAVIORAL_COVERAGE_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            BEHAVIORAL_COVERAGE_TABLE_KEY: m__analytics__behavioral_coverage,
        },
    )


__all__ = [
    "behavioral_coverage__base",
    "behavioral_coverage__table",
    "t__behavioral_coverage",
    "t__test_coverage_edges",
    "t__test_graph_metrics",
    "t__test_profile",
    "test_coverage_edges__base",
    "test_coverage_edges__table",
    "test_graph_metrics__table_materializations",
    "test_graph_metrics_functions__base",
    "test_graph_metrics_functions__table",
    "test_graph_metrics_tests__base",
    "test_graph_metrics_tests__table",
    "test_profile__base",
    "test_profile__table",
]
