"""Testing analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import polars as pl

from codeintel.build.analytics.testing.compute import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.build.analytics.testing.profiles.builder import (
    TestProfileFrameInputs,
    build_test_profile_result,
)
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
from codeintel.build.hamilton.nodes.module_attach import attach_node
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

TEST_GRAPH_TARGET_NAME = "test_graph_metrics"
TEST_GRAPH_TESTS_TABLE_KEY = "analytics.test_graph_metrics_tests"
TEST_GRAPH_FUNCTIONS_TABLE_KEY = "analytics.test_graph_metrics_functions"
TEST_GRAPH_TABLE_KEYS = (TEST_GRAPH_TESTS_TABLE_KEY, TEST_GRAPH_FUNCTIONS_TABLE_KEY)
TEST_GRAPH_COLLECT_GROUP = "test_graph_metrics_core"
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

_MODULE = sys.modules[__name__]


def _graph_metrics_result(
    env: BuildEnv,
) -> TestGraphMetricsResult:
    """Compute test graph metrics rows.

    Returns
    -------
    TestGraphMetricsResult
        Computed test graph metrics result.
    """
    return compute_test_graph_metrics_pure(
        env.snapshot,
        goid_risk_factors_frame=None,
    )


attach_node(_MODULE, node_name="test_graph_metrics_result", fn=_graph_metrics_result)
test_graph_metrics_result = _MODULE.test_graph_metrics_result
del _graph_metrics_result


def _graph_metrics_tests__base(
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


attach_node(_MODULE, node_name="test_graph_metrics_tests__base", fn=_graph_metrics_tests__base)
test_graph_metrics_tests__base = _MODULE.test_graph_metrics_tests__base
del _graph_metrics_tests__base


@save_dataset(
    context=TEST_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=TEST_GRAPH_TESTS_TABLE_KEY,
        collect_group=TEST_GRAPH_COLLECT_GROUP,
    ),
)
@table_contract(TEST_GRAPH_TESTS_CONTRACT)
def _graph_metrics_tests__table(
    test_graph_metrics_tests__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist test graph metrics rows for tests.

    Returns
    -------
    pl.LazyFrame
        Persisted test graph metrics frame for tests.
    """
    return test_graph_metrics_tests__base


attach_node(_MODULE, node_name="test_graph_metrics_tests__table", fn=_graph_metrics_tests__table)
test_graph_metrics_tests__table = _MODULE.test_graph_metrics_tests__table
del _graph_metrics_tests__table


def _graph_metrics_functions__base(
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


attach_node(
    _MODULE,
    node_name="test_graph_metrics_functions__base",
    fn=_graph_metrics_functions__base,
)
test_graph_metrics_functions__base = _MODULE.test_graph_metrics_functions__base
del _graph_metrics_functions__base


@save_dataset(
    context=TEST_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY,
        collect_group=TEST_GRAPH_COLLECT_GROUP,
    ),
)
@table_contract(TEST_GRAPH_FUNCTIONS_CONTRACT)
def _graph_metrics_functions__table(
    test_graph_metrics_functions__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist test graph metrics rows for functions.

    Returns
    -------
    pl.LazyFrame
        Persisted test graph metrics frame for functions.
    """
    return test_graph_metrics_functions__base


attach_node(
    _MODULE,
    node_name="test_graph_metrics_functions__table",
    fn=_graph_metrics_functions__table,
)
test_graph_metrics_functions__table = _MODULE.test_graph_metrics_functions__table
del _graph_metrics_functions__table


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


@dataclass(frozen=True)
class TestProfileCoreFrames:
    test_catalog_frame: pl.DataFrame
    goids_frame: pl.DataFrame
    modules_frame: pl.DataFrame


@dataclass(frozen=True)
class TestProfileSubsystemFrames:
    subsystem_modules_frame: pl.DataFrame
    subsystems_frame: pl.DataFrame


@dataclass(frozen=True)
class TestProfileGraphFrames:
    test_graph_metrics_frame: pl.DataFrame


def test_profile_core_frames(
    q__analytics__test_catalog: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> TestProfileCoreFrames:
    return TestProfileCoreFrames(
        test_catalog_frame=tabular_to_lazyframe(q__analytics__test_catalog).collect(),
        goids_frame=tabular_to_lazyframe(q__core__goids).collect(),
        modules_frame=tabular_to_lazyframe(q__core__modules).collect(),
    )


def test_profile_subsystem_frames(
    q__analytics__subsystem_modules: InferableTabularInput,
    q__analytics__subsystems: InferableTabularInput,
) -> TestProfileSubsystemFrames:
    return TestProfileSubsystemFrames(
        subsystem_modules_frame=tabular_to_lazyframe(q__analytics__subsystem_modules).collect(),
        subsystems_frame=tabular_to_lazyframe(q__analytics__subsystems).collect(),
    )


def test_profile_graph_frames(
    q__analytics__test_graph_metrics_tests: InferableTabularInput,
) -> TestProfileGraphFrames:
    return TestProfileGraphFrames(
        test_graph_metrics_frame=tabular_to_lazyframe(
            q__analytics__test_graph_metrics_tests
        ).collect(),
    )


def test_profile_inputs(
    test_profile_core_frames: TestProfileCoreFrames,
    test_profile_subsystem_frames: TestProfileSubsystemFrames,
    test_profile_graph_frames: TestProfileGraphFrames,
) -> TestProfileFrameInputs:
    return TestProfileFrameInputs(
        test_catalog_frame=test_profile_core_frames.test_catalog_frame,
        goids_frame=test_profile_core_frames.goids_frame,
        modules_frame=test_profile_core_frames.modules_frame,
        subsystem_modules_frame=test_profile_subsystem_frames.subsystem_modules_frame,
        subsystems_frame=test_profile_subsystem_frames.subsystems_frame,
        test_graph_metrics_frame=test_profile_graph_frames.test_graph_metrics_frame,
    )


def _profile__base(
    env: BuildEnv,
    test_profile_inputs: TestProfileFrameInputs,
) -> pl.LazyFrame:
    """Build test profile rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing test profile rows.
    """
    result = build_test_profile_result(
        env.snapshot,
        test_profile_inputs,
    )
    if result.rows is None:
        return empty_frame_for_table(TEST_PROFILE_TABLE_KEY)
    return rows_to_frame(TEST_PROFILE_TABLE_KEY, result.rows)


attach_node(_MODULE, node_name="test_profile__base", fn=_profile__base)
test_profile__base = _MODULE.test_profile__base
del _profile__base


@save_dataset(
    context=TEST_PROFILE_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
)
@table_contract(TEST_PROFILE_CONTRACT)
def _profile__table(test_profile__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist test profile rows.

    Returns
    -------
    pl.LazyFrame
        Persisted test profile frame.
    """
    return test_profile__base


attach_node(_MODULE, node_name="test_profile__table", fn=_profile__table)
test_profile__table = _MODULE.test_profile__table
del _profile__table


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


__all__ = [
    "t__test_graph_metrics",
    "t__test_profile",
    "test_graph_metrics__table_materializations",
    "test_graph_metrics_functions__base",
    "test_graph_metrics_functions__table",
    "test_graph_metrics_tests__base",
    "test_graph_metrics_tests__table",
    "test_profile__base",
    "test_profile__table",
]
