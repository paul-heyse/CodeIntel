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
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.nodes.module_attach import attach_node
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

TEST_GRAPH_TARGET_NAME = "test_graph_metrics"
TEST_GRAPH_TESTS_TABLE_KEY = "analytics.test_graph_metrics_tests"
TEST_GRAPH_FUNCTIONS_TABLE_KEY = "analytics.test_graph_metrics_functions"
TEST_GRAPH_COLLECT_GROUP = "test_graph_metrics_core"
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


_TEST_GRAPH_METRICS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=TEST_GRAPH_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=TEST_GRAPH_TESTS_TABLE_KEY,
            base_node="test_graph_metrics_tests__base",
            contract=TEST_GRAPH_TESTS_CONTRACT,
            save_spec=DatasetSaveSpec(
                table_key=TEST_GRAPH_TESTS_TABLE_KEY,
                collect_group=TEST_GRAPH_COLLECT_GROUP,
            ),
            node_name="test_graph_metrics_tests__table",
        ),
        TableTargetTableSpec(
            table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY,
            base_node="test_graph_metrics_functions__base",
            contract=TEST_GRAPH_FUNCTIONS_CONTRACT,
            save_spec=DatasetSaveSpec(
                table_key=TEST_GRAPH_FUNCTIONS_TABLE_KEY,
                collect_group=TEST_GRAPH_COLLECT_GROUP,
            ),
            node_name="test_graph_metrics_functions__table",
        ),
    ),
    table_materializations_node="test_graph_metrics__table_materializations",
    anchor_node_name="t__test_graph_metrics",
)
attach_table_target_template(_MODULE, spec=_TEST_GRAPH_METRICS_TABLE_TARGET_SPEC)
test_graph_metrics_tests__table = _MODULE.test_graph_metrics_tests__table
test_graph_metrics_functions__table = _MODULE.test_graph_metrics_functions__table
test_graph_metrics__table_materializations = _MODULE.test_graph_metrics__table_materializations
t__test_graph_metrics = _MODULE.t__test_graph_metrics


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


_TEST_PROFILE_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=TEST_PROFILE_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=TEST_PROFILE_TABLE_KEY,
            base_node="test_profile__base",
            contract=TEST_PROFILE_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=TEST_PROFILE_TABLE_KEY),
            node_name="test_profile__table",
        ),
    ),
    table_materializations_node="test_profile__table_materializations",
    anchor_node_name="t__test_profile",
)
attach_table_target_template(_MODULE, spec=_TEST_PROFILE_TABLE_TARGET_SPEC)
test_profile__table = _MODULE.test_profile__table
test_profile__table_materializations = _MODULE.test_profile__table_materializations
t__test_profile = _MODULE.t__test_profile


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
