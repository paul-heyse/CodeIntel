"""Entrypoint analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import polars as pl
from hamilton.function_modifiers import cache

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.entrypoints.compute import (
    ENTRYPOINT_TESTS_COLS,
    ENTRYPOINTS_COLS,
    EntrypointsResult,
    compute_entrypoints_pure,
)
from codeintel.build.analytics.entrypoints.core import (
    EntrypointBuildInputs,
    EntrypointContextInputs,
)
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
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
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

ENTRYPOINTS_TARGET_NAME = "entrypoints"
ENTRYPOINTS_TABLE_KEY = "analytics.entrypoints"
ENTRYPOINT_TESTS_TABLE_KEY = "analytics.entrypoint_tests"
ENTRYPOINTS_CONTRACT = TableContractSpec(
    table_key=ENTRYPOINTS_TABLE_KEY,
    domain="analytics",
    target=ENTRYPOINTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="entrypoints__base",
)
ENTRYPOINT_TESTS_CONTRACT = TableContractSpec(
    table_key=ENTRYPOINT_TESTS_TABLE_KEY,
    domain="analytics",
    target=ENTRYPOINTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="entrypoint_tests__base",
)


@dataclass(frozen=True)
class EntrypointModuleFrames:
    """Frame inputs for entrypoint module context."""

    modules_frame: pl.DataFrame
    goids_frame: pl.DataFrame
    features_frame: pl.DataFrame


@dataclass(frozen=True)
class EntrypointTestFrames:
    """Frame inputs for entrypoint test context."""

    test_catalog_frame: pl.DataFrame


@dataclass(frozen=True)
class EntrypointSubsystemFrames:
    """Frame inputs for entrypoint subsystem context."""

    subsystems_frame: pl.DataFrame
    subsystem_modules_frame: pl.DataFrame


def _parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _module_map(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_map: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def _features_by_goid(features_frame: pl.DataFrame) -> dict[int, FunctionAstFeatures]:
    features_map: dict[int, FunctionAstFeatures] = {}
    for row in features_frame.iter_rows(named=True):
        goid_raw = row.get("function_goid_h128")
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        features_map[int(goid_value)] = FunctionAstFeatures(
            goid=int(goid_value),
            rel_path=rel_path,
            qualname=qualname,
            is_async=bool(row.get("is_async")),
            decorators=tuple(_parse_json_list(row.get("decorators"))),
            imports={},
            libraries_used=frozenset(_parse_json_list(row.get("libraries_used"))),
            io_flags=IoFlags(
                uses_network=bool(row.get("uses_network")),
                uses_db=bool(row.get("uses_db")),
                uses_filesystem=bool(row.get("uses_filesystem")),
                uses_subprocess=bool(row.get("uses_subprocess")),
            ),
            uses_concurrency_lib=bool(row.get("uses_concurrency_lib")),
            uses_threading=bool(row.get("uses_threading")),
            uses_asyncio_lib=bool(row.get("uses_asyncio_lib")),
            http_client_libs=frozenset(_parse_json_list(row.get("http_client_libs"))),
            http_server_libs=frozenset(_parse_json_list(row.get("http_server_libs"))),
            db_libs=frozenset(_parse_json_list(row.get("db_libs"))),
            message_libs=frozenset(_parse_json_list(row.get("message_libs"))),
            config_read_count=int(row.get("config_read_count") or 0),
            feature_flag_count=int(row.get("feature_flag_count") or 0),
            extra={},
        )
    return features_map


def entrypoint_module_frames(
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__analytics__function_ast_features: InferableTabularInput,
) -> EntrypointModuleFrames:
    """Collect module-related frames for entrypoint detection.

    Returns
    -------
    EntrypointModuleFrames
        Frame bundle with modules, goids, and AST features.
    """
    return EntrypointModuleFrames(
        modules_frame=tabular_to_lazyframe(q__core__modules).collect(),
        goids_frame=tabular_to_lazyframe(q__core__goids).collect(),
        features_frame=tabular_to_lazyframe(q__analytics__function_ast_features).collect(),
    )


def entrypoint_test_frames(
    q__analytics__test_catalog: InferableTabularInput,
) -> EntrypointTestFrames:
    """Collect test-related frames for entrypoint detection.

    Returns
    -------
    EntrypointTestFrames
        Frame bundle with the test catalog snapshot.
    """
    return EntrypointTestFrames(
        test_catalog_frame=tabular_to_lazyframe(q__analytics__test_catalog).collect(),
    )


def entrypoint_subsystem_frames(
    q__analytics__subsystems: InferableTabularInput,
    q__analytics__subsystem_modules: InferableTabularInput,
) -> EntrypointSubsystemFrames:
    """Collect subsystem-related frames for entrypoint detection.

    Returns
    -------
    EntrypointSubsystemFrames
        Frame bundle with subsystem metadata and module mapping.
    """
    return EntrypointSubsystemFrames(
        subsystems_frame=tabular_to_lazyframe(q__analytics__subsystems).collect(),
        subsystem_modules_frame=tabular_to_lazyframe(q__analytics__subsystem_modules).collect(),
    )


@cache()
def entrypoints_result(
    env: BuildEnv,
    entrypoint_module_frames: EntrypointModuleFrames,
    entrypoint_test_frames: EntrypointTestFrames,
    entrypoint_subsystem_frames: EntrypointSubsystemFrames,
) -> EntrypointsResult:
    """Compute entrypoint rows using module and AST feature inputs.

    Returns
    -------
    EntrypointsResult
        Entrypoints result containing entrypoint and test rows.
    """
    module_map = _module_map(entrypoint_module_frames.modules_frame)
    if not module_map:
        return EntrypointsResult(entrypoint_rows=(), test_rows=())
    catalog = catalog_provider_from_frames(
        goids_frame=entrypoint_module_frames.goids_frame,
        modules_frame=entrypoint_module_frames.modules_frame,
    )
    inputs = EntrypointBuildInputs(
        catalog_provider=catalog,
        module_map=module_map,
        features_map=_features_by_goid(entrypoint_module_frames.features_frame),
    )
    context_inputs = EntrypointContextInputs(
        modules_frame=entrypoint_module_frames.modules_frame,
        test_catalog_frame=entrypoint_test_frames.test_catalog_frame,
        subsystem_modules_frame=entrypoint_subsystem_frames.subsystem_modules_frame,
        subsystems_frame=entrypoint_subsystem_frames.subsystems_frame,
    )
    return compute_entrypoints_pure(
        env.snapshot,
        inputs,
        context_inputs,
    )


def entrypoints__base(entrypoints_result: EntrypointsResult) -> pl.LazyFrame:
    """Build entrypoint rows from computed entrypoints metadata.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing entrypoint rows.
    """
    return rows_to_frame(
        ENTRYPOINTS_TABLE_KEY,
        entrypoints_result.entrypoint_rows,
        columns=ENTRYPOINTS_COLS,
    )


def entrypoint_tests__base(entrypoints_result: EntrypointsResult) -> pl.LazyFrame:
    """Build entrypoint test rows from computed entrypoints metadata.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing entrypoint test rows.
    """
    if not entrypoints_result.test_rows:
        return empty_frame_for_table(ENTRYPOINT_TESTS_TABLE_KEY)
    return rows_to_frame(
        ENTRYPOINT_TESTS_TABLE_KEY,
        entrypoints_result.test_rows,
        columns=ENTRYPOINT_TESTS_COLS,
    )


_MODULE = sys.modules[__name__]
_ENTRYPOINTS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=ENTRYPOINTS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=ENTRYPOINTS_TABLE_KEY,
            base_node="entrypoints__base",
            contract=ENTRYPOINTS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=ENTRYPOINTS_TABLE_KEY),
            node_name="entrypoints__table",
        ),
        TableTargetTableSpec(
            table_key=ENTRYPOINT_TESTS_TABLE_KEY,
            base_node="entrypoint_tests__base",
            contract=ENTRYPOINT_TESTS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=ENTRYPOINT_TESTS_TABLE_KEY),
            node_name="entrypoint_tests__table",
        ),
    ),
    table_materializations_node="entrypoints__table_materializations",
    anchor_node_name="t__entrypoints",
)
attach_table_target_template(_MODULE, spec=_ENTRYPOINTS_TABLE_TARGET_SPEC)
entrypoints__table = _MODULE.entrypoints__table
entrypoint_tests__table = _MODULE.entrypoint_tests__table
entrypoints__table_materializations = _MODULE.entrypoints__table_materializations
t__entrypoints = _MODULE.t__entrypoints


__all__ = [
    "entrypoint_tests__base",
    "entrypoint_tests__table",
    "entrypoints__base",
    "entrypoints__table",
    "entrypoints__table_materializations",
    "t__entrypoints",
]
