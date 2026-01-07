"""Entrypoint analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pyarrow as pa
from hamilton.function_modifiers import cache

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.entrypoints.compute import (
    EntrypointsResult,
    compute_entrypoints_pure,
)
from codeintel.build.analytics.entrypoints.core import (
    EntrypointBuildInputs,
    EntrypointContextInputs,
)
from codeintel.build.analytics.entrypoints.runtime import load_entrypoint_module_sources
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_int
from codeintel.core.serialization.json import decode_json_list
from codeintel.core.serialization.payload import decode_payload

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

    modules_frame: pa.Table
    goids_frame: pa.Table
    features_frame: pa.Table


@dataclass(frozen=True)
class EntrypointTestFrames:
    """Frame inputs for entrypoint test context."""

    test_catalog_frame: pa.Table


@dataclass(frozen=True)
class EntrypointSubsystemFrames:
    """Frame inputs for entrypoint subsystem context."""

    subsystems_frame: pa.Table
    subsystem_modules_frame: pa.Table


def _parse_json_list(value: object) -> list[str]:
    decoded = decode_payload(value)
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    if isinstance(decoded, str):
        return [str(item) for item in decode_json_list(decoded)]
    return []


def _module_map(modules_frame: pa.Table) -> dict[str, str]:
    module_map: dict[str, str] = {}
    if modules_frame.num_rows == 0:
        return module_map
    if not {"path", "module"}.issubset(set(modules_frame.column_names)):
        return module_map
    for row in iter_rows(modules_frame, ["path", "module"]):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def _features_by_goid(features_frame: pa.Table) -> dict[int, FunctionAstFeatures]:
    features_map: dict[int, FunctionAstFeatures] = {}
    if features_frame.num_rows == 0:
        return features_map
    required = {"function_goid_h128", "rel_path", "qualname"}
    if not required.issubset(set(features_frame.column_names)):
        return features_map
    columns = [
        "function_goid_h128",
        "rel_path",
        "qualname",
        "is_async",
        "decorators",
        "libraries_used",
        "uses_network",
        "uses_db",
        "uses_filesystem",
        "uses_subprocess",
        "uses_concurrency_lib",
        "uses_threading",
        "uses_asyncio_lib",
        "http_client_libs",
        "http_server_libs",
        "db_libs",
        "message_libs",
        "config_read_count",
        "feature_flag_count",
    ]
    for row in iter_rows(features_frame, columns):
        feature = _feature_from_row(row)
        if feature is None:
            continue
        features_map[feature.goid] = feature
    return features_map


def _feature_from_row(row: dict[str, object]) -> FunctionAstFeatures | None:
    goid_value = normalize_decimal_id(row.get("function_goid_h128"))
    rel_path = row.get("rel_path")
    qualname = row.get("qualname")
    if goid_value is None or not isinstance(rel_path, str) or not isinstance(qualname, str):
        return None
    return FunctionAstFeatures(
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
        config_read_count=coerce_optional_int(
            row.get("config_read_count"),
            ctx="entrypoints.config_read_count",
        )
        or 0,
        feature_flag_count=coerce_optional_int(
            row.get("feature_flag_count"),
            ctx="entrypoints.feature_flag_count",
        )
        or 0,
        extra={},
    )


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
        modules_frame=tabular_to_arrow_table(q__core__modules),
        goids_frame=tabular_to_arrow_table(q__core__goids),
        features_frame=tabular_to_arrow_table(q__analytics__function_ast_features),
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
        test_catalog_frame=tabular_to_arrow_table(q__analytics__test_catalog),
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
        subsystems_frame=tabular_to_arrow_table(q__analytics__subsystems),
        subsystem_modules_frame=tabular_to_arrow_table(q__analytics__subsystem_modules),
    )


@cache(behavior="default")
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
    module_sources = load_entrypoint_module_sources(
        module_map,
        env.snapshot.repo_root,
        scan_profile=inputs.scan_profile,
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
        module_sources,
    )


def entrypoints__base(entrypoints_result: EntrypointsResult) -> pa.Table:
    """Build entrypoint rows from computed entrypoints metadata.

    Returns
    -------
    pa.Table
        Reader containing entrypoint rows.
    """
    if not entrypoints_result.entrypoint_rows:
        return empty_table_for_table(ENTRYPOINTS_TABLE_KEY)
    reader, _ = table_for_rows(
        ENTRYPOINTS_TABLE_KEY,
        entrypoints_result.entrypoint_rows,
    )
    return reader


def entrypoint_tests__base(entrypoints_result: EntrypointsResult) -> pa.Table:
    """Build entrypoint test rows from computed entrypoints metadata.

    Returns
    -------
    pa.Table
        Reader containing entrypoint test rows.
    """
    if not entrypoints_result.test_rows:
        return empty_table_for_table(ENTRYPOINT_TESTS_TABLE_KEY)
    reader, _ = table_for_rows(
        ENTRYPOINT_TESTS_TABLE_KEY,
        entrypoints_result.test_rows,
    )
    return reader


_MODULE = sys.modules[__name__]
_ENTRYPOINTS_TABLE_TARGET_SPEC = build_multi_table_target_spec(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=ENTRYPOINTS_TARGET_NAME,
        tables=(
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=ENTRYPOINTS_TABLE_KEY,
                    base_node="entrypoints__base",
                    contract=ENTRYPOINTS_CONTRACT,
                    node_name="entrypoints__table",
                    input_type=pa.Table,
                ),
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=ENTRYPOINT_TESTS_TABLE_KEY,
                    base_node="entrypoint_tests__base",
                    contract=ENTRYPOINT_TESTS_CONTRACT,
                    node_name="entrypoint_tests__table",
                    input_type=pa.Table,
                ),
            ),
        ),
        table_materializations_node="entrypoints__table_materializations",
        anchor_node_name="t__entrypoints",
    )
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
