"""Consolidated ingestion targets for AST/CST/docstring extraction.

This module replaces the per-target files for:
- ``ast``: stdlib AST extraction
- ``cst``: LibCST extraction
- ``docstrings``: docstring extraction/parsing
- ``syntax_index``: LibCST parse manifest + syntax fact tables
- ``symtable``: CPython symtable scope/binding extraction
- ``bytecode``: CPython dis bytecode extraction
- ``inspect``: optional runtime inspect overlays

The targets share a common pattern:
1) Load module paths from the current snapshot
2) Convert paths into ``ModuleRecord``s
3) Execute pure ingestion compute-steps that return columnar rows
4) Materialize columnar rows via Hamilton materializers and emit ``TargetRunRecord``
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import (
    AstExtractOptions,
    BytecodeExtractOptions,
    InspectExtractOptions,
    SymtableExtractOptions,
    SyntaxIndexOptions,
)
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    TableOutputSpec,
    ToolRunContext,
    ToolTargetSpec,
    attach_tool_target_template,
    run_tool_step,
)
from codeintel.build.hamilton.native.patterns.tool_target import TabularByTable
from codeintel.build.hamilton.native.target_decorators import TargetSpecDescriptor
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.ingestion_normalize import (
    finalize_ingest_reader,
    finalize_ingest_table,
)
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_helpers import array_from_compute, safe_filter
from codeintel.build.tabular.compute_masks import equal_mask, or_kleene
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_table,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec, JoinType, Plan
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.execution.ids import RUN_PREFIX_INGEST, new_run_id
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.ast_extract import AstExtractStep
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.compute.dis_extract import DisExtractStep
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.compute.inspect_extract import InspectExtractStep
from codeintel.ingestion.compute.symtable_extract import SymtableExtractStep
from codeintel.ingestion.infrastructure.py_frontend import PyFrontend, PyFrontendOptions
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    ModuleRecord,
    PyFrontend,
)

AST_TARGET_NAME = "ast"
CST_TARGET_NAME = "cst"
DOCSTRINGS_TARGET_NAME = "docstrings"
SYNTAX_INDEX_TARGET_NAME = "syntax_index"
SYMTABLE_TARGET_NAME = "symtable"
BYTECODE_TARGET_NAME = "bytecode"
INSPECT_TARGET_NAME = "inspect"

AST_NODES_TABLE_KEY = "core.ast_nodes"
AST_METRICS_TABLE_KEY = "core.ast_metrics"
AST_TABLE_KEYS = (AST_NODES_TABLE_KEY, AST_METRICS_TABLE_KEY)

CST_NODES_TABLE_KEY = "core.cst_nodes"
CST_TABLE_KEYS = (CST_NODES_TABLE_KEY,)

DOCSTRINGS_TABLE_KEY = "core.docstrings"
DOCSTRINGS_TABLE_KEYS = (DOCSTRINGS_TABLE_KEY,)

PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
SYNTAX_SPANS_TABLE_KEY = "core.syntax_spans"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_EDGES_TABLE_KEY = "core.syntax_edges"
SYNTAX_SCOPES_TABLE_KEY = "core.syntax_scopes"
SYNTAX_DEFS_TABLE_KEY = "core.syntax_defs"
SYNTAX_REFS_TABLE_KEY = "core.syntax_refs"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_CALL_ARGS_TABLE_KEY = "core.syntax_call_args"
SYNTAX_FUNC_PARAMS_TABLE_KEY = "core.syntax_func_params"
SYNTAX_IMPORTS_TABLE_KEY = "core.syntax_imports"
SYNTAX_INDEX_TABLE_KEYS = (
    PARSE_MANIFEST_TABLE_KEY,
    SYNTAX_SPANS_TABLE_KEY,
    SYNTAX_NODES_TABLE_KEY,
    SYNTAX_EDGES_TABLE_KEY,
    SYNTAX_SCOPES_TABLE_KEY,
    SYNTAX_DEFS_TABLE_KEY,
    SYNTAX_REFS_TABLE_KEY,
    SYNTAX_CALLS_TABLE_KEY,
    SYNTAX_CALL_ARGS_TABLE_KEY,
    SYNTAX_FUNC_PARAMS_TABLE_KEY,
    SYNTAX_IMPORTS_TABLE_KEY,
)

PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_SYMBOLS_TABLE_KEY = "core.py_sym_symbols"
PY_SYM_SCOPE_EDGES_TABLE_KEY = "core.py_sym_scope_edges"
PY_SYM_NAMESPACE_EDGES_TABLE_KEY = "core.py_sym_namespace_edges"
PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY = "core.py_sym_function_partitions"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY = "core.py_sym_unresolved_bindings"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"
PY_SYM_TABLE_KEYS = (
    PY_SYM_SCOPES_TABLE_KEY,
    PY_SYM_SYMBOLS_TABLE_KEY,
    PY_SYM_SCOPE_EDGES_TABLE_KEY,
    PY_SYM_NAMESPACE_EDGES_TABLE_KEY,
    PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY,
    PY_SYM_BINDINGS_TABLE_KEY,
    PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY,
    PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
)

PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_EXCEPTION_TABLE_KEY = "core.py_bc_exception_table"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"
PY_BC_CFG_EDGES_TABLE_KEY = "core.py_bc_cfg_edges"
PY_BC_DEFUSE_EVENTS_TABLE_KEY = "core.py_bc_defuse_events"
PY_COMPILER_META_TABLE_KEY = "core.py_compiler_metadata"
PY_BC_TABLE_KEYS = (
    PY_BC_CODE_UNITS_TABLE_KEY,
    PY_BC_INSTRUCTIONS_TABLE_KEY,
    PY_BC_EXCEPTION_TABLE_KEY,
    PY_BC_BLOCKS_TABLE_KEY,
    PY_BC_CFG_EDGES_TABLE_KEY,
    PY_BC_DEFUSE_EVENTS_TABLE_KEY,
    PY_COMPILER_META_TABLE_KEY,
)

PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_MEMBERS_TABLE_KEY = "core.py_inspect_members_static"
PY_INSPECT_CLASS_MRO_TABLE_KEY = "core.py_inspect_class_mro"
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = "core.py_inspect_class_attrs"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_ANNOTATIONS_TABLE_KEY = "core.py_inspect_annotations_kv"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = "core.py_inspect_runtime_state"
PY_INSPECT_TABLE_KEYS = (
    PY_INSPECT_OBJECTS_TABLE_KEY,
    PY_INSPECT_MEMBERS_TABLE_KEY,
    PY_INSPECT_CLASS_MRO_TABLE_KEY,
    PY_INSPECT_CLASS_ATTRS_TABLE_KEY,
    PY_INSPECT_UNWRAP_TABLE_KEY,
    PY_INSPECT_SIGNATURES_TABLE_KEY,
    PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY,
    PY_INSPECT_ANNOTATIONS_TABLE_KEY,
    PY_INSPECT_SOURCE_TABLE_KEY,
    PY_INSPECT_RUNTIME_STATE_TABLE_KEY,
)


def py_frontend__options(env: BuildEnv) -> PyFrontendOptions:
    """Return shared frontend options for ingestion steps.

    Returns
    -------
    PyFrontendOptions
        Shared frontend configuration.
    """
    params = env.config.parameters_for("py_frontend")
    defaults = PyFrontendOptions()
    return PyFrontendOptions(
        max_entries=params.get_typed(
            "py_frontend_cache_entries",
            int,
            default=defaults.max_entries,
        ),
        cache_bytes=params.get_typed(
            "py_frontend_cache_bytes",
            bool,
            default=defaults.cache_bytes,
        ),
        cache_text=params.get_typed(
            "py_frontend_cache_text",
            bool,
            default=defaults.cache_text,
        ),
        cache_line_index=params.get_typed(
            "py_frontend_cache_line_index",
            bool,
            default=defaults.cache_line_index,
        ),
        cache_ast=params.get_typed(
            "py_frontend_cache_ast",
            bool,
            default=defaults.cache_ast,
        ),
        cache_code=params.get_typed(
            "py_frontend_cache_code",
            bool,
            default=defaults.cache_code,
        ),
        decode_errors=params.get_typed(
            "py_frontend_decode_errors",
            str,
            default=defaults.decode_errors,
        ),
    )


def py_frontend(
    env: BuildEnv,
    py_frontend__options: PyFrontendOptions,
) -> PyFrontend:
    """Create shared Python frontend for ingestion steps.

    Returns
    -------
    PyFrontend
        Shared frontend instance.
    """
    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    return PyFrontend(discovery=discovery, options=py_frontend__options)


@dataclass(frozen=True)
class DocstringsToolOutput(ToolStepOutput):
    """Tool step output for docstrings extraction."""

    rows: pa.Table = field(default_factory=lambda: empty_table_for_table(DOCSTRINGS_TABLE_KEY))
    row_count: int = 0


@dataclass(frozen=True)
class AstToolOutput(ToolStepOutput):
    """Tool step output for AST extraction."""

    ast_rows: pa.Table = field(default_factory=lambda: empty_table_for_table(AST_NODES_TABLE_KEY))
    metric_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(AST_METRICS_TABLE_KEY)
    )
    ast_row_count: int = 0
    metric_row_count: int = 0


@dataclass(frozen=True)
class CstToolOutput(ToolStepOutput):
    """Tool step output for CST extraction."""

    rows: pa.Table = field(default_factory=lambda: empty_table_for_table(CST_NODES_TABLE_KEY))
    row_count: int = 0


@dataclass(frozen=True)
class CstSyntaxIndexToolOutput(ToolStepOutput):
    """Tool step output for combined CST + syntax index extraction."""

    cst_rows: pa.Table = field(default_factory=lambda: empty_table_for_table(CST_NODES_TABLE_KEY))
    parse_manifest_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PARSE_MANIFEST_TABLE_KEY)
    )
    syntax_spans_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_SPANS_TABLE_KEY)
    )
    syntax_nodes_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_NODES_TABLE_KEY)
    )
    syntax_edges_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_EDGES_TABLE_KEY)
    )
    syntax_scopes_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_SCOPES_TABLE_KEY)
    )
    syntax_defs_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_DEFS_TABLE_KEY)
    )
    syntax_refs_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_REFS_TABLE_KEY)
    )
    syntax_calls_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_CALLS_TABLE_KEY)
    )
    syntax_call_args_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_CALL_ARGS_TABLE_KEY)
    )
    syntax_func_params_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_FUNC_PARAMS_TABLE_KEY)
    )
    syntax_imports_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_IMPORTS_TABLE_KEY)
    )
    cst_row_count: int = 0
    parse_manifest_row_count: int = 0
    syntax_spans_row_count: int = 0
    syntax_nodes_row_count: int = 0
    syntax_edges_row_count: int = 0
    syntax_scopes_row_count: int = 0
    syntax_defs_row_count: int = 0
    syntax_refs_row_count: int = 0
    syntax_calls_row_count: int = 0
    syntax_call_args_row_count: int = 0
    syntax_func_params_row_count: int = 0
    syntax_imports_row_count: int = 0


@dataclass(frozen=True)
class SyntaxIndexToolOutput(ToolStepOutput):
    """Tool step output for syntax index extraction."""

    parse_manifest_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PARSE_MANIFEST_TABLE_KEY)
    )
    syntax_spans_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_SPANS_TABLE_KEY)
    )
    syntax_nodes_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_NODES_TABLE_KEY)
    )
    syntax_edges_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_EDGES_TABLE_KEY)
    )
    syntax_scopes_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_SCOPES_TABLE_KEY)
    )
    syntax_defs_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_DEFS_TABLE_KEY)
    )
    syntax_refs_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_REFS_TABLE_KEY)
    )
    syntax_calls_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_CALLS_TABLE_KEY)
    )
    syntax_call_args_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_CALL_ARGS_TABLE_KEY)
    )
    syntax_func_params_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_FUNC_PARAMS_TABLE_KEY)
    )
    syntax_imports_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(SYNTAX_IMPORTS_TABLE_KEY)
    )
    parse_manifest_row_count: int = 0
    syntax_spans_row_count: int = 0
    syntax_nodes_row_count: int = 0
    syntax_edges_row_count: int = 0
    syntax_scopes_row_count: int = 0
    syntax_defs_row_count: int = 0
    syntax_refs_row_count: int = 0
    syntax_calls_row_count: int = 0
    syntax_call_args_row_count: int = 0
    syntax_func_params_row_count: int = 0
    syntax_imports_row_count: int = 0


@dataclass(frozen=True)
class SymtableToolOutput(ToolStepOutput):
    """Tool step output for symtable extraction."""

    scope_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SCOPES_TABLE_KEY)
    )
    symbol_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SYMBOLS_TABLE_KEY)
    )
    scope_edge_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SCOPE_EDGES_TABLE_KEY)
    )
    namespace_edge_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_NAMESPACE_EDGES_TABLE_KEY)
    )
    function_partition_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY)
    )
    binding_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_BINDINGS_TABLE_KEY)
    )
    resolution_edge_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_RESOLUTION_EDGES_TABLE_KEY)
    )
    scope_row_count: int = 0
    symbol_row_count: int = 0
    scope_edge_row_count: int = 0
    namespace_edge_row_count: int = 0
    function_partition_row_count: int = 0
    binding_row_count: int = 0
    resolution_edge_row_count: int = 0


@dataclass(frozen=True)
class BytecodeToolOutput(ToolStepOutput):
    """Tool step output for bytecode extraction."""

    compiler_meta_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_COMPILER_META_TABLE_KEY)
    )
    code_unit_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_CODE_UNITS_TABLE_KEY)
    )
    instruction_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_INSTRUCTIONS_TABLE_KEY)
    )
    exception_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_EXCEPTION_TABLE_KEY)
    )
    block_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_BLOCKS_TABLE_KEY)
    )
    cfg_edge_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_CFG_EDGES_TABLE_KEY)
    )
    defuse_event_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_DEFUSE_EVENTS_TABLE_KEY)
    )
    compiler_meta_row_count: int = 0
    code_unit_row_count: int = 0
    instruction_row_count: int = 0
    exception_row_count: int = 0
    block_row_count: int = 0
    cfg_edge_row_count: int = 0
    defuse_event_row_count: int = 0


@dataclass(frozen=True)
class InspectToolOutput(ToolStepOutput):
    """Tool step output for inspect extraction."""

    object_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_OBJECTS_TABLE_KEY)
    )
    member_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_MEMBERS_TABLE_KEY)
    )
    class_mro_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_CLASS_MRO_TABLE_KEY)
    )
    class_attr_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_CLASS_ATTRS_TABLE_KEY)
    )
    unwrap_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_UNWRAP_TABLE_KEY)
    )
    signature_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_SIGNATURES_TABLE_KEY)
    )
    signature_param_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY)
    )
    annotation_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_ANNOTATIONS_TABLE_KEY)
    )
    source_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_SOURCE_TABLE_KEY)
    )
    runtime_state_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_INSPECT_RUNTIME_STATE_TABLE_KEY)
    )
    object_row_count: int = 0
    member_row_count: int = 0
    class_mro_row_count: int = 0
    class_attr_row_count: int = 0
    unwrap_row_count: int = 0
    signature_row_count: int = 0
    signature_param_row_count: int = 0
    annotation_row_count: int = 0
    source_row_count: int = 0
    runtime_state_row_count: int = 0


def _module_inventory_precheck(
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> tuple[ExecutionResult | None, tuple[str, ...]]:
    if t__modules.status == "succeeded":
        return None, ()
    if not module_records:
        message = t__modules.error or "No module inventory available"
        failure = ExecutionResult.failed(
            f"Upstream modules target {t__modules.status}: {message}",
        )
        return failure, ()
    warnings = (f"Upstream modules target {t__modules.status}; using stored module inventory.",)
    return None, warnings


def _merge_result_warnings(
    result: ExecutionResult,
    warnings: tuple[str, ...],
    *,
    skip_reason: str | None = None,
    error_message: str,
) -> ExecutionResult:
    if not warnings:
        return result

    merged = (*result.warnings, *warnings)
    if result.skipped:
        return ExecutionResult.skip(
            result.skip_reason or skip_reason,
            table_counts=result.table_counts,
            warnings=merged,
        )
    if result.success:
        return ExecutionResult.ok(table_counts=result.table_counts, warnings=merged)
    return ExecutionResult.failed(
        result.error or error_message,
        table_counts=result.table_counts,
        warnings=merged,
    )


def _coerce_ast_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> AstToolOutput:
    if isinstance(output, AstToolOutput):
        if warnings:
            return AstToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="AST extraction failed",
                ),
                ast_rows=output.ast_rows,
                metric_rows=output.metric_rows,
                ast_row_count=output.ast_row_count,
                metric_row_count=output.metric_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="AST extraction failed",
    )
    return AstToolOutput(
        result=merged,
        ast_rows=empty_table_for_table(AST_NODES_TABLE_KEY),
        metric_rows=empty_table_for_table(AST_METRICS_TABLE_KEY),
        ast_row_count=0,
        metric_row_count=0,
    )


def _coerce_cst_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> CstToolOutput:
    if isinstance(output, CstToolOutput):
        if warnings:
            return CstToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="CST extraction failed",
                ),
                rows=output.rows,
                row_count=output.row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="CST extraction failed",
    )
    return CstToolOutput(
        result=merged,
        rows=empty_table_for_table(CST_NODES_TABLE_KEY),
        row_count=0,
    )


def _coerce_cst_syntax_index_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> CstSyntaxIndexToolOutput:
    if isinstance(output, CstSyntaxIndexToolOutput):
        if warnings:
            return CstSyntaxIndexToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="CST/syntax index extraction failed",
                ),
                cst_rows=output.cst_rows,
                parse_manifest_rows=output.parse_manifest_rows,
                syntax_spans_rows=output.syntax_spans_rows,
                syntax_nodes_rows=output.syntax_nodes_rows,
                syntax_edges_rows=output.syntax_edges_rows,
                syntax_scopes_rows=output.syntax_scopes_rows,
                syntax_defs_rows=output.syntax_defs_rows,
                syntax_refs_rows=output.syntax_refs_rows,
                syntax_calls_rows=output.syntax_calls_rows,
                syntax_call_args_rows=output.syntax_call_args_rows,
                syntax_func_params_rows=output.syntax_func_params_rows,
                syntax_imports_rows=output.syntax_imports_rows,
                cst_row_count=output.cst_row_count,
                parse_manifest_row_count=output.parse_manifest_row_count,
                syntax_spans_row_count=output.syntax_spans_row_count,
                syntax_nodes_row_count=output.syntax_nodes_row_count,
                syntax_edges_row_count=output.syntax_edges_row_count,
                syntax_scopes_row_count=output.syntax_scopes_row_count,
                syntax_defs_row_count=output.syntax_defs_row_count,
                syntax_refs_row_count=output.syntax_refs_row_count,
                syntax_calls_row_count=output.syntax_calls_row_count,
                syntax_call_args_row_count=output.syntax_call_args_row_count,
                syntax_func_params_row_count=output.syntax_func_params_row_count,
                syntax_imports_row_count=output.syntax_imports_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="CST/syntax index extraction failed",
    )
    return CstSyntaxIndexToolOutput(
        result=merged,
        cst_rows=empty_table_for_table(CST_NODES_TABLE_KEY),
        parse_manifest_rows=empty_table_for_table(PARSE_MANIFEST_TABLE_KEY),
        syntax_spans_rows=empty_table_for_table(SYNTAX_SPANS_TABLE_KEY),
        syntax_nodes_rows=empty_table_for_table(SYNTAX_NODES_TABLE_KEY),
        syntax_edges_rows=empty_table_for_table(SYNTAX_EDGES_TABLE_KEY),
        syntax_scopes_rows=empty_table_for_table(SYNTAX_SCOPES_TABLE_KEY),
        syntax_defs_rows=empty_table_for_table(SYNTAX_DEFS_TABLE_KEY),
        syntax_refs_rows=empty_table_for_table(SYNTAX_REFS_TABLE_KEY),
        syntax_calls_rows=empty_table_for_table(SYNTAX_CALLS_TABLE_KEY),
        syntax_call_args_rows=empty_table_for_table(SYNTAX_CALL_ARGS_TABLE_KEY),
        syntax_func_params_rows=empty_table_for_table(SYNTAX_FUNC_PARAMS_TABLE_KEY),
        syntax_imports_rows=empty_table_for_table(SYNTAX_IMPORTS_TABLE_KEY),
        cst_row_count=0,
        parse_manifest_row_count=0,
        syntax_spans_row_count=0,
        syntax_nodes_row_count=0,
        syntax_edges_row_count=0,
        syntax_scopes_row_count=0,
        syntax_defs_row_count=0,
        syntax_refs_row_count=0,
        syntax_calls_row_count=0,
        syntax_call_args_row_count=0,
        syntax_func_params_row_count=0,
        syntax_imports_row_count=0,
    )


def _coerce_syntax_index_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> SyntaxIndexToolOutput:
    if isinstance(output, SyntaxIndexToolOutput):
        if warnings:
            return SyntaxIndexToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Syntax index extraction failed",
                ),
                parse_manifest_rows=output.parse_manifest_rows,
                syntax_spans_rows=output.syntax_spans_rows,
                syntax_nodes_rows=output.syntax_nodes_rows,
                syntax_edges_rows=output.syntax_edges_rows,
                syntax_scopes_rows=output.syntax_scopes_rows,
                syntax_defs_rows=output.syntax_defs_rows,
                syntax_refs_rows=output.syntax_refs_rows,
                syntax_calls_rows=output.syntax_calls_rows,
                syntax_call_args_rows=output.syntax_call_args_rows,
                syntax_func_params_rows=output.syntax_func_params_rows,
                syntax_imports_rows=output.syntax_imports_rows,
                parse_manifest_row_count=output.parse_manifest_row_count,
                syntax_spans_row_count=output.syntax_spans_row_count,
                syntax_nodes_row_count=output.syntax_nodes_row_count,
                syntax_edges_row_count=output.syntax_edges_row_count,
                syntax_scopes_row_count=output.syntax_scopes_row_count,
                syntax_defs_row_count=output.syntax_defs_row_count,
                syntax_refs_row_count=output.syntax_refs_row_count,
                syntax_calls_row_count=output.syntax_calls_row_count,
                syntax_call_args_row_count=output.syntax_call_args_row_count,
                syntax_func_params_row_count=output.syntax_func_params_row_count,
                syntax_imports_row_count=output.syntax_imports_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Syntax index extraction failed",
    )
    return SyntaxIndexToolOutput(
        result=merged,
        parse_manifest_rows=empty_table_for_table(PARSE_MANIFEST_TABLE_KEY),
        syntax_spans_rows=empty_table_for_table(SYNTAX_SPANS_TABLE_KEY),
        syntax_nodes_rows=empty_table_for_table(SYNTAX_NODES_TABLE_KEY),
        syntax_edges_rows=empty_table_for_table(SYNTAX_EDGES_TABLE_KEY),
        syntax_scopes_rows=empty_table_for_table(SYNTAX_SCOPES_TABLE_KEY),
        syntax_defs_rows=empty_table_for_table(SYNTAX_DEFS_TABLE_KEY),
        syntax_refs_rows=empty_table_for_table(SYNTAX_REFS_TABLE_KEY),
        syntax_calls_rows=empty_table_for_table(SYNTAX_CALLS_TABLE_KEY),
        syntax_call_args_rows=empty_table_for_table(SYNTAX_CALL_ARGS_TABLE_KEY),
        syntax_func_params_rows=empty_table_for_table(SYNTAX_FUNC_PARAMS_TABLE_KEY),
        syntax_imports_rows=empty_table_for_table(SYNTAX_IMPORTS_TABLE_KEY),
        parse_manifest_row_count=0,
        syntax_spans_row_count=0,
        syntax_nodes_row_count=0,
        syntax_edges_row_count=0,
        syntax_scopes_row_count=0,
        syntax_defs_row_count=0,
        syntax_refs_row_count=0,
        syntax_calls_row_count=0,
        syntax_call_args_row_count=0,
        syntax_func_params_row_count=0,
        syntax_imports_row_count=0,
    )


def _coerce_symtable_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> SymtableToolOutput:
    if isinstance(output, SymtableToolOutput):
        if warnings:
            return SymtableToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Symtable extraction failed",
                ),
                scope_rows=output.scope_rows,
                symbol_rows=output.symbol_rows,
                scope_edge_rows=output.scope_edge_rows,
                namespace_edge_rows=output.namespace_edge_rows,
                function_partition_rows=output.function_partition_rows,
                binding_rows=output.binding_rows,
                resolution_edge_rows=output.resolution_edge_rows,
                scope_row_count=output.scope_row_count,
                symbol_row_count=output.symbol_row_count,
                scope_edge_row_count=output.scope_edge_row_count,
                namespace_edge_row_count=output.namespace_edge_row_count,
                function_partition_row_count=output.function_partition_row_count,
                binding_row_count=output.binding_row_count,
                resolution_edge_row_count=output.resolution_edge_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Symtable extraction failed",
    )
    return SymtableToolOutput(
        result=merged,
        scope_rows=empty_table_for_table(PY_SYM_SCOPES_TABLE_KEY),
        symbol_rows=empty_table_for_table(PY_SYM_SYMBOLS_TABLE_KEY),
        scope_edge_rows=empty_table_for_table(PY_SYM_SCOPE_EDGES_TABLE_KEY),
        namespace_edge_rows=empty_table_for_table(PY_SYM_NAMESPACE_EDGES_TABLE_KEY),
        function_partition_rows=empty_table_for_table(PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY),
        binding_rows=empty_table_for_table(PY_SYM_BINDINGS_TABLE_KEY),
        resolution_edge_rows=empty_table_for_table(PY_SYM_RESOLUTION_EDGES_TABLE_KEY),
        scope_row_count=0,
        symbol_row_count=0,
        scope_edge_row_count=0,
        namespace_edge_row_count=0,
        function_partition_row_count=0,
        binding_row_count=0,
        resolution_edge_row_count=0,
    )


def _coerce_bytecode_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> BytecodeToolOutput:
    if isinstance(output, BytecodeToolOutput):
        if warnings:
            return BytecodeToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Bytecode extraction failed",
                ),
                compiler_meta_rows=output.compiler_meta_rows,
                code_unit_rows=output.code_unit_rows,
                instruction_rows=output.instruction_rows,
                exception_rows=output.exception_rows,
                block_rows=output.block_rows,
                cfg_edge_rows=output.cfg_edge_rows,
                defuse_event_rows=output.defuse_event_rows,
                compiler_meta_row_count=output.compiler_meta_row_count,
                code_unit_row_count=output.code_unit_row_count,
                instruction_row_count=output.instruction_row_count,
                exception_row_count=output.exception_row_count,
                block_row_count=output.block_row_count,
                cfg_edge_row_count=output.cfg_edge_row_count,
                defuse_event_row_count=output.defuse_event_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Bytecode extraction failed",
    )
    return BytecodeToolOutput(
        result=merged,
        compiler_meta_rows=empty_table_for_table(PY_COMPILER_META_TABLE_KEY),
        code_unit_rows=empty_table_for_table(PY_BC_CODE_UNITS_TABLE_KEY),
        instruction_rows=empty_table_for_table(PY_BC_INSTRUCTIONS_TABLE_KEY),
        exception_rows=empty_table_for_table(PY_BC_EXCEPTION_TABLE_KEY),
        block_rows=empty_table_for_table(PY_BC_BLOCKS_TABLE_KEY),
        cfg_edge_rows=empty_table_for_table(PY_BC_CFG_EDGES_TABLE_KEY),
        defuse_event_rows=empty_table_for_table(PY_BC_DEFUSE_EVENTS_TABLE_KEY),
        compiler_meta_row_count=0,
        code_unit_row_count=0,
        instruction_row_count=0,
        exception_row_count=0,
        block_row_count=0,
        cfg_edge_row_count=0,
        defuse_event_row_count=0,
    )


def _resolve_ingest_run_id(env: BuildEnv) -> str:
    run_context = env.run_context
    if run_context is not None:
        return run_context.run_id
    return new_run_id(RUN_PREFIX_INGEST)


def _py_compiler_meta_frame(
    env: BuildEnv,
    options: BytecodeExtractOptions,
) -> pa.Table:
    run_id = _resolve_ingest_run_id(env)
    rows = [
        {
            "repo": env.repo,
            "commit": env.commit,
            "run_id": run_id,
            "python_version": sys.version.split()[0],
            "magic_number": importlib.util.MAGIC_NUMBER,
            "optimize": options.optimize,
            "dont_inherit": options.dont_inherit,
            "flags": options.compile_flags,
        }
    ]
    reader, _ = table_for_rows(PY_COMPILER_META_TABLE_KEY, rows)
    return reader


def _coerce_inspect_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> InspectToolOutput:
    if isinstance(output, InspectToolOutput):
        if warnings:
            return InspectToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Inspect extraction failed",
                ),
                object_rows=output.object_rows,
                member_rows=output.member_rows,
                class_mro_rows=output.class_mro_rows,
                class_attr_rows=output.class_attr_rows,
                unwrap_rows=output.unwrap_rows,
                signature_rows=output.signature_rows,
                signature_param_rows=output.signature_param_rows,
                annotation_rows=output.annotation_rows,
                source_rows=output.source_rows,
                runtime_state_rows=output.runtime_state_rows,
                object_row_count=output.object_row_count,
                member_row_count=output.member_row_count,
                class_mro_row_count=output.class_mro_row_count,
                class_attr_row_count=output.class_attr_row_count,
                unwrap_row_count=output.unwrap_row_count,
                signature_row_count=output.signature_row_count,
                signature_param_row_count=output.signature_param_row_count,
                annotation_row_count=output.annotation_row_count,
                source_row_count=output.source_row_count,
                runtime_state_row_count=output.runtime_state_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Inspect extraction failed",
    )
    return InspectToolOutput(
        result=merged,
        object_rows=empty_table_for_table(PY_INSPECT_OBJECTS_TABLE_KEY),
        member_rows=empty_table_for_table(PY_INSPECT_MEMBERS_TABLE_KEY),
        class_mro_rows=empty_table_for_table(PY_INSPECT_CLASS_MRO_TABLE_KEY),
        class_attr_rows=empty_table_for_table(PY_INSPECT_CLASS_ATTRS_TABLE_KEY),
        unwrap_rows=empty_table_for_table(PY_INSPECT_UNWRAP_TABLE_KEY),
        signature_rows=empty_table_for_table(PY_INSPECT_SIGNATURES_TABLE_KEY),
        signature_param_rows=empty_table_for_table(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY),
        annotation_rows=empty_table_for_table(PY_INSPECT_ANNOTATIONS_TABLE_KEY),
        source_rows=empty_table_for_table(PY_INSPECT_SOURCE_TABLE_KEY),
        runtime_state_rows=empty_table_for_table(PY_INSPECT_RUNTIME_STATE_TABLE_KEY),
        object_row_count=0,
        member_row_count=0,
        class_mro_row_count=0,
        class_attr_row_count=0,
        unwrap_row_count=0,
        signature_row_count=0,
        signature_param_row_count=0,
        annotation_row_count=0,
        source_row_count=0,
        runtime_state_row_count=0,
    )


def _coerce_docstrings_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> DocstringsToolOutput:
    if isinstance(output, DocstringsToolOutput):
        if warnings:
            return DocstringsToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Docstrings extraction failed",
                ),
                rows=output.rows,
                row_count=output.row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Docstrings extraction failed",
    )
    return DocstringsToolOutput(
        result=merged,
        rows=empty_table_for_table(DOCSTRINGS_TABLE_KEY),
        row_count=0,
    )


def t__ast__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    py_frontend: PyFrontend,
) -> AstToolOutput:
    """Execute AST extraction on repository modules.

    Returns
    -------
    AstToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return AstToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=AST_TARGET_NAME,
    )

    def _execute() -> AstToolOutput:
        get_schema_service()
        options = load_target_options(
            env,
            target_name=AST_TARGET_NAME,
            options_type=AstExtractOptions,
        )
        step = AstExtractStep(
            discovery=py_frontend.discovery,
            options=options,
            frontend=py_frontend,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return AstToolOutput(
            result=extract_result.result,
            ast_rows=extract_result.ast_rows_reader,
            metric_rows=extract_result.metric_rows_reader,
            ast_row_count=extract_result.ast_row_count,
            metric_row_count=extract_result.metric_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_ast_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("AST extraction warning: %s", warning)
    return coerced


def t__ast__ingest(
    t__ast__run: AstToolOutput,
) -> IngestStep[TabularByTable]:
    """Package AST rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__ast__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "AST extraction skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "AST extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {
        AST_NODES_TABLE_KEY: t__ast__run.ast_rows,
        AST_METRICS_TABLE_KEY: t__ast__run.metric_rows,
    }
    table_counts = {
        AST_NODES_TABLE_KEY: t__ast__run.ast_row_count,
        AST_METRICS_TABLE_KEY: t__ast__run.metric_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def py_frontend__cst_syntax_index__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    py_frontend: PyFrontend,
) -> CstSyntaxIndexToolOutput:
    """Execute CST + syntax index extraction once per module set.

    Returns
    -------
    CstSyntaxIndexToolOutput
        Combined CST + syntax index tool output.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return CstSyntaxIndexToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=f"{CST_TARGET_NAME}+{SYNTAX_INDEX_TARGET_NAME}",
    )

    def _execute() -> CstSyntaxIndexToolOutput:
        get_schema_service()
        options_cst = load_target_options(
            env,
            target_name=CST_TARGET_NAME,
            options_type=SyntaxIndexOptions,
        )
        options_syntax = load_target_options(
            env,
            target_name=SYNTAX_INDEX_TARGET_NAME,
            options_type=SyntaxIndexOptions,
        )
        step = CstExtractStep(
            discovery=py_frontend.discovery,
            emit_ast_nodes=options_syntax.emit_ast_nodes,
            batch_size=max(options_cst.batch_size, options_syntax.batch_size),
            frontend=py_frontend,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return CstSyntaxIndexToolOutput(
            result=extract_result.result,
            cst_rows=extract_result.rows_reader,
            parse_manifest_rows=extract_result.parse_manifest_rows_reader,
            syntax_spans_rows=extract_result.syntax_spans_rows_reader,
            syntax_nodes_rows=extract_result.syntax_nodes_rows_reader,
            syntax_edges_rows=extract_result.syntax_edges_rows_reader,
            syntax_scopes_rows=extract_result.syntax_scopes_rows_reader,
            syntax_defs_rows=extract_result.syntax_defs_rows_reader,
            syntax_refs_rows=extract_result.syntax_refs_rows_reader,
            syntax_calls_rows=extract_result.syntax_calls_rows_reader,
            syntax_call_args_rows=extract_result.syntax_call_args_rows_reader,
            syntax_func_params_rows=extract_result.syntax_func_params_rows_reader,
            syntax_imports_rows=extract_result.syntax_imports_rows_reader,
            cst_row_count=extract_result.row_count,
            parse_manifest_row_count=extract_result.parse_manifest_row_count,
            syntax_spans_row_count=extract_result.syntax_spans_row_count,
            syntax_nodes_row_count=extract_result.syntax_nodes_row_count,
            syntax_edges_row_count=extract_result.syntax_edges_row_count,
            syntax_scopes_row_count=extract_result.syntax_scopes_row_count,
            syntax_defs_row_count=extract_result.syntax_defs_row_count,
            syntax_refs_row_count=extract_result.syntax_refs_row_count,
            syntax_calls_row_count=extract_result.syntax_calls_row_count,
            syntax_call_args_row_count=extract_result.syntax_call_args_row_count,
            syntax_func_params_row_count=extract_result.syntax_func_params_row_count,
            syntax_imports_row_count=extract_result.syntax_imports_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_cst_syntax_index_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("CST/syntax index extraction warning: %s", warning)
    return coerced


def t__cst__run(
    py_frontend__cst_syntax_index__run: CstSyntaxIndexToolOutput,
) -> CstToolOutput:
    """Execute CST extraction on repository modules.

    Returns
    -------
    CstToolOutput
        Tool output with row payloads and execution status.
    """
    output = py_frontend__cst_syntax_index__run
    return CstToolOutput(
        result=output.result,
        rows=output.cst_rows,
        row_count=output.cst_row_count,
    )


def t__cst__ingest(
    t__cst__run: CstToolOutput,
) -> IngestStep[TabularByTable]:
    """Package CST rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__cst__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "CST extraction skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "CST extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {CST_NODES_TABLE_KEY: t__cst__run.rows}
    table_counts = {CST_NODES_TABLE_KEY: t__cst__run.row_count}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def t__syntax_index__run(
    py_frontend__cst_syntax_index__run: CstSyntaxIndexToolOutput,
) -> SyntaxIndexToolOutput:
    """Execute syntax index extraction on repository modules.

    Returns
    -------
    SyntaxIndexToolOutput
        Tool output with row payloads and execution status.
    """
    output = py_frontend__cst_syntax_index__run
    return SyntaxIndexToolOutput(
        result=output.result,
        parse_manifest_rows=output.parse_manifest_rows,
        syntax_spans_rows=output.syntax_spans_rows,
        syntax_nodes_rows=output.syntax_nodes_rows,
        syntax_edges_rows=output.syntax_edges_rows,
        syntax_scopes_rows=output.syntax_scopes_rows,
        syntax_defs_rows=output.syntax_defs_rows,
        syntax_refs_rows=output.syntax_refs_rows,
        syntax_calls_rows=output.syntax_calls_rows,
        syntax_call_args_rows=output.syntax_call_args_rows,
        syntax_func_params_rows=output.syntax_func_params_rows,
        syntax_imports_rows=output.syntax_imports_rows,
        parse_manifest_row_count=output.parse_manifest_row_count,
        syntax_spans_row_count=output.syntax_spans_row_count,
        syntax_nodes_row_count=output.syntax_nodes_row_count,
        syntax_edges_row_count=output.syntax_edges_row_count,
        syntax_scopes_row_count=output.syntax_scopes_row_count,
        syntax_defs_row_count=output.syntax_defs_row_count,
        syntax_refs_row_count=output.syntax_refs_row_count,
        syntax_calls_row_count=output.syntax_calls_row_count,
        syntax_call_args_row_count=output.syntax_call_args_row_count,
        syntax_func_params_row_count=output.syntax_func_params_row_count,
        syntax_imports_row_count=output.syntax_imports_row_count,
    )


def t__syntax_index__ingest(
    t__syntax_index__run: SyntaxIndexToolOutput,
) -> IngestStep[TabularByTable]:
    """Package syntax index rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__syntax_index__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Syntax index skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Syntax index extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {
        PARSE_MANIFEST_TABLE_KEY: t__syntax_index__run.parse_manifest_rows,
        SYNTAX_SPANS_TABLE_KEY: t__syntax_index__run.syntax_spans_rows,
        SYNTAX_NODES_TABLE_KEY: t__syntax_index__run.syntax_nodes_rows,
        SYNTAX_EDGES_TABLE_KEY: t__syntax_index__run.syntax_edges_rows,
        SYNTAX_SCOPES_TABLE_KEY: t__syntax_index__run.syntax_scopes_rows,
        SYNTAX_DEFS_TABLE_KEY: t__syntax_index__run.syntax_defs_rows,
        SYNTAX_REFS_TABLE_KEY: t__syntax_index__run.syntax_refs_rows,
        SYNTAX_CALLS_TABLE_KEY: t__syntax_index__run.syntax_calls_rows,
        SYNTAX_CALL_ARGS_TABLE_KEY: t__syntax_index__run.syntax_call_args_rows,
        SYNTAX_FUNC_PARAMS_TABLE_KEY: t__syntax_index__run.syntax_func_params_rows,
        SYNTAX_IMPORTS_TABLE_KEY: t__syntax_index__run.syntax_imports_rows,
    }
    table_counts = {
        PARSE_MANIFEST_TABLE_KEY: t__syntax_index__run.parse_manifest_row_count,
        SYNTAX_SPANS_TABLE_KEY: t__syntax_index__run.syntax_spans_row_count,
        SYNTAX_NODES_TABLE_KEY: t__syntax_index__run.syntax_nodes_row_count,
        SYNTAX_EDGES_TABLE_KEY: t__syntax_index__run.syntax_edges_row_count,
        SYNTAX_SCOPES_TABLE_KEY: t__syntax_index__run.syntax_scopes_row_count,
        SYNTAX_DEFS_TABLE_KEY: t__syntax_index__run.syntax_defs_row_count,
        SYNTAX_REFS_TABLE_KEY: t__syntax_index__run.syntax_refs_row_count,
        SYNTAX_CALLS_TABLE_KEY: t__syntax_index__run.syntax_calls_row_count,
        SYNTAX_CALL_ARGS_TABLE_KEY: t__syntax_index__run.syntax_call_args_row_count,
        SYNTAX_FUNC_PARAMS_TABLE_KEY: t__syntax_index__run.syntax_func_params_row_count,
        SYNTAX_IMPORTS_TABLE_KEY: t__syntax_index__run.syntax_imports_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def t__symtable__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    py_frontend: PyFrontend,
) -> SymtableToolOutput:
    """Execute symtable extraction on repository modules.

    Returns
    -------
    SymtableToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return SymtableToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=SYMTABLE_TARGET_NAME,
    )

    def _execute() -> SymtableToolOutput:
        get_schema_service()
        options = load_target_options(
            env,
            target_name=SYMTABLE_TARGET_NAME,
            options_type=SymtableExtractOptions,
        )
        if not options.enable:
            return SymtableToolOutput(
                result=ExecutionResult.skip("Symtable extraction disabled by options")
            )
        step = SymtableExtractStep(
            discovery=py_frontend.discovery,
            options=options,
            frontend=py_frontend,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return SymtableToolOutput(
            result=extract_result.result,
            scope_rows=extract_result.scope_rows_reader,
            symbol_rows=extract_result.symbol_rows_reader,
            scope_edge_rows=extract_result.scope_edge_rows_reader,
            namespace_edge_rows=extract_result.namespace_edge_rows_reader,
            function_partition_rows=extract_result.function_partition_rows_reader,
            binding_rows=extract_result.binding_rows_reader,
            resolution_edge_rows=extract_result.resolution_edge_rows_reader,
            scope_row_count=extract_result.scope_row_count,
            symbol_row_count=extract_result.symbol_row_count,
            scope_edge_row_count=extract_result.scope_edge_row_count,
            namespace_edge_row_count=extract_result.namespace_edge_row_count,
            function_partition_row_count=extract_result.function_partition_row_count,
            binding_row_count=extract_result.binding_row_count,
            resolution_edge_row_count=extract_result.resolution_edge_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_symtable_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("Symtable extraction warning: %s", warning)
    return coerced


_JOIN_STRING_KEYS = {"repo", "commit", "rel_path", "binding_id"}


@dataclass(frozen=True, slots=True)
class _JoinSpec:
    left_keys: Sequence[str]
    right_keys: Sequence[str]
    left_table_key: str | None = None
    right_table_key: str | None = None


def _join_casts(keys: Sequence[str]) -> dict[str, str]:
    casts: dict[str, str] = {}
    for key in keys:
        if key in _JOIN_STRING_KEYS:
            casts[key] = "string"
    return casts


def _project_with_cast(
    table: pa.Table,
    *,
    casts: dict[str, str],
) -> dict[str, pc.Expression]:
    exprs: dict[str, pc.Expression] = {}
    for name in table.column_names:
        if name in casts:
            exprs[name] = E.cast(E.field(name), casts[name])
        else:
            exprs[name] = E.field(name)
    return exprs


def _precheck_join_table(
    table: pa.Table,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0 or not join_keys:
        return table
    if table_key is None:
        result = finalize_join_keys(
            table,
            required_non_null=join_keys,
            key_fields=join_keys,
        )
    else:
        result = finalize_table(
            table,
            spec=FinalizeSpec(
                table_key=table_key,
                mode="tolerant",
                required_non_null=join_keys,
                key_fields=join_keys,
                dedupe=FinalizeDedupe(enabled=False),
                target_name=SYMTABLE_TARGET_NAME,
            ),
        )
    _log_join_precheck_errors(result, table_key=table_key, join_keys=join_keys)
    return result.good


def _log_join_precheck_errors(
    result: FinalizeResult,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> None:
    if result.errors.num_rows == 0:
        return
    table_label = table_key or "derived"
    log.warning(
        "Join key precheck dropped %d rows table=%s keys=%s",
        result.errors.num_rows,
        table_label,
        ",".join(join_keys),
    )


def _hash_join_reader(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: _JoinSpec,
    how: JoinType = "left outer",
) -> pa.RecordBatchReader:
    left_checked = _precheck_join_table(
        left,
        table_key=spec.left_table_key,
        join_keys=spec.left_keys,
    )
    right_checked = _precheck_join_table(
        right,
        table_key=spec.right_table_key,
        join_keys=spec.right_keys,
    )
    left_exprs = _project_with_cast(left, casts=_join_casts(spec.left_keys))
    right_exprs = _project_with_cast(right, casts=_join_casts(spec.right_keys))
    left_plan = Plan.table(left_checked).project(left_exprs)
    right_plan = Plan.table(right_checked).project(right_exprs)
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=list(spec.left_keys),
            right_keys=list(spec.right_keys),
            how=how,
            left_output=list(left_exprs.keys()),
        ),
    )
    joined = joined.order_by(sort_keys=[(key, "ascending") for key in spec.left_keys])
    return joined.to_reader(use_threads=True)


def _build_py_sym_unresolved_bindings(
    resolution_edges: pa.Table,
    bindings: pa.Table,
) -> pa.Table:
    table_key = PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY
    required = {"repo", "commit", "rel_path", "dst_binding_id", "kind"}
    if resolution_edges.num_rows == 0 or not required.issubset(resolution_edges.column_names):
        return empty_table_for_table(table_key)
    ends_with_mask = array_from_compute(
        "ends_with",
        [resolution_edges["dst_binding_id"], pa.scalar(":unknown")],
    )
    if ends_with_mask is None:
        ends_with_mask = array_from_compute(
            "match_substring_regex",
            [resolution_edges["dst_binding_id"], pa.scalar(":unknown$")],
        )
    if ends_with_mask is None:
        ends_with_mask = pa.array([False] * resolution_edges.num_rows)
    unknown_mask = or_kleene(
        ends_with_mask,
        equal_mask(resolution_edges["kind"], "UNKNOWN"),
    )
    unknown = safe_filter(resolution_edges, unknown_mask)
    if unknown.num_rows == 0:
        return empty_table_for_table(table_key)
    unknown = unknown.select(
        ["repo", "commit", "rel_path", "dst_binding_id", "kind", "confidence", "reason"]
    ).rename_columns(
        ["repo", "commit", "rel_path", "binding_id", "resolution_kind", "confidence", "reason"]
    )
    binding_required = {"repo", "commit", "rel_path", "binding_id"}
    if bindings.num_rows == 0 or not binding_required.issubset(bindings.column_names):
        return finalize_ingest_table(
            table_key,
            unknown,
            target_name=SYMTABLE_TARGET_NAME,
        )
    left = normalize_table_for_join(unknown)
    right = normalize_table_for_join(bindings.select(["repo", "commit", "rel_path", "binding_id"]))
    join_keys = ["repo", "commit", "rel_path", "binding_id"]
    joined_reader = _hash_join_reader(
        left,
        right,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            left_table_key=PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY,
            right_table_key=PY_SYM_BINDINGS_TABLE_KEY,
        ),
        how="left anti",
    )
    return finalize_ingest_reader(
        table_key,
        joined_reader,
        target_name=SYMTABLE_TARGET_NAME,
    )


def t__symtable__ingest(
    t__symtable__run: SymtableToolOutput,
) -> IngestStep[TabularByTable]:
    """Package symtable rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__symtable__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Symtable extraction skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Symtable extraction failed",
                warnings=result.warnings,
            )
        )

    unresolved_bindings = _build_py_sym_unresolved_bindings(
        t__symtable__run.resolution_edge_rows,
        t__symtable__run.binding_rows,
    )
    scope_rows = finalize_ingest_table(
        PY_SYM_SCOPES_TABLE_KEY,
        t__symtable__run.scope_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    symbol_rows = finalize_ingest_table(
        PY_SYM_SYMBOLS_TABLE_KEY,
        t__symtable__run.symbol_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    scope_edge_rows = finalize_ingest_table(
        PY_SYM_SCOPE_EDGES_TABLE_KEY,
        t__symtable__run.scope_edge_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    namespace_edge_rows = finalize_ingest_table(
        PY_SYM_NAMESPACE_EDGES_TABLE_KEY,
        t__symtable__run.namespace_edge_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    function_partition_rows = finalize_ingest_table(
        PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY,
        t__symtable__run.function_partition_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    binding_rows = finalize_ingest_table(
        PY_SYM_BINDINGS_TABLE_KEY,
        t__symtable__run.binding_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    resolution_edge_rows = finalize_ingest_table(
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
        t__symtable__run.resolution_edge_rows,
        target_name=SYMTABLE_TARGET_NAME,
    )
    payload = {
        PY_SYM_SCOPES_TABLE_KEY: scope_rows,
        PY_SYM_SYMBOLS_TABLE_KEY: symbol_rows,
        PY_SYM_SCOPE_EDGES_TABLE_KEY: scope_edge_rows,
        PY_SYM_NAMESPACE_EDGES_TABLE_KEY: namespace_edge_rows,
        PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY: function_partition_rows,
        PY_SYM_BINDINGS_TABLE_KEY: binding_rows,
        PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY: unresolved_bindings,
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY: resolution_edge_rows,
    }
    table_counts = {
        PY_SYM_SCOPES_TABLE_KEY: scope_rows.num_rows,
        PY_SYM_SYMBOLS_TABLE_KEY: symbol_rows.num_rows,
        PY_SYM_SCOPE_EDGES_TABLE_KEY: scope_edge_rows.num_rows,
        PY_SYM_NAMESPACE_EDGES_TABLE_KEY: namespace_edge_rows.num_rows,
        PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY: function_partition_rows.num_rows,
        PY_SYM_BINDINGS_TABLE_KEY: binding_rows.num_rows,
        PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY: unresolved_bindings.num_rows,
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY: resolution_edge_rows.num_rows,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def t__bytecode__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    py_frontend: PyFrontend,
) -> BytecodeToolOutput:
    """Execute bytecode extraction on repository modules.

    Returns
    -------
    BytecodeToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return BytecodeToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=BYTECODE_TARGET_NAME,
    )

    def _execute() -> BytecodeToolOutput:
        get_schema_service()
        options = load_target_options(
            env,
            target_name=BYTECODE_TARGET_NAME,
            options_type=BytecodeExtractOptions,
        )
        if not options.enable:
            return BytecodeToolOutput(
                result=ExecutionResult.skip("Bytecode extraction disabled by options")
            )
        if options.cache_dir is None:
            options = replace(options, cache_dir=env.paths.tool_cache / "bytecode")
        step = DisExtractStep(
            discovery=py_frontend.discovery,
            options=options,
            frontend=py_frontend,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        compiler_meta_frame = _py_compiler_meta_frame(env, options)
        return BytecodeToolOutput(
            result=extract_result.result,
            compiler_meta_rows=compiler_meta_frame,
            code_unit_rows=extract_result.code_unit_rows_reader,
            instruction_rows=extract_result.instruction_rows_reader,
            exception_rows=extract_result.exception_rows_reader,
            block_rows=extract_result.block_rows_reader,
            cfg_edge_rows=extract_result.cfg_edge_rows_reader,
            defuse_event_rows=extract_result.defuse_event_rows_reader,
            compiler_meta_row_count=1,
            code_unit_row_count=extract_result.code_unit_row_count,
            instruction_row_count=extract_result.instruction_row_count,
            exception_row_count=extract_result.exception_row_count,
            block_row_count=extract_result.block_row_count,
            cfg_edge_row_count=extract_result.cfg_edge_row_count,
            defuse_event_row_count=extract_result.defuse_event_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_bytecode_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("Bytecode extraction warning: %s", warning)
    return coerced


def t__bytecode__ingest(
    t__bytecode__run: BytecodeToolOutput,
) -> IngestStep[TabularByTable]:
    """Package bytecode rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__bytecode__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Bytecode extraction skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Bytecode extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {
        PY_COMPILER_META_TABLE_KEY: t__bytecode__run.compiler_meta_rows,
        PY_BC_CODE_UNITS_TABLE_KEY: t__bytecode__run.code_unit_rows,
        PY_BC_INSTRUCTIONS_TABLE_KEY: t__bytecode__run.instruction_rows,
        PY_BC_EXCEPTION_TABLE_KEY: t__bytecode__run.exception_rows,
        PY_BC_BLOCKS_TABLE_KEY: t__bytecode__run.block_rows,
        PY_BC_CFG_EDGES_TABLE_KEY: t__bytecode__run.cfg_edge_rows,
        PY_BC_DEFUSE_EVENTS_TABLE_KEY: t__bytecode__run.defuse_event_rows,
    }
    table_counts = {
        PY_COMPILER_META_TABLE_KEY: t__bytecode__run.compiler_meta_row_count,
        PY_BC_CODE_UNITS_TABLE_KEY: t__bytecode__run.code_unit_row_count,
        PY_BC_INSTRUCTIONS_TABLE_KEY: t__bytecode__run.instruction_row_count,
        PY_BC_EXCEPTION_TABLE_KEY: t__bytecode__run.exception_row_count,
        PY_BC_BLOCKS_TABLE_KEY: t__bytecode__run.block_row_count,
        PY_BC_CFG_EDGES_TABLE_KEY: t__bytecode__run.cfg_edge_row_count,
        PY_BC_DEFUSE_EVENTS_TABLE_KEY: t__bytecode__run.defuse_event_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def t__inspect__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> InspectToolOutput:
    """Execute inspect extraction on repository modules.

    Returns
    -------
    InspectToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return InspectToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=INSPECT_TARGET_NAME,
    )

    def _execute() -> InspectToolOutput:
        get_schema_service()
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        options = load_target_options(
            env,
            target_name=INSPECT_TARGET_NAME,
            options_type=InspectExtractOptions,
        )
        step = InspectExtractStep(discovery=discovery, options=options)
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return InspectToolOutput(
            result=extract_result.result,
            object_rows=extract_result.object_rows_reader,
            member_rows=extract_result.member_rows_reader,
            class_mro_rows=extract_result.class_mro_rows_reader,
            class_attr_rows=extract_result.class_attr_rows_reader,
            unwrap_rows=extract_result.unwrap_rows_reader,
            signature_rows=extract_result.signature_rows_reader,
            signature_param_rows=extract_result.signature_param_rows_reader,
            annotation_rows=extract_result.annotation_rows_reader,
            source_rows=extract_result.source_rows_reader,
            runtime_state_rows=extract_result.runtime_state_rows_reader,
            object_row_count=extract_result.object_row_count,
            member_row_count=extract_result.member_row_count,
            class_mro_row_count=extract_result.class_mro_row_count,
            class_attr_row_count=extract_result.class_attr_row_count,
            unwrap_row_count=extract_result.unwrap_row_count,
            signature_row_count=extract_result.signature_row_count,
            signature_param_row_count=extract_result.signature_param_row_count,
            annotation_row_count=extract_result.annotation_row_count,
            source_row_count=extract_result.source_row_count,
            runtime_state_row_count=extract_result.runtime_state_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_inspect_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("Inspect extraction warning: %s", warning)
    return coerced


def t__inspect__ingest(
    t__inspect__run: InspectToolOutput,
) -> IngestStep[TabularByTable]:
    """Package inspect rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__inspect__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Inspect extraction skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Inspect extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {
        PY_INSPECT_OBJECTS_TABLE_KEY: t__inspect__run.object_rows,
        PY_INSPECT_MEMBERS_TABLE_KEY: t__inspect__run.member_rows,
        PY_INSPECT_CLASS_MRO_TABLE_KEY: t__inspect__run.class_mro_rows,
        PY_INSPECT_CLASS_ATTRS_TABLE_KEY: t__inspect__run.class_attr_rows,
        PY_INSPECT_UNWRAP_TABLE_KEY: t__inspect__run.unwrap_rows,
        PY_INSPECT_SIGNATURES_TABLE_KEY: t__inspect__run.signature_rows,
        PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY: t__inspect__run.signature_param_rows,
        PY_INSPECT_ANNOTATIONS_TABLE_KEY: t__inspect__run.annotation_rows,
        PY_INSPECT_SOURCE_TABLE_KEY: t__inspect__run.source_rows,
        PY_INSPECT_RUNTIME_STATE_TABLE_KEY: t__inspect__run.runtime_state_rows,
    }
    table_counts = {
        PY_INSPECT_OBJECTS_TABLE_KEY: t__inspect__run.object_row_count,
        PY_INSPECT_MEMBERS_TABLE_KEY: t__inspect__run.member_row_count,
        PY_INSPECT_CLASS_MRO_TABLE_KEY: t__inspect__run.class_mro_row_count,
        PY_INSPECT_CLASS_ATTRS_TABLE_KEY: t__inspect__run.class_attr_row_count,
        PY_INSPECT_UNWRAP_TABLE_KEY: t__inspect__run.unwrap_row_count,
        PY_INSPECT_SIGNATURES_TABLE_KEY: t__inspect__run.signature_row_count,
        PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY: t__inspect__run.signature_param_row_count,
        PY_INSPECT_ANNOTATIONS_TABLE_KEY: t__inspect__run.annotation_row_count,
        PY_INSPECT_SOURCE_TABLE_KEY: t__inspect__run.source_row_count,
        PY_INSPECT_RUNTIME_STATE_TABLE_KEY: t__inspect__run.runtime_state_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def t__docstrings__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    py_frontend: PyFrontend,
) -> DocstringsToolOutput:
    """Execute docstring extraction on repository modules.

    Returns
    -------
    DocstringsToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return DocstringsToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=DOCSTRINGS_TARGET_NAME,
    )

    def _execute() -> DocstringsToolOutput:
        get_schema_service()
        step = DocstringsExtractStep(
            discovery=py_frontend.discovery,
            frontend=py_frontend,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return DocstringsToolOutput(
            result=extract_result.result,
            rows=extract_result.rows_reader,
            row_count=extract_result.row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_docstrings_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("Docstring extraction warning: %s", warning)
    return coerced


def t__docstrings__ingest(
    t__docstrings__run: DocstringsToolOutput,
) -> IngestStep[TabularByTable]:
    """Package docstrings rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest result with table frames.
    """
    result = t__docstrings__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Docstrings skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Docstrings extraction failed",
                warnings=result.warnings,
            )
        )

    payload = {DOCSTRINGS_TABLE_KEY: t__docstrings__run.rows}
    table_counts = {DOCSTRINGS_TABLE_KEY: t__docstrings__run.row_count}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


_INTENSIVE_SPEC = TargetSpecDescriptor(
    resources=TargetResources(tracker=True, modules=True),
    execution=CPU_INTENSIVE_EXECUTION,
)
_AST_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=AST_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(table_key=AST_NODES_TABLE_KEY, node_name="ast__node_rows"),
        TableOutputSpec(table_key=AST_METRICS_TABLE_KEY, node_name="ast__metric_rows"),
    ),
)
_CST_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=CST_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(TableOutputSpec(table_key=CST_NODES_TABLE_KEY, node_name="cst__node_rows"),),
)
_SYNTAX_INDEX_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=SYNTAX_INDEX_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(
            table_key=PARSE_MANIFEST_TABLE_KEY,
            node_name="syntax_index__parse_manifest_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_NODES_TABLE_KEY,
            node_name="syntax_index__node_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_EDGES_TABLE_KEY,
            node_name="syntax_index__edge_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_SPANS_TABLE_KEY,
            node_name="syntax_index__span_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_SCOPES_TABLE_KEY,
            node_name="syntax_index__scope_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_DEFS_TABLE_KEY,
            node_name="syntax_index__def_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_REFS_TABLE_KEY,
            node_name="syntax_index__ref_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_CALLS_TABLE_KEY,
            node_name="syntax_index__call_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_CALL_ARGS_TABLE_KEY,
            node_name="syntax_index__call_arg_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_FUNC_PARAMS_TABLE_KEY,
            node_name="syntax_index__func_param_rows",
        ),
        TableOutputSpec(
            table_key=SYNTAX_IMPORTS_TABLE_KEY,
            node_name="syntax_index__import_rows",
        ),
    ),
)
_SYMTABLE_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=SYMTABLE_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(
            table_key=PY_SYM_SCOPES_TABLE_KEY,
            node_name="symtable__scope_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_SYMBOLS_TABLE_KEY,
            node_name="symtable__symbol_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_SCOPE_EDGES_TABLE_KEY,
            node_name="symtable__scope_edge_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_NAMESPACE_EDGES_TABLE_KEY,
            node_name="symtable__namespace_edge_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY,
            node_name="symtable__function_partition_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_BINDINGS_TABLE_KEY,
            node_name="symtable__binding_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY,
            node_name="symtable__unresolved_binding_rows",
        ),
        TableOutputSpec(
            table_key=PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
            node_name="symtable__resolution_edge_rows",
        ),
    ),
)
_BYTECODE_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=BYTECODE_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(
            table_key=PY_COMPILER_META_TABLE_KEY,
            node_name="bytecode__compiler_meta_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_CODE_UNITS_TABLE_KEY,
            node_name="bytecode__code_unit_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_INSTRUCTIONS_TABLE_KEY,
            node_name="bytecode__instruction_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_EXCEPTION_TABLE_KEY,
            node_name="bytecode__exception_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_BLOCKS_TABLE_KEY,
            node_name="bytecode__block_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_CFG_EDGES_TABLE_KEY,
            node_name="bytecode__cfg_edge_rows",
        ),
        TableOutputSpec(
            table_key=PY_BC_DEFUSE_EVENTS_TABLE_KEY,
            node_name="bytecode__defuse_event_rows",
        ),
    ),
)
_INSPECT_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=INSPECT_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(
            table_key=PY_INSPECT_OBJECTS_TABLE_KEY,
            node_name="inspect__object_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_MEMBERS_TABLE_KEY,
            node_name="inspect__member_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_CLASS_MRO_TABLE_KEY,
            node_name="inspect__class_mro_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_CLASS_ATTRS_TABLE_KEY,
            node_name="inspect__class_attr_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_UNWRAP_TABLE_KEY,
            node_name="inspect__unwrap_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_SIGNATURES_TABLE_KEY,
            node_name="inspect__signature_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY,
            node_name="inspect__signature_param_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_ANNOTATIONS_TABLE_KEY,
            node_name="inspect__annotation_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_SOURCE_TABLE_KEY,
            node_name="inspect__source_rows",
        ),
        TableOutputSpec(
            table_key=PY_INSPECT_RUNTIME_STATE_TABLE_KEY,
            node_name="inspect__runtime_state_rows",
        ),
    ),
)
_DOCSTRINGS_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=DOCSTRINGS_TARGET_NAME,
    spec=TargetSpecDescriptor(),
    tables=(TableOutputSpec(table_key=DOCSTRINGS_TABLE_KEY, node_name="docstrings__rows"),),
)

_MODULE = sys.modules[__name__]

attach_tool_target_template(
    _MODULE,
    spec=_AST_TARGET_SPEC,
    run_fn=t__ast__run,
    ingest_fn=t__ast__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_CST_TARGET_SPEC,
    run_fn=t__cst__run,
    ingest_fn=t__cst__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_SYNTAX_INDEX_TARGET_SPEC,
    run_fn=t__syntax_index__run,
    ingest_fn=t__syntax_index__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_SYMTABLE_TARGET_SPEC,
    run_fn=t__symtable__run,
    ingest_fn=t__symtable__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_BYTECODE_TARGET_SPEC,
    run_fn=t__bytecode__run,
    ingest_fn=t__bytecode__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_INSPECT_TARGET_SPEC,
    run_fn=t__inspect__run,
    ingest_fn=t__inspect__ingest,
)
attach_tool_target_template(
    _MODULE,
    spec=_DOCSTRINGS_TARGET_SPEC,
    run_fn=t__docstrings__run,
    ingest_fn=t__docstrings__ingest,
)

t__ast = _MODULE.t__ast
t__cst = _MODULE.t__cst
t__syntax_index = _MODULE.t__syntax_index
t__symtable = _MODULE.t__symtable
t__bytecode = _MODULE.t__bytecode
t__inspect = _MODULE.t__inspect
t__docstrings = _MODULE.t__docstrings


__all__ = [
    "t__ast",
    "t__ast__ingest",
    "t__ast__run",
    "t__bytecode",
    "t__bytecode__ingest",
    "t__bytecode__run",
    "t__cst",
    "t__cst__ingest",
    "t__cst__run",
    "t__docstrings",
    "t__docstrings__ingest",
    "t__docstrings__run",
    "t__inspect",
    "t__inspect__ingest",
    "t__inspect__run",
    "t__symtable",
    "t__symtable__ingest",
    "t__symtable__run",
    "t__syntax_index",
    "t__syntax_index__ingest",
    "t__syntax_index__run",
]
