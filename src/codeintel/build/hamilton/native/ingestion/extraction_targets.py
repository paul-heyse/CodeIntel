"""Consolidated ingestion targets for AST/CST/docstring extraction.

This module replaces the per-target files for:
- ``ast``: stdlib AST extraction
- ``cst``: LibCST extraction
- ``docstrings``: docstring extraction/parsing

The targets share a common pattern:
1) Load module paths from the current snapshot
2) Convert paths into ``ModuleRecord``s
3) Execute pure ingestion compute-steps that return columnar rows
4) Materialize columnar rows via Hamilton materializers and emit ``TargetRunRecord``
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import polars as pl

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.ingestion.frame_utils import (
    empty_lazyframe_for_table,
    lazyframe_for_ingest_columns,
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
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.ast_extract import AstExtractStep
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    ModuleRecord,
)

AST_TARGET_NAME = "ast"
CST_TARGET_NAME = "cst"
DOCSTRINGS_TARGET_NAME = "docstrings"

AST_NODES_TABLE_KEY = "core.ast_nodes"
AST_METRICS_TABLE_KEY = "core.ast_metrics"
AST_TABLE_KEYS = (AST_NODES_TABLE_KEY, AST_METRICS_TABLE_KEY)

CST_NODES_TABLE_KEY = "core.cst_nodes"
CST_TABLE_KEYS = (CST_NODES_TABLE_KEY,)

DOCSTRINGS_TABLE_KEY = "core.docstrings"
DOCSTRINGS_TABLE_KEYS = (DOCSTRINGS_TABLE_KEY,)


@dataclass(frozen=True)
class DocstringsToolOutput(ToolStepOutput):
    """Tool step output for docstrings extraction."""

    rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(DOCSTRINGS_TABLE_KEY)
    )
    row_count: int = 0


@dataclass(frozen=True)
class AstToolOutput(ToolStepOutput):
    """Tool step output for AST extraction."""

    ast_rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(AST_NODES_TABLE_KEY)
    )
    metric_rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(AST_METRICS_TABLE_KEY)
    )
    ast_row_count: int = 0
    metric_row_count: int = 0


@dataclass(frozen=True)
class CstToolOutput(ToolStepOutput):
    """Tool step output for CST extraction."""

    rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(CST_NODES_TABLE_KEY)
    )
    row_count: int = 0


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
        ast_rows=empty_lazyframe_for_table(AST_NODES_TABLE_KEY),
        metric_rows=empty_lazyframe_for_table(AST_METRICS_TABLE_KEY),
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
        rows=empty_lazyframe_for_table(CST_NODES_TABLE_KEY),
        row_count=0,
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
        rows=empty_lazyframe_for_table(DOCSTRINGS_TABLE_KEY),
        row_count=0,
    )


def t__ast__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
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
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = AstExtractStep(discovery=discovery)
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        ast_frame = lazyframe_for_ingest_columns(AST_NODES_TABLE_KEY, extract_result.ast_rows)
        metric_frame = lazyframe_for_ingest_columns(
            AST_METRICS_TABLE_KEY, extract_result.metric_rows
        )
        return AstToolOutput(
            result=extract_result.result,
            ast_rows=ast_frame,
            metric_rows=metric_frame,
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


def t__cst__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CstToolOutput:
    """Execute CST extraction on repository modules.

    Returns
    -------
    CstToolOutput
        Tool output with row payloads and execution status.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return CstToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=CST_TARGET_NAME,
    )

    def _execute() -> CstToolOutput:
        get_schema_service()
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = CstExtractStep(discovery=discovery)
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        frame = lazyframe_for_ingest_columns(CST_NODES_TABLE_KEY, extract_result.rows)
        return CstToolOutput(
            result=extract_result.result,
            rows=frame,
            row_count=extract_result.row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_cst_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("CST extraction warning: %s", warning)
    return coerced


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


def t__docstrings__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
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
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = DocstringsExtractStep(discovery=discovery)
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        frame = lazyframe_for_ingest_columns(DOCSTRINGS_TABLE_KEY, extract_result.rows)
        return DocstringsToolOutput(
            result=extract_result.result,
            rows=frame,
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
    spec=_DOCSTRINGS_TARGET_SPEC,
    run_fn=t__docstrings__run,
    ingest_fn=t__docstrings__ingest,
)

t__ast = _MODULE.t__ast
t__cst = _MODULE.t__cst
t__docstrings = _MODULE.t__docstrings


__all__ = [
    "t__ast",
    "t__ast__ingest",
    "t__ast__run",
    "t__cst",
    "t__cst__ingest",
    "t__cst__run",
    "t__docstrings",
    "t__docstrings__ingest",
    "t__docstrings__run",
]
