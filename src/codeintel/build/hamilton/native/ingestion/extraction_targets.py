"""Consolidated ingestion targets for AST/CST/docstring extraction.

This module replaces the per-target files for:
- ``ast``: stdlib AST extraction
- ``cst``: LibCST extraction
- ``docstrings``: docstring extraction/parsing

The targets share a common pattern:
1) Load module paths from the current snapshot
2) Convert paths into ``ModuleRecord``s
3) Execute pure ingestion compute-steps that return row tuples
4) Materialize rows via Hamilton materializers and emit ``TargetRunRecord``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.run_records import TargetRunRecord, options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import InputHashOptions
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.ast_extract import AstExtractStep
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    MaterializationMetadata,
    TargetGraph,
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

AST_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=AST_TARGET_NAME,
    hash_options_node="ast__hash_options",
)
CST_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=CST_TARGET_NAME,
    hash_options_node="cst__hash_options",
)
DOCSTRINGS_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=DOCSTRINGS_TARGET_NAME,
    hash_options_node="docstrings__hash_options",
)


@dataclass(frozen=True)
class DocstringsToolOutput(ToolStepOutput):
    """Tool step output for docstrings extraction."""

    rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class AstToolOutput(ToolStepOutput):
    """Tool step output for AST extraction."""

    ast_rows: tuple[tuple[object, ...], ...] = ()
    metric_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class CstToolOutput(ToolStepOutput):
    """Tool step output for CST extraction."""

    rows: tuple[tuple[object, ...], ...] = ()


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
    skip_reason: str,
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
                    skip_reason="AST extraction skipped",
                    error_message="AST extraction failed",
                ),
                ast_rows=output.ast_rows,
                metric_rows=output.metric_rows,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        skip_reason="AST extraction skipped",
        error_message="AST extraction failed",
    )
    return AstToolOutput(result=merged, ast_rows=(), metric_rows=())


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
                    skip_reason="CST extraction skipped",
                    error_message="CST extraction failed",
                ),
                rows=output.rows,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        skip_reason="CST extraction skipped",
        error_message="CST extraction failed",
    )
    return CstToolOutput(result=merged, rows=())


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
                    skip_reason="Docstrings skipped",
                    error_message="Docstrings extraction failed",
                ),
                rows=output.rows,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        skip_reason="Docstrings skipped",
        error_message="Docstrings extraction failed",
    )
    return DocstringsToolOutput(result=merged, rows=())


@tag_helper(domain="ingestion", target=AST_TARGET_NAME)
def ast__hash_options(
    env: BuildEnv,
    modules__hash_options: InputHashOptions,
) -> InputHashOptions:
    """Build input hash options for AST extraction.

    Returns
    -------
    InputHashOptions
        Hash inputs used to gate AST execution.
    """
    options_hash = options_hash_for_target(env, AST_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=modules__hash_options.file_state_hash,
    )


@tag_tool(domain="ingestion", target=AST_TARGET_NAME)
def t__ast__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    ast__hash_options: InputHashOptions,
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
        graph=graph,
        target_name=AST_TARGET_NAME,
        hash_options=ast__hash_options,
        skip_reason="AST extraction skipped",
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
        return AstToolOutput(
            result=extract_result.result,
            ast_rows=extract_result.ast_rows,
            metric_rows=extract_result.metric_rows,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_ast_output(output, warnings)


@tag_compute(domain="ingestion", target=AST_TARGET_NAME)
def t__ast__ingest(
    t__ast__run: AstToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package AST rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
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
        AST_NODES_TABLE_KEY: len(t__ast__run.ast_rows),
        AST_METRICS_TABLE_KEY: len(t__ast__run.metric_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=AST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=AST_NODES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=AST_TARGET_NAME, target_="ast__node_rows")
def ast__node_rows(
    t__ast__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.ast_nodes.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.ast_nodes, or None when skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__ast__ingest.result.skipped or not t__ast__ingest.result.success:
        return None

    payload = t__ast__ingest.payload
    if payload is None:
        msg = "Missing AST ingest payload"
        raise ValueError(msg)
    rows = payload.get(AST_NODES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {AST_NODES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=AST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=AST_METRICS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=AST_TARGET_NAME, target_="ast__metric_rows")
def ast__metric_rows(
    t__ast__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.ast_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.ast_metrics, or None when skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__ast__ingest.result.skipped or not t__ast__ingest.result.success:
        return None

    payload = t__ast__ingest.payload
    if payload is None:
        msg = "Missing AST ingest payload"
        raise ValueError(msg)
    rows = payload.get(AST_METRICS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {AST_METRICS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=AST_TARGET_NAME)
def ast__table_materializations(
    m__core__ast_nodes: MaterializationMetadata,
    m__core__ast_metrics: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect AST materialization metadata.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of table keys to materialization metadata.
    """
    return {
        AST_NODES_TABLE_KEY: m__core__ast_nodes,
        AST_METRICS_TABLE_KEY: m__core__ast_metrics,
    }


@tag_helper(domain="ingestion", target=AST_TARGET_NAME)
def ast__finalize_context(
    env: BuildEnv,
    graph: TargetGraph,
    ast__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for the AST target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for AST extraction.
    """
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=AST_TARGET_NAME,
        hash_options=ast__hash_options,
    )


@codeintel_target(
    domain="ingestion",
    target=AST_TARGET_NAME,
    spec=TargetSpecDescriptor(
        resources=TargetResources(tracker=True, modules=True),
        execution=CPU_INTENSIVE_EXECUTION,
    ),
)
def t__ast(
    ast__finalize_context: ToolFinalizeContext,
    t__ast__run: AstToolOutput,
    t__ast__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    ast__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Python AST extraction and metrics.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    for warning in t__ast__run.result.warnings:
        log.warning("AST extraction warning: %s", warning)

    return finalize_target_from_materializations(
        context=ast__finalize_context,
        tool_step=t__ast__run,
        ingest_step=t__ast__ingest,
        artifact_materializations=None,
        table_materializations=ast__table_materializations,
    )


@tag_helper(domain="ingestion", target=CST_TARGET_NAME)
def cst__hash_options(
    env: BuildEnv,
    modules__hash_options: InputHashOptions,
) -> InputHashOptions:
    """Build input hash options for CST extraction.

    Returns
    -------
    InputHashOptions
        Hash inputs used to gate CST execution.
    """
    options_hash = options_hash_for_target(env, CST_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=modules__hash_options.file_state_hash,
    )


@tag_tool(domain="ingestion", target=CST_TARGET_NAME)
def t__cst__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    cst__hash_options: InputHashOptions,
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
        graph=graph,
        target_name=CST_TARGET_NAME,
        hash_options=cst__hash_options,
        skip_reason="CST extraction skipped",
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
        return CstToolOutput(result=extract_result.result, rows=extract_result.rows)

    output = run_tool_step(context=context, run=_execute)
    return _coerce_cst_output(output, warnings)


@tag_compute(domain="ingestion", target=CST_TARGET_NAME)
def t__cst__ingest(
    t__cst__run: CstToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package CST rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
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
    table_counts = {CST_NODES_TABLE_KEY: len(t__cst__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=CST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CST_NODES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=CST_TARGET_NAME, target_="cst__node_rows")
def cst__node_rows(
    t__cst__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.cst_nodes.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.cst_nodes, or None when skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__cst__ingest.result.skipped or not t__cst__ingest.result.success:
        return None

    payload = t__cst__ingest.payload
    if payload is None:
        msg = "Missing CST ingest payload"
        raise ValueError(msg)
    rows = payload.get(CST_NODES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CST_NODES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=CST_TARGET_NAME)
def cst__table_materializations(
    m__core__cst_nodes: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect CST materialization metadata.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of table keys to materialization metadata.
    """
    return {CST_NODES_TABLE_KEY: m__core__cst_nodes}


@tag_helper(domain="ingestion", target=CST_TARGET_NAME)
def cst__finalize_context(
    env: BuildEnv,
    graph: TargetGraph,
    cst__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for the CST target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for CST extraction.
    """
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=CST_TARGET_NAME,
        hash_options=cst__hash_options,
    )


@codeintel_target(domain="ingestion", target=CST_TARGET_NAME)
def t__cst(
    cst__finalize_context: ToolFinalizeContext,
    t__cst__run: CstToolOutput,
    t__cst__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    cst__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Concrete syntax tree extraction.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    for warning in t__cst__run.result.warnings:
        log.warning("CST extraction warning: %s", warning)

    return finalize_target_from_materializations(
        context=cst__finalize_context,
        tool_step=t__cst__run,
        ingest_step=t__cst__ingest,
        artifact_materializations=None,
        table_materializations=cst__table_materializations,
    )


@tag_helper(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def docstrings__hash_options(
    env: BuildEnv,
    modules__hash_options: InputHashOptions,
) -> InputHashOptions:
    """Build input hash options for docstrings extraction.

    Returns
    -------
    InputHashOptions
        Hash inputs used to gate docstrings execution.
    """
    options_hash = options_hash_for_target(env, DOCSTRINGS_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=modules__hash_options.file_state_hash,
    )


@tag_tool(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
    docstrings__hash_options: InputHashOptions,
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
        graph=graph,
        target_name=DOCSTRINGS_TARGET_NAME,
        hash_options=docstrings__hash_options,
        skip_reason="Docstrings skipped",
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
        return DocstringsToolOutput(result=extract_result.result, rows=extract_result.rows)

    output = run_tool_step(context=context, run=_execute)
    return _coerce_docstrings_output(output, warnings)


@tag_compute(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings__ingest(
    t__docstrings__run: DocstringsToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package docstrings rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
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
    table_counts = {DOCSTRINGS_TABLE_KEY: len(t__docstrings__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=DOCSTRINGS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=DOCSTRINGS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=DOCSTRINGS_TARGET_NAME, target_="docstrings__rows")
def docstrings__rows(
    t__docstrings__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.docstrings.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.docstrings, or None when skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__docstrings__ingest.result.skipped or not t__docstrings__ingest.result.success:
        return None

    payload = t__docstrings__ingest.payload
    if payload is None:
        msg = "Missing docstrings ingest payload"
        raise ValueError(msg)
    rows = payload.get(DOCSTRINGS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {DOCSTRINGS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def docstrings__table_materializations(
    m__core__docstrings: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect docstrings materialization metadata.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of table keys to materialization metadata.
    """
    return {DOCSTRINGS_TABLE_KEY: m__core__docstrings}


@tag_helper(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def docstrings__finalize_context(
    env: BuildEnv,
    graph: TargetGraph,
    docstrings__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for the docstrings target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for docstrings.
    """
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=DOCSTRINGS_TARGET_NAME,
        hash_options=docstrings__hash_options,
    )


@codeintel_target(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings(
    docstrings__finalize_context: ToolFinalizeContext,
    t__docstrings__run: DocstringsToolOutput,
    t__docstrings__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    docstrings__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Docstring extraction and parsing.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    for warning in t__docstrings__run.result.warnings:
        log.warning("Docstring extraction warning: %s", warning)

    return finalize_target_from_materializations(
        context=docstrings__finalize_context,
        tool_step=t__docstrings__run,
        ingest_step=t__docstrings__ingest,
        artifact_materializations=None,
        table_materializations=docstrings__table_materializations,
    )


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
