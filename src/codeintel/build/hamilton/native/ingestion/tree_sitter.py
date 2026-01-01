"""Tree-sitter ingestion target for query-pack captures."""

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
from codeintel.ingestion.compute.tree_sitter_index import TreeSitterIndexStep
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

TREE_SITTER_TARGET_NAME = "tree_sitter_index"
PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
TS_CAPTURES_TABLE_KEY = "core.ts_captures"
TS_PARSE_ERRORS_TABLE_KEY = "core.ts_parse_errors"

TREE_SITTER_TABLE_KEYS = (
    PARSE_MANIFEST_TABLE_KEY,
    TS_CAPTURES_TABLE_KEY,
    TS_PARSE_ERRORS_TABLE_KEY,
)


@dataclass(frozen=True)
class TreeSitterToolOutput(ToolStepOutput):
    """Tool step output for tree-sitter indexing."""

    parse_manifest_rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(PARSE_MANIFEST_TABLE_KEY)
    )
    captures_rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(TS_CAPTURES_TABLE_KEY)
    )
    parse_errors_rows: pl.LazyFrame = field(
        default_factory=lambda: empty_lazyframe_for_table(TS_PARSE_ERRORS_TABLE_KEY)
    )
    parse_manifest_row_count: int = 0
    captures_row_count: int = 0
    parse_errors_row_count: int = 0


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


def _coerce_tree_sitter_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> TreeSitterToolOutput:
    if isinstance(output, TreeSitterToolOutput):
        if warnings:
            return TreeSitterToolOutput(
                result=_merge_result_warnings(
                    output.result,
                    warnings,
                    error_message="Tree-sitter indexing failed",
                ),
                parse_manifest_rows=output.parse_manifest_rows,
                captures_rows=output.captures_rows,
                parse_errors_rows=output.parse_errors_rows,
                parse_manifest_row_count=output.parse_manifest_row_count,
                captures_row_count=output.captures_row_count,
                parse_errors_row_count=output.parse_errors_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Tree-sitter indexing failed",
    )
    return TreeSitterToolOutput(
        result=merged,
        parse_manifest_rows=empty_lazyframe_for_table(PARSE_MANIFEST_TABLE_KEY),
        captures_rows=empty_lazyframe_for_table(TS_CAPTURES_TABLE_KEY),
        parse_errors_rows=empty_lazyframe_for_table(TS_PARSE_ERRORS_TABLE_KEY),
        parse_manifest_row_count=0,
        captures_row_count=0,
        parse_errors_row_count=0,
    )


def t__tree_sitter_index__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TreeSitterToolOutput:
    """Execute tree-sitter indexing on repository modules.

    Returns
    -------
    TreeSitterToolOutput
        Tool output with parse manifests and capture rows.
    """
    failure, warnings = _module_inventory_precheck(t__modules, module_records)
    if failure is not None:
        return TreeSitterToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=TREE_SITTER_TARGET_NAME,
    )

    def _execute() -> TreeSitterToolOutput:
        get_schema_service()
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = TreeSitterIndexStep(discovery=discovery)
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        parse_manifest_frame = lazyframe_for_ingest_columns(
            PARSE_MANIFEST_TABLE_KEY,
            extract_result.parse_manifest_rows,
        )
        captures_frame = lazyframe_for_ingest_columns(
            TS_CAPTURES_TABLE_KEY,
            extract_result.captures_rows,
        )
        parse_errors_frame = lazyframe_for_ingest_columns(
            TS_PARSE_ERRORS_TABLE_KEY,
            extract_result.parse_errors_rows,
        )
        return TreeSitterToolOutput(
            result=extract_result.result,
            parse_manifest_rows=parse_manifest_frame,
            captures_rows=captures_frame,
            parse_errors_rows=parse_errors_frame,
            parse_manifest_row_count=extract_result.parse_manifest_row_count,
            captures_row_count=extract_result.captures_row_count,
            parse_errors_row_count=extract_result.parse_errors_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    coerced = _coerce_tree_sitter_output(output, warnings)
    for warning in coerced.result.warnings:
        log.warning("Tree-sitter indexing warning: %s", warning)
    return coerced


def t__tree_sitter_index__ingest(
    t__tree_sitter_index__run: TreeSitterToolOutput,
) -> IngestStep[TabularByTable]:
    """Package tree-sitter rows for table materialization.

    Returns
    -------
    IngestStep[TabularByTable]
        Ingest step for materializing table outputs.
    """
    result = t__tree_sitter_index__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Tree-sitter indexing skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Tree-sitter indexing failed",
                warnings=result.warnings,
            )
        )

    payload = {
        PARSE_MANIFEST_TABLE_KEY: t__tree_sitter_index__run.parse_manifest_rows,
        TS_CAPTURES_TABLE_KEY: t__tree_sitter_index__run.captures_rows,
        TS_PARSE_ERRORS_TABLE_KEY: t__tree_sitter_index__run.parse_errors_rows,
    }
    table_counts = {
        PARSE_MANIFEST_TABLE_KEY: t__tree_sitter_index__run.parse_manifest_row_count,
        TS_CAPTURES_TABLE_KEY: t__tree_sitter_index__run.captures_row_count,
        TS_PARSE_ERRORS_TABLE_KEY: t__tree_sitter_index__run.parse_errors_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


_INTENSIVE_SPEC = TargetSpecDescriptor(
    resources=TargetResources(tracker=True, modules=True),
    execution=CPU_INTENSIVE_EXECUTION,
)
_TREE_SITTER_TARGET_SPEC = ToolTargetSpec(
    domain="ingestion",
    target_name=TREE_SITTER_TARGET_NAME,
    spec=_INTENSIVE_SPEC,
    tables=(
        TableOutputSpec(
            table_key=PARSE_MANIFEST_TABLE_KEY,
            node_name="tree_sitter_index__parse_manifest_rows",
        ),
        TableOutputSpec(
            table_key=TS_CAPTURES_TABLE_KEY,
            node_name="tree_sitter_index__captures_rows",
        ),
        TableOutputSpec(
            table_key=TS_PARSE_ERRORS_TABLE_KEY,
            node_name="tree_sitter_index__parse_error_rows",
        ),
    ),
)

_MODULE = sys.modules[__name__]

attach_tool_target_template(
    _MODULE,
    spec=_TREE_SITTER_TARGET_SPEC,
    run_fn=t__tree_sitter_index__run,
    ingest_fn=t__tree_sitter_index__ingest,
)

t__tree_sitter_index = _MODULE.t__tree_sitter_index


__all__ = [
    "t__tree_sitter_index",
    "t__tree_sitter_index__ingest",
    "t__tree_sitter_index__run",
]
