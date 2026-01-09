"""Tree-sitter ingestion target for query-pack captures."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import pyarrow as pa

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.options.ingestion import TreeSitterIndexOptions
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
from codeintel.build.hamilton.transforms.ingestion_normalize import finalize_ingest_reader
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import table_to_reader, tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.tree_sitter_index import (
    TreeSitterIndexRunOptions,
    TreeSitterIndexStep,
)
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

TREE_SITTER_TARGET_NAME = "tree_sitter_index"
TS_PARSE_MANIFEST_TABLE_KEY = "core.ts_parse_manifest"
TS_CAPTURES_TABLE_KEY = "core.ts_captures"
TS_NODES_TABLE_KEY = "core.ts_nodes"
TS_EDGES_TABLE_KEY = "core.ts_edges"
TS_PARSE_ERRORS_TABLE_KEY = "core.ts_parse_errors"
TS_CHANGED_RANGES_TABLE_KEY = "core.ts_changed_ranges"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"
TS_LANGUAGE_METADATA_TABLE_KEY = "core.ts_language_metadata"
TS_PARSE_MANIFEST_OUTPUT_NAME = (
    f"{materialize_node(TS_PARSE_MANIFEST_TABLE_KEY)}__{TREE_SITTER_TARGET_NAME}"
)

TREE_SITTER_TABLE_KEYS = (
    TS_PARSE_MANIFEST_TABLE_KEY,
    TS_CAPTURES_TABLE_KEY,
    TS_NODES_TABLE_KEY,
    TS_EDGES_TABLE_KEY,
    TS_PARSE_ERRORS_TABLE_KEY,
    TS_CHANGED_RANGES_TABLE_KEY,
    TS_TOKENS_TABLE_KEY,
    TS_TRIVIA_TABLE_KEY,
    TS_LANGUAGE_METADATA_TABLE_KEY,
)


@dataclass(frozen=True)
class TreeSitterToolOutput(ToolStepOutput):
    """Tool step output for tree-sitter indexing."""

    parse_manifest_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_PARSE_MANIFEST_TABLE_KEY)
    )
    captures_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_CAPTURES_TABLE_KEY)
    )
    nodes_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_NODES_TABLE_KEY)
    )
    edges_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_EDGES_TABLE_KEY)
    )
    parse_errors_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_PARSE_ERRORS_TABLE_KEY)
    )
    changed_ranges_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_CHANGED_RANGES_TABLE_KEY)
    )
    tokens_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_TOKENS_TABLE_KEY)
    )
    trivia_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_TRIVIA_TABLE_KEY)
    )
    language_metadata_rows: InferableTabularInput = field(
        default_factory=lambda: empty_table_for_table(TS_LANGUAGE_METADATA_TABLE_KEY)
    )
    parse_manifest_row_count: int = 0
    captures_row_count: int = 0
    nodes_row_count: int = 0
    edges_row_count: int = 0
    parse_errors_row_count: int = 0
    changed_ranges_row_count: int = 0
    tokens_row_count: int = 0
    trivia_row_count: int = 0
    language_metadata_row_count: int = 0


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
                nodes_rows=output.nodes_rows,
                edges_rows=output.edges_rows,
                parse_errors_rows=output.parse_errors_rows,
                changed_ranges_rows=output.changed_ranges_rows,
                tokens_rows=output.tokens_rows,
                trivia_rows=output.trivia_rows,
                language_metadata_rows=output.language_metadata_rows,
                parse_manifest_row_count=output.parse_manifest_row_count,
                captures_row_count=output.captures_row_count,
                nodes_row_count=output.nodes_row_count,
                edges_row_count=output.edges_row_count,
                parse_errors_row_count=output.parse_errors_row_count,
                changed_ranges_row_count=output.changed_ranges_row_count,
                tokens_row_count=output.tokens_row_count,
                trivia_row_count=output.trivia_row_count,
                language_metadata_row_count=output.language_metadata_row_count,
            )
        return output

    merged = _merge_result_warnings(
        output.result,
        warnings,
        error_message="Tree-sitter indexing failed",
    )
    return TreeSitterToolOutput(
        result=merged,
        parse_manifest_rows=empty_table_for_table(TS_PARSE_MANIFEST_TABLE_KEY),
        captures_rows=empty_table_for_table(TS_CAPTURES_TABLE_KEY),
        nodes_rows=empty_table_for_table(TS_NODES_TABLE_KEY),
        edges_rows=empty_table_for_table(TS_EDGES_TABLE_KEY),
        parse_errors_rows=empty_table_for_table(TS_PARSE_ERRORS_TABLE_KEY),
        changed_ranges_rows=empty_table_for_table(TS_CHANGED_RANGES_TABLE_KEY),
        tokens_rows=empty_table_for_table(TS_TOKENS_TABLE_KEY),
        trivia_rows=empty_table_for_table(TS_TRIVIA_TABLE_KEY),
        language_metadata_rows=empty_table_for_table(TS_LANGUAGE_METADATA_TABLE_KEY),
        parse_manifest_row_count=0,
        captures_row_count=0,
        nodes_row_count=0,
        edges_row_count=0,
        parse_errors_row_count=0,
        changed_ranges_row_count=0,
        tokens_row_count=0,
        trivia_row_count=0,
        language_metadata_row_count=0,
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
        options = load_target_options(
            env,
            target_name=TREE_SITTER_TARGET_NAME,
            options_type=TreeSitterIndexOptions,
        )
        step = TreeSitterIndexStep(discovery=discovery)
        run_options = TreeSitterIndexRunOptions(
            emit_nodes_edges=options.emit_nodes_edges,
            emit_tokens=options.emit_tokens,
            emit_trivia=options.emit_trivia,
            emit_language_metadata=options.emit_language_metadata,
            enable_incremental=options.enable_incremental,
            match_limit=options.match_limit,
            allow_non_local_patterns=options.allow_non_local_patterns,
        )
        extract_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            options=run_options,
        )
        return TreeSitterToolOutput(
            result=extract_result.result,
            parse_manifest_rows=extract_result.parse_manifest_rows,
            captures_rows=extract_result.captures_rows,
            nodes_rows=extract_result.nodes_rows,
            edges_rows=extract_result.edges_rows,
            parse_errors_rows=extract_result.parse_errors_rows,
            changed_ranges_rows=extract_result.changed_ranges_rows,
            tokens_rows=extract_result.tokens_rows,
            trivia_rows=extract_result.trivia_rows,
            language_metadata_rows=extract_result.language_metadata_rows,
            parse_manifest_row_count=extract_result.parse_manifest_row_count,
            captures_row_count=extract_result.captures_row_count,
            nodes_row_count=extract_result.nodes_row_count,
            edges_row_count=extract_result.edges_row_count,
            parse_errors_row_count=extract_result.parse_errors_row_count,
            changed_ranges_row_count=extract_result.changed_ranges_row_count,
            tokens_row_count=extract_result.tokens_row_count,
            trivia_row_count=extract_result.trivia_row_count,
            language_metadata_row_count=extract_result.language_metadata_row_count,
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

    tolerant_keys = {TS_PARSE_ERRORS_TABLE_KEY, TS_CHANGED_RANGES_TABLE_KEY}

    def _finalize_table(table_key: str, value: InferableTabularInput) -> pa.Table:
        table = tabular_to_scoped_table(
            value,
            columns=None,
            scope=None,
            require_scope_columns=False,
        )
        mode = "tolerant" if table_key in tolerant_keys else None
        reader = table_to_reader(table, batch_size=None)
        return finalize_ingest_reader(
            table_key,
            reader,
            target_name=TREE_SITTER_TARGET_NAME,
            mode=mode,
        )

    parse_manifest_table = _finalize_table(
        TS_PARSE_MANIFEST_TABLE_KEY,
        t__tree_sitter_index__run.parse_manifest_rows,
    )
    captures_table = _finalize_table(
        TS_CAPTURES_TABLE_KEY,
        t__tree_sitter_index__run.captures_rows,
    )
    nodes_table = _finalize_table(TS_NODES_TABLE_KEY, t__tree_sitter_index__run.nodes_rows)
    edges_table = _finalize_table(TS_EDGES_TABLE_KEY, t__tree_sitter_index__run.edges_rows)
    parse_errors_table = _finalize_table(
        TS_PARSE_ERRORS_TABLE_KEY,
        t__tree_sitter_index__run.parse_errors_rows,
    )
    changed_ranges_table = _finalize_table(
        TS_CHANGED_RANGES_TABLE_KEY,
        t__tree_sitter_index__run.changed_ranges_rows,
    )
    tokens_table = _finalize_table(TS_TOKENS_TABLE_KEY, t__tree_sitter_index__run.tokens_rows)
    trivia_table = _finalize_table(TS_TRIVIA_TABLE_KEY, t__tree_sitter_index__run.trivia_rows)
    language_metadata_table = _finalize_table(
        TS_LANGUAGE_METADATA_TABLE_KEY,
        t__tree_sitter_index__run.language_metadata_rows,
    )

    payload = {
        TS_PARSE_MANIFEST_TABLE_KEY: parse_manifest_table,
        TS_CAPTURES_TABLE_KEY: captures_table,
        TS_NODES_TABLE_KEY: nodes_table,
        TS_EDGES_TABLE_KEY: edges_table,
        TS_PARSE_ERRORS_TABLE_KEY: parse_errors_table,
        TS_CHANGED_RANGES_TABLE_KEY: changed_ranges_table,
        TS_TOKENS_TABLE_KEY: tokens_table,
        TS_TRIVIA_TABLE_KEY: trivia_table,
        TS_LANGUAGE_METADATA_TABLE_KEY: language_metadata_table,
    }
    table_counts = {
        TS_PARSE_MANIFEST_TABLE_KEY: parse_manifest_table.num_rows,
        TS_CAPTURES_TABLE_KEY: captures_table.num_rows,
        TS_NODES_TABLE_KEY: nodes_table.num_rows,
        TS_EDGES_TABLE_KEY: edges_table.num_rows,
        TS_PARSE_ERRORS_TABLE_KEY: parse_errors_table.num_rows,
        TS_CHANGED_RANGES_TABLE_KEY: changed_ranges_table.num_rows,
        TS_TOKENS_TABLE_KEY: tokens_table.num_rows,
        TS_TRIVIA_TABLE_KEY: trivia_table.num_rows,
        TS_LANGUAGE_METADATA_TABLE_KEY: language_metadata_table.num_rows,
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
            table_key=TS_PARSE_MANIFEST_TABLE_KEY,
            node_name="tree_sitter_index__parse_manifest_rows",
            output_name=TS_PARSE_MANIFEST_OUTPUT_NAME,
        ),
        TableOutputSpec(
            table_key=TS_CAPTURES_TABLE_KEY,
            node_name="tree_sitter_index__captures_rows",
        ),
        TableOutputSpec(
            table_key=TS_NODES_TABLE_KEY,
            node_name="tree_sitter_index__nodes_rows",
        ),
        TableOutputSpec(
            table_key=TS_EDGES_TABLE_KEY,
            node_name="tree_sitter_index__edges_rows",
        ),
        TableOutputSpec(
            table_key=TS_PARSE_ERRORS_TABLE_KEY,
            node_name="tree_sitter_index__parse_error_rows",
        ),
        TableOutputSpec(
            table_key=TS_CHANGED_RANGES_TABLE_KEY,
            node_name="tree_sitter_index__changed_ranges_rows",
        ),
        TableOutputSpec(
            table_key=TS_TOKENS_TABLE_KEY,
            node_name="tree_sitter_index__tokens_rows",
        ),
        TableOutputSpec(
            table_key=TS_TRIVIA_TABLE_KEY,
            node_name="tree_sitter_index__trivia_rows",
        ),
        TableOutputSpec(
            table_key=TS_LANGUAGE_METADATA_TABLE_KEY,
            node_name="tree_sitter_index__language_metadata_rows",
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
