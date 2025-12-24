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

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.ast_extract import AstExtractResult, AstExtractStep
from codeintel.ingestion.compute.cst_extract import CstExtractResult, CstExtractStep
from codeintel.ingestion.compute.docstrings_extract import (
    DocstringsExtractResult,
    DocstringsExtractStep,
)
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


def _should_skip_extract(env: BuildEnv, graph: TargetGraph, target_name: str) -> bool:
    target = graph.get(target_name)
    if target is None:
        return False
    options_hash = options_hash_for_target(env, target_name)
    hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
    input_hash = compute_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        settings=env.settings,
        options=hash_options,
    )
    return should_skip_native_target(env, target, input_hash, options_hash=options_hash)


@tag_tool(domain="ingestion", target=AST_TARGET_NAME)
def t__ast__extract(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> AstExtractResult:
    """Execute AST extraction on repository modules.

    Returns
    -------
    AstExtractResult
        Result bundle with row tuples and execution status.
    """
    if t__modules.status != "succeeded":
        return AstExtractResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_extract(env, graph, AST_TARGET_NAME):
        return AstExtractResult(result=ExecutionResult.skip("AST extraction skipped"))

    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    step = AstExtractStep(discovery=discovery)
    return step.execute(
        module_records,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(AST_NODES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(AST_TARGET_NAME),
    table_key=value(AST_NODES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(AST_NODES_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=AST_TARGET_NAME, target_="ast__node_rows")
def ast__node_rows(
    t__ast__extract: AstExtractResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.ast_nodes.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.ast_nodes, or None when skipped/failed.
    """
    if t__ast__extract.result.skipped or not t__ast__extract.result.success:
        return None
    return t__ast__extract.ast_rows


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(AST_METRICS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(AST_TARGET_NAME),
    table_key=value(AST_METRICS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(AST_METRICS_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=AST_TARGET_NAME, target_="ast__metric_rows")
def ast__metric_rows(
    t__ast__extract: AstExtractResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.ast_metrics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.ast_metrics, or None when skipped/failed.
    """
    if t__ast__extract.result.skipped or not t__ast__extract.result.success:
        return None
    return t__ast__extract.metric_rows


@codeintel_target(
    domain="ingestion",
    target=AST_TARGET_NAME,
    spec=TargetSpecDescriptor(
        resources=TargetResources(tracker=True, modules=True),
        execution=CPU_INTENSIVE_EXECUTION,
    ),
)
def t__ast(
    env: BuildEnv,
    graph: TargetGraph,
    t__ast__extract: AstExtractResult,
    m__core__ast_nodes: MaterializationMetadata,
    m__core__ast_metrics: MaterializationMetadata,
) -> TargetRunRecord:
    """Python AST extraction and metrics.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if not t__ast__extract.result.success and not t__ast__extract.result.skipped:
        executor = NativeTargetExecutor.for_target(env, graph, AST_TARGET_NAME)
        return executor.fail(RuntimeError(t__ast__extract.result.error or "AST extraction failed"))

    for warning in t__ast__extract.result.warnings:
        log.warning("AST extraction warning: %s", warning)

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=AST_TARGET_NAME,
        materializations={
            AST_NODES_TABLE_KEY: m__core__ast_nodes,
            AST_METRICS_TABLE_KEY: m__core__ast_metrics,
        },
    )


@tag_tool(domain="ingestion", target=CST_TARGET_NAME)
def t__cst__extract(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CstExtractResult:
    """Execute CST extraction on repository modules.

    Returns
    -------
    CstExtractResult
        Result bundle with row tuples and execution status.
    """
    if t__modules.status != "succeeded":
        return CstExtractResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_extract(env, graph, CST_TARGET_NAME):
        return CstExtractResult(result=ExecutionResult.skip("CST extraction skipped"))

    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    step = CstExtractStep(discovery=discovery)
    return step.execute(
        module_records,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(CST_NODES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(CST_TARGET_NAME),
    table_key=value(CST_NODES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(CST_NODES_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=CST_TARGET_NAME, target_="cst__node_rows")
def cst__node_rows(
    t__cst__extract: CstExtractResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.cst_nodes.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.cst_nodes, or None when skipped/failed.
    """
    if t__cst__extract.result.skipped or not t__cst__extract.result.success:
        return None
    return t__cst__extract.rows


@codeintel_target(domain="ingestion", target=CST_TARGET_NAME)
def t__cst(
    env: BuildEnv,
    graph: TargetGraph,
    t__cst__extract: CstExtractResult,
    m__core__cst_nodes: MaterializationMetadata,
) -> TargetRunRecord:
    """Concrete syntax tree extraction.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if not t__cst__extract.result.success and not t__cst__extract.result.skipped:
        executor = NativeTargetExecutor.for_target(env, graph, CST_TARGET_NAME)
        return executor.fail(RuntimeError(t__cst__extract.result.error or "CST extraction failed"))

    for warning in t__cst__extract.result.warnings:
        log.warning("CST extraction warning: %s", warning)

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=CST_TARGET_NAME,
        materializations={
            CST_NODES_TABLE_KEY: m__core__cst_nodes,
        },
    )


@tag_tool(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings__extract(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> DocstringsExtractResult:
    """Execute docstring extraction on repository modules.

    Returns
    -------
    DocstringsExtractResult
        Result bundle with row tuples and execution status.
    """
    if t__modules.status != "succeeded":
        return DocstringsExtractResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_extract(env, graph, DOCSTRINGS_TARGET_NAME):
        return DocstringsExtractResult(result=ExecutionResult.skip("Docstrings skipped"))

    get_schema_service()
    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    step = DocstringsExtractStep(discovery=discovery)
    return step.execute(
        module_records,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DOCSTRINGS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DOCSTRINGS_TARGET_NAME),
    table_key=value(DOCSTRINGS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DOCSTRINGS_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=DOCSTRINGS_TARGET_NAME, target_="docstrings__rows")
def docstrings__rows(
    t__docstrings__extract: DocstringsExtractResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.docstrings.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.docstrings, or None when skipped/failed.
    """
    if t__docstrings__extract.result.skipped or not t__docstrings__extract.result.success:
        return None
    return t__docstrings__extract.rows


@codeintel_target(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings(
    env: BuildEnv,
    graph: TargetGraph,
    t__docstrings__extract: DocstringsExtractResult,
    m__core__docstrings: MaterializationMetadata,
) -> TargetRunRecord:
    """Docstring extraction and parsing.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if not t__docstrings__extract.result.success and not t__docstrings__extract.result.skipped:
        executor = NativeTargetExecutor.for_target(env, graph, DOCSTRINGS_TARGET_NAME)
        return executor.fail(
            RuntimeError(t__docstrings__extract.result.error or "Docstring extraction failed")
        )

    for warning in t__docstrings__extract.result.warnings:
        log.warning("Docstring extraction warning: %s", warning)

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=DOCSTRINGS_TARGET_NAME,
        materializations={
            DOCSTRINGS_TABLE_KEY: m__core__docstrings,
        },
    )


__all__ = [
    "t__ast",
    "t__ast__extract",
    "t__cst",
    "t__cst__extract",
    "t__docstrings",
    "t__docstrings__extract",
]
