"""Consolidated ingestion targets for AST/CST/docstring extraction.

This module replaces the per-target files for:
- ``ast``: stdlib AST extraction
- ``cst``: LibCST extraction
- ``docstrings``: docstring extraction/parsing

The targets share a common pattern:
1) Load module paths from the current snapshot
2) Convert paths into ``ModuleRecord``s
3) Execute an ingestion compute-step that writes tables
4) Return a ``TargetRunRecord`` via ``NativeTargetExecutor``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.table_counts import normalize_table_counts
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_materialize, tag_tool
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import AstExtractStep, CstExtractStep, DocstringsExtractStep
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ModuleRecord)

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

TARGET_SPECS = (
    make_output_target(
        name=AST_TARGET_NAME,
        module="ingestion",
        description="Python AST extraction and metrics.",
        options=TargetSpecOptions(
            table_keys=AST_TABLE_KEYS,
            resources=TargetResources(tracker=True, modules=True),
            execution=CPU_INTENSIVE_EXECUTION,
        ),
    ),
    make_output_target(
        name=CST_TARGET_NAME,
        module="ingestion",
        description="Concrete syntax tree extraction.",
        options=TargetSpecOptions(table_keys=CST_TABLE_KEYS),
    ),
    make_output_target(
        name=DOCSTRINGS_TARGET_NAME,
        module="ingestion",
        description="Docstring extraction and parsing.",
        options=TargetSpecOptions(table_keys=DOCSTRINGS_TABLE_KEYS),
    ),
)


@dataclass(frozen=True)
class AstExtractResult:
    """Result from AST extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    table_counts
        Row counts per produced table.
    errors
        List of extraction errors (non-fatal).
    error
        Fatal error message if extraction failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class CstExtractResult:
    """Result from CST extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    table_counts
        Row counts per produced table.
    errors
        List of extraction errors (non-fatal).
    error
        Fatal error message if extraction failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class DocstringsExtractResult:
    """Result from docstring extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    table_counts
        Row counts per produced table.
    errors
        List of extraction errors (non-fatal).
    error
        Fatal error message if extraction failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@tag_tool(domain="ingestion", target=AST_TARGET_NAME)
def t__ast__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> AstExtractResult:
    """Execute AST extraction on repository modules.

    Returns
    -------
    AstExtractResult
        Extraction status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return AstExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        if not module_records:
            log.info("No modules found for AST extraction")
            return AstExtractResult(
                success=True,
                table_counts=normalize_table_counts(AST_TABLE_KEYS, None),
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = AstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return AstExtractResult(
            success=True,
            table_counts=normalize_table_counts(
                AST_TABLE_KEYS,
                dict(result.table_counts) if result.table_counts else None,
            ),
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("AST extraction failed")
        return AstExtractResult(success=False, error=str(exc))


@tag_materialize(domain="ingestion", target=AST_TARGET_NAME)
def t__ast(
    env: BuildEnv,
    graph: TargetGraph,
    t__ast__extract: AstExtractResult,
) -> TargetRunRecord:
    """Materialize AST target with validation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, AST_TARGET_NAME)
    if executor.should_skip():
        return executor.skip()
    if not t__ast__extract.success:
        return executor.fail(RuntimeError(t__ast__extract.error or "AST extraction failed"))

    for warning in t__ast__extract.errors:
        log.warning("AST extraction warning: %s", warning)

    return executor.execute(
        lambda: normalize_table_counts(
            AST_TABLE_KEYS,
            dict(t__ast__extract.table_counts),
        )
    )


@tag_tool(domain="ingestion", target=CST_TARGET_NAME)
def t__cst__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CstExtractResult:
    """Execute CST extraction on repository modules.

    Returns
    -------
    CstExtractResult
        Extraction status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return CstExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        if not module_records:
            log.info("No modules found for CST extraction")
            return CstExtractResult(
                success=True,
                table_counts=normalize_table_counts(CST_TABLE_KEYS, None),
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = CstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return CstExtractResult(
            success=True,
            table_counts=normalize_table_counts(
                CST_TABLE_KEYS,
                dict(result.table_counts) if result.table_counts else None,
            ),
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("CST extraction failed")
        return CstExtractResult(success=False, error=str(exc))


@tag_materialize(domain="ingestion", target=CST_TARGET_NAME)
def t__cst(
    env: BuildEnv,
    graph: TargetGraph,
    t__cst__extract: CstExtractResult,
) -> TargetRunRecord:
    """Materialize CST target with validation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, CST_TARGET_NAME)
    if executor.should_skip():
        return executor.skip()
    if not t__cst__extract.success:
        return executor.fail(RuntimeError(t__cst__extract.error or "CST extraction failed"))

    for warning in t__cst__extract.errors:
        log.warning("CST extraction warning: %s", warning)

    return executor.execute(
        lambda: normalize_table_counts(
            CST_TABLE_KEYS,
            dict(t__cst__extract.table_counts),
        )
    )


@tag_tool(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> DocstringsExtractResult:
    """Execute docstring extraction on repository modules.

    Returns
    -------
    DocstringsExtractResult
        Extraction status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return DocstringsExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        if not module_records:
            log.info("No modules found for docstring extraction")
            return DocstringsExtractResult(
                success=True,
                table_counts=normalize_table_counts(DOCSTRINGS_TABLE_KEYS, None),
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = DocstringsExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return DocstringsExtractResult(
            success=True,
            table_counts=normalize_table_counts(
                DOCSTRINGS_TABLE_KEYS,
                dict(result.table_counts) if result.table_counts else None,
            ),
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("Docstring extraction failed")
        return DocstringsExtractResult(success=False, error=str(exc))


@tag_materialize(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
def t__docstrings(
    env: BuildEnv,
    graph: TargetGraph,
    t__docstrings__extract: DocstringsExtractResult,
) -> TargetRunRecord:
    """Materialize docstrings target with validation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, DOCSTRINGS_TARGET_NAME)
    if executor.should_skip():
        return executor.skip()
    if not t__docstrings__extract.success:
        return executor.fail(
            RuntimeError(t__docstrings__extract.error or "Docstring extraction failed")
        )

    for warning in t__docstrings__extract.errors:
        log.warning("Docstring extraction warning: %s", warning)

    return executor.execute(
        lambda: normalize_table_counts(
            DOCSTRINGS_TABLE_KEYS,
            dict(t__docstrings__extract.table_counts),
        )
    )


__all__ = [
    "AstExtractResult",
    "CstExtractResult",
    "DocstringsExtractResult",
    "t__ast",
    "t__ast__extract",
    "t__cst",
    "t__cst__extract",
    "t__docstrings",
    "t__docstrings__extract",
]
