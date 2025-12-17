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

from hamilton.function_modifiers import cache, tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.target_spec_helpers import make_output_target
from codeintel.build.resources import CPU_INTENSIVE_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import AstExtractStep, CstExtractStep, DocstringsExtractStep
from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ModuleRecord)

TARGET_SPECS = (
    make_output_target(
        name="ast",
        module="ingestion",
        description="Python AST extraction and metrics.",
        table_keys=(
            "core.ast_nodes",
            "core.ast_metrics",
        ),
        resources=TargetResources(tracker=True, modules=True),
        execution=CPU_INTENSIVE_EXECUTION,
    ),
    make_output_target(
        name="cst",
        module="ingestion",
        description="Concrete syntax tree extraction.",
        table_keys=("core.cst_nodes",),
    ),
    make_output_target(
        name="docstrings",
        module="ingestion",
        description="Docstring extraction and parsing.",
        table_keys=("core.docstrings",),
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


@cache(format="memory")
@tag(domain="ingestion", target="ast", node_type="tool")
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
            return AstExtractResult(success=True, table_counts={})

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
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("AST extraction failed")
        return AstExtractResult(success=False, error=str(exc))


@tag(domain="ingestion", target="ast", node_type="materialize")
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
    executor = NativeTargetExecutor.for_target(env, graph, "ast")
    if executor.should_skip():
        return executor.skip()
    if not t__ast__extract.success:
        return executor.fail(RuntimeError(t__ast__extract.error or "AST extraction failed"))

    for warning in t__ast__extract.errors:
        log.warning("AST extraction warning: %s", warning)

    return executor.execute(lambda: dict(t__ast__extract.table_counts))


@cache(format="memory")
@tag(domain="ingestion", target="cst", node_type="tool")
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
            return CstExtractResult(success=True, table_counts={})

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
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("CST extraction failed")
        return CstExtractResult(success=False, error=str(exc))


@tag(domain="ingestion", target="cst", node_type="materialize")
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
    executor = NativeTargetExecutor.for_target(env, graph, "cst")
    if executor.should_skip():
        return executor.skip()
    if not t__cst__extract.success:
        return executor.fail(RuntimeError(t__cst__extract.error or "CST extraction failed"))

    for warning in t__cst__extract.errors:
        log.warning("CST extraction warning: %s", warning)

    return executor.execute(lambda: dict(t__cst__extract.table_counts))


@cache(format="memory")
@tag(domain="ingestion", target="docstrings", node_type="tool")
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
            return DocstringsExtractResult(success=True, table_counts={})

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
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )
    except Exception as exc:
        log.exception("Docstring extraction failed")
        return DocstringsExtractResult(success=False, error=str(exc))


@tag(domain="ingestion", target="docstrings", node_type="materialize")
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
    executor = NativeTargetExecutor.for_target(env, graph, "docstrings")
    if executor.should_skip():
        return executor.skip()
    if not t__docstrings__extract.success:
        return executor.fail(
            RuntimeError(t__docstrings__extract.error or "Docstring extraction failed")
        )

    for warning in t__docstrings__extract.errors:
        log.warning("Docstring extraction warning: %s", warning)

    return executor.execute(lambda: dict(t__docstrings__extract.table_counts))


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
