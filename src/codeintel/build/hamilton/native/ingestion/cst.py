"""Native Hamilton implementation for cst target.

This module implements CST extraction as a native Hamilton pipeline with:
- t__cst__extract: Execute CstExtractStep to parse LibCST
- t__cst: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.ingestion.ast import get_module_paths_from_env
from codeintel.build.plugins.ingestion.helpers import paths_to_modules
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import CstExtractStep

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


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


@tag(domain="ingestion", target="cst", node_type="compute")
def t__cst__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> CstExtractResult:
    """Execute CST extraction on repository modules.

    This is the compute node for the cst target. It parses Python source
    files using LibCST, extracting concrete syntax tree nodes for
    detailed analysis.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    CstExtractResult
        Result containing table row counts and any extraction errors.

    Notes
    -----
    Produces:
    - core.cst_nodes: CST node information
    """
    if t__modules.status != "succeeded":
        return CstExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        paths = get_module_paths_from_env(env)
        modules = paths_to_modules(paths, env.snapshot.repo_root)

        if not modules:
            log.info("No modules found for CST extraction")
            return CstExtractResult(
                success=True,
                table_counts={},
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

        step = CstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        return CstExtractResult(
            success=True,
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )

    except Exception:
        log.exception("CST extraction failed")
        return CstExtractResult(
            success=False,
            error="CST extraction failed with exception",
        )


@tag(domain="ingestion", target="cst", node_type="materialize")
def t__cst(
    env: BuildEnv,
    graph: TargetGraph,
    t__cst__extract: CstExtractResult,
) -> TargetRunRecord:
    """Materialize CST target with validation.

    This is the entry point for the cst target. It orchestrates
    CST extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__cst__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "cst")

    if executor.should_skip():
        return executor.skip()

    if not t__cst__extract.success:
        return executor.fail(RuntimeError(t__cst__extract.error or "CST extraction failed"))

    for error in t__cst__extract.errors:
        log.warning("CST extraction warning: %s", error)

    def compute() -> dict[str, int]:
        return dict(t__cst__extract.table_counts)

    return executor.execute(compute)


__all__ = [
    "CstExtractResult",
    "t__cst",
    "t__cst__extract",
]
