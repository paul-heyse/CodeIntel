"""Native Hamilton implementation for docstrings target.

This module implements docstring extraction as a native Hamilton pipeline with:
- t__docstrings__extract: Execute DocstringsExtractStep to parse docstrings
- t__docstrings: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import get_module_paths_from_env, paths_to_modules
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import DocstringsExtractStep

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


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


@tag(domain="ingestion", target="docstrings", node_type="compute")
def t__docstrings__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> DocstringsExtractResult:
    """Execute docstring extraction on repository modules.

    This is the compute node for the docstrings target. It parses Python
    source files to extract docstrings from modules, classes, and functions,
    persisting structured information for documentation analysis.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    DocstringsExtractResult
        Result containing table row counts and any extraction errors.

    Notes
    -----
    Produces:
    - core.docstrings: Structured docstring data
    """
    if t__modules.status != "succeeded":
        return DocstringsExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        paths = get_module_paths_from_env(env)
        modules = paths_to_modules(paths, env.snapshot.repo_root)

        if not modules:
            log.info("No modules found for docstring extraction")
            return DocstringsExtractResult(
                success=True,
                table_counts={},
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

        step = DocstringsExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        return DocstringsExtractResult(
            success=True,
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )

    except Exception:
        log.exception("Docstring extraction failed")
        return DocstringsExtractResult(
            success=False,
            error="Docstring extraction failed with exception",
        )


@tag(domain="ingestion", target="docstrings", node_type="materialize")
def t__docstrings(
    env: BuildEnv,
    graph: TargetGraph,
    t__docstrings__extract: DocstringsExtractResult,
) -> TargetRunRecord:
    """Materialize docstrings target with validation.

    This is the entry point for the docstrings target. It orchestrates
    docstring extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__docstrings__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "docstrings")

    if executor.should_skip():
        return executor.skip()

    if not t__docstrings__extract.success:
        return executor.fail(
            RuntimeError(t__docstrings__extract.error or "Docstring extraction failed")
        )

    for error in t__docstrings__extract.errors:
        log.warning("Docstring extraction warning: %s", error)

    def compute() -> dict[str, int]:
        return dict(t__docstrings__extract.table_counts)

    return executor.execute(compute)


__all__ = [
    "DocstringsExtractResult",
    "t__docstrings",
    "t__docstrings__extract",
]
