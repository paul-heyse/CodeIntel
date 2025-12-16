"""Native Hamilton implementation for ast target.

This module implements AST extraction as a native Hamilton pipeline with:
- t__ast__extract: Execute AstExtractStep to parse Python AST
- t__ast: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import paths_to_modules
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import AstExtractStep
from codeintel.storage.ibis_types import ibis_bool

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


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


@tag(domain="ingestion", target="ast", node_type="compute")
def t__ast__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> AstExtractResult:
    """Execute AST extraction on repository modules.

    This is the compute node for the ast target. It parses Python source
    files using the stdlib AST module, extracting node information and
    computing file-level metrics.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    AstExtractResult
        Result containing table row counts and any extraction errors.

    Notes
    -----
    Produces two tables:
    - core.ast_nodes: Individual AST node information
    - core.ast_metrics: File-level AST metrics
    """
    if t__modules.status != "succeeded":
        return AstExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        # Get module paths from gateway
        paths = _get_module_paths_from_env(env)
        modules = paths_to_modules(paths, env.snapshot.repo_root)

        if not modules:
            log.info("No modules found for AST extraction")
            return AstExtractResult(
                success=True,
                table_counts={},
            )

        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

        step = AstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
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
        return AstExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="ingestion", target="ast", node_type="materialize")
def t__ast(
    env: BuildEnv,
    graph: TargetGraph,
    t__ast__extract: AstExtractResult,
) -> TargetRunRecord:
    """Materialize AST target with validation.

    This is the entry point for the ast target. It orchestrates
    AST extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__ast__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "ast")

    if executor.should_skip():
        return executor.skip()

    if not t__ast__extract.success:
        return executor.fail(RuntimeError(t__ast__extract.error or "AST extraction failed"))

    # Log non-fatal errors
    for error in t__ast__extract.errors:
        log.warning("AST extraction warning: %s", error)

    def compute() -> dict[str, int]:
        return dict(t__ast__extract.table_counts)

    return executor.execute(compute)


def _get_module_paths_from_env(env: BuildEnv) -> list[str]:
    """Get module paths from environment.

    This is a private helper function, not a Hamilton node.

    Parameters
    ----------
    env
        Build environment.

    Returns
    -------
    list[str]
        List of module paths.
    """
    try:
        table = env.gateway.ibis.table("core.modules")
        df = (
            table.filter(
                [
                    ibis_bool(table.repo == env.snapshot.repo),
                    ibis_bool(table.commit == env.snapshot.commit),
                ]
            )
            .select("path")
            .execute()
        )
        return [str(path) for path in df["path"].tolist()]
    except (RuntimeError, OSError) as exc:
        log.warning("gateway error fetching module paths: %s", exc)
        return []


# Public alias for backward compatibility
get_module_paths_from_env = _get_module_paths_from_env


__all__ = [
    "AstExtractResult",
    "t__ast",
    "t__ast__extract",
]
