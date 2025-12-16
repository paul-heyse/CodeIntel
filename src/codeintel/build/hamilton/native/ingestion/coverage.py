"""Native Hamilton implementation for coverage target.

This module implements coverage ingestion as a native Hamilton pipeline with:
- t__coverage__ingest: Execute CoverageIngestStep to load coverage data
- t__coverage: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.ingestion.ast import get_module_paths_from_env
from codeintel.build.plugins.ingestion.helpers import paths_to_modules
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute import CoverageIngestStep

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class CoverageIngestResult:
    """Result from coverage ingestion.

    Attributes
    ----------
    success
        Whether ingestion completed successfully.
    table_counts
        Row counts per produced table.
    skipped
        Whether ingestion was skipped (e.g., no coverage file found).
    error
        Error message if ingestion failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


def _resolve_coverage_file(env: BuildEnv) -> Path | None:
    """Resolve the coverage data file.

    Parameters
    ----------
    env
        Build environment.

    Returns
    -------
    Path | None
        Path to coverage file or None if not found.
    """
    repo_root = env.snapshot.repo_root
    build_dir = env.paths.build_dir

    candidates = [
        repo_root / ".coverage",
        repo_root / "coverage.json",
        build_dir / "coverage.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@tag(domain="ingestion", target="coverage_ingest", node_type="compute")
async def t__coverage_ingest__ingest(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> CoverageIngestResult:
    """Execute coverage data ingestion from coverage.py output.

    This is the compute node for the coverage target. It reads coverage.py's
    database or JSON export and extracts line-level coverage data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    CoverageIngestResult
        Result containing table row counts or skip indication.

    Notes
    -----
    Produces:
    - analytics.coverage_lines: Line-level coverage data
    """
    if t__modules.status != "succeeded":
        return CoverageIngestResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    coverage_path = _resolve_coverage_file(env)
    if coverage_path is None:
        log.info("No coverage file found, skipping coverage ingestion")
        return CoverageIngestResult(
            success=True,
            skipped=True,
            table_counts={},
        )

    try:
        paths = get_module_paths_from_env(env)
        modules = paths_to_modules(paths, env.snapshot.repo_root)

        storage = DuckDBStorageAdapter(env.gateway)
        tool = BuildToolAdapter(
            coverage_collector=None,  # Coverage collector from resources if available
        )

        step = CoverageIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(
            modules,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            coverage_file=coverage_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return CoverageIngestResult(
                success=False,
                error=f"Coverage ingest failed: {errors}",
            )

        return CoverageIngestResult(
            success=True,
            table_counts=result.table_counts or {},
        )

    except Exception:
        log.exception("Coverage ingestion failed")
        return CoverageIngestResult(
            success=False,
            error="Coverage ingestion failed with exception",
        )


@tag(domain="ingestion", target="coverage_ingest", node_type="materialize")
def t__coverage_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_ingest__ingest: CoverageIngestResult,
) -> TargetRunRecord:
    """Materialize coverage target with validation.

    This is the entry point for the coverage target. It orchestrates
    coverage ingestion and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__coverage_ingest__ingest
        Ingestion result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "coverage_ingest")

    if executor.should_skip():
        return executor.skip()

    if not t__coverage_ingest__ingest.success:
        return executor.fail(
            RuntimeError(t__coverage_ingest__ingest.error or "Coverage ingestion failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__coverage_ingest__ingest.table_counts)

    return executor.execute(compute)


__all__ = [
    "CoverageIngestResult",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
]

