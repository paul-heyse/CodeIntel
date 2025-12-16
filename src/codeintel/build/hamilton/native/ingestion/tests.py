"""Native Hamilton implementation for tests target.

This module implements test results ingestion as a native Hamilton pipeline with:
- t__tests__ingest: Execute TestsIngestStep to parse pytest reports
- t__tests: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import get_module_paths_from_env, paths_to_modules
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.compute import TestsIngestStep

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class TestsIngestResult:
    """Result from test results ingestion.

    Attributes
    ----------
    success
        Whether ingestion completed successfully.
    table_counts
        Row counts per produced table.
    skipped
        Whether ingestion was skipped (e.g., no report found).
    error
        Error message if ingestion failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


def _resolve_report_file(env: BuildEnv) -> Path | None:
    """Resolve the pytest report file.

    Parameters
    ----------
    env
        Build environment.

    Returns
    -------
    Path | None
        Path to report file or None if not found.
    """
    build_dir = env.paths.build_dir
    repo_root = env.snapshot.repo_root

    candidates = [
        build_dir / "test-results" / "pytest-report.json",
        build_dir / "test-results" / "pytest_report.json",
        build_dir / "pytest-report.json",
        build_dir / "pytest_report.json",
        build_dir / "report.json",
        repo_root / "pytest-report.json",
        repo_root / "pytest_report.json",
        repo_root / "report.json",
        repo_root / "test-results" / "pytest-report.json",
        repo_root / ".pytest_cache" / "pytest_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@tag(domain="ingestion", target="tests_ingest", node_type="compute")
def t__tests_ingest__ingest(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> TestsIngestResult:
    """Execute test results ingestion from pytest reports.

    This is the compute node for the tests target. It reads pytest's
    JSON report output and extracts test results for storage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    TestsIngestResult
        Result containing table row counts or skip indication.

    Notes
    -----
    Produces:
    - analytics.test_results: Test execution results
    """
    if t__modules.status != "succeeded":
        return TestsIngestResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    report_path = _resolve_report_file(env)
    if report_path is None:
        log.info("No pytest report found, skipping tests ingestion")
        return TestsIngestResult(
            success=True,
            skipped=True,
            table_counts={},
        )

    try:
        paths = get_module_paths_from_env(env)
        modules = paths_to_modules(paths, env.snapshot.repo_root)

        storage = DuckDBStorageAdapter(env.gateway)

        step = TestsIngestStep(storage=storage)
        result = step.execute(
            modules,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            json_report_path=report_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TestsIngestResult(
                success=False,
                error=f"Tests ingest failed: {errors}",
            )

        return TestsIngestResult(
            success=True,
            table_counts=result.table_counts or {},
        )

    except Exception:
        log.exception("Tests ingestion failed")
        return TestsIngestResult(
            success=False,
            error="Tests ingestion failed with exception",
        )


@tag(domain="ingestion", target="tests_ingest", node_type="materialize")
def t__tests_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__tests_ingest__ingest: TestsIngestResult,
) -> TargetRunRecord:
    """Materialize tests target with validation.

    This is the entry point for the tests target. It orchestrates
    test results ingestion and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__tests_ingest__ingest
        Ingestion result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "tests_ingest")

    if executor.should_skip():
        return executor.skip()

    if not t__tests_ingest__ingest.success:
        return executor.fail(
            RuntimeError(t__tests_ingest__ingest.error or "Tests ingestion failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__tests_ingest__ingest.table_counts)

    return executor.execute(compute)


__all__ = [
    "TestsIngestResult",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
]
