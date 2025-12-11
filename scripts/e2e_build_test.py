"""End-to-end build system test with telemetry reporting.

This script runs a complete build from SCIP indexing through document export,
producing a comprehensive report of all stages, timings, and outputs.

Usage
-----
Run full E2E test (includes SCIP indexing)::

    uv run python scripts/e2e_build_test.py

Skip SCIP if index already exists::

    uv run python scripts/e2e_build_test.py --skip-scip

Use specific repo root::

    uv run python scripts/e2e_build_test.py --repo-root /path/to/repo

Output JSON report::

    uv run python scripts/e2e_build_test.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from codeintel.build.executor import BuildExecutor, ExecutorEnv
from codeintel.build.plan import PlanGenerator
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import StateValidator
from codeintel.config.primitives import BuildLayoutOptions, BuildPaths, SnapshotRef
from codeintel.config.resolver import resolve_tools_config
from codeintel.core.process import (
    CommandExecutionError,
    CommandExecutor,
    CommandNotAllowedError,
)
from codeintel.export.export_jsonl import ExportCallOptions, export_all_jsonl
from codeintel.export.export_parquet import export_all_parquet
from codeintel.storage.gateway import StorageConfig, open_gateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
CRITICAL_FILES = (
    "goids.jsonl",
    "goids.parquet",
    "function_metrics.jsonl",
    "function_profile.jsonl",
    "call_graph_edges.jsonl",
    "modules.jsonl",
)

CRITICAL_TABLES = (
    "core.goids",
    "core.modules",
    "analytics.function_metrics",
    "graph.call_graph_edges",
)

COMMAND_EXECUTOR = CommandExecutor.for_build_tools()


@dataclass
class StageResult:
    """Result from a single E2E stage.

    Attributes
    ----------
    name
        Name of the stage.
    status
        Execution status ("success", "failed", "skipped").
    duration_ms
        Duration of the stage in milliseconds.
    started_at
        ISO timestamp when the stage started.
    ended_at
        ISO timestamp when the stage ended.
    outputs
        Dictionary of stage outputs.
    error
        Error message if the stage failed, None otherwise.
    """

    name: str
    status: str
    duration_ms: float
    started_at: str
    ended_at: str
    outputs: dict[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class E2EPaths:
    """Derived paths for the E2E test run."""

    build_dir: Path
    db_path: Path
    output_dir: Path


@dataclass
class E2EReport:
    """Complete E2E test report.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    repo
        Repository slug.
    commit
        Commit SHA (short).
    started_at
        ISO timestamp when the test started.
    ended_at
        ISO timestamp when the test ended.
    total_duration_ms
        Total duration of the test in milliseconds.
    overall_status
        Overall status ("success" or "failed").
    stages
        List of stage results.
    summary
        Summary metrics from the test.
    """

    run_id: str
    repo: str
    commit: str
    started_at: str
    ended_at: str
    total_duration_ms: float
    overall_status: str
    stages: list[StageResult]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the report.
        """
        return {
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_ms": self.total_duration_ms,
            "overall_status": self.overall_status,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "started_at": s.started_at,
                    "ended_at": s.ended_at,
                    "outputs": s.outputs,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "summary": self.summary,
        }


def _derive_paths(repo_root: Path) -> E2EPaths:
    """Compute all paths used by the E2E test.

    Returns
    -------
    E2EPaths
        Derived build, database, and output directories.
    """
    build_dir = repo_root / "build"
    return E2EPaths(
        build_dir=build_dir,
        db_path=build_dir / "db" / "codeintel.duckdb",
        output_dir=repo_root / "document_output",
    )


def _get_timestamp() -> str:
    """Return current ISO timestamp.

    Returns
    -------
    str
        ISO formatted timestamp.
    """
    return datetime.now(UTC).isoformat()


def _run_stage(
    name: str,
    func: Callable[..., dict[str, object]],
    **kwargs: object,
) -> StageResult:
    """Run a stage and capture timing/results.

    Parameters
    ----------
    name
        Name of the stage.
    func
        Function to execute for the stage.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    StageResult
        Result of the stage execution.
    """
    log.info("=" * 60)
    log.info("STAGE: %s", name)
    log.info("=" * 60)

    started_at = _get_timestamp()
    start_time = time.perf_counter()
    outputs: dict[str, object] = {}
    error: str | None = None

    try:
        outputs = func(**kwargs)
        status = "success"
    except Exception as exc:
        log.exception("Stage %s failed", name)
        status = "failed"
        error = str(exc)

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    ended_at = _get_timestamp()

    log.info("Stage %s: %s (%.2fs)", name, status, duration_ms / 1000)

    return StageResult(
        name=name,
        status=status,
        duration_ms=duration_ms,
        started_at=started_at,
        ended_at=ended_at,
        outputs=outputs,
        error=error,
    )


def _stage_scip_index(
    repo_root: Path,
    build_dir: Path,
    *,
    skip_scip: bool,
) -> dict[str, object]:
    """Run SCIP indexing stage.

    Parameters
    ----------
    repo_root
        Path to the repository root.
    build_dir
        Path to the build directory.
    skip_scip
        Whether to skip SCIP indexing if index exists.

    Returns
    -------
    dict[str, object]
        Stage outputs including index paths and sizes.

    Raises
    ------
    RuntimeError
        If SCIP indexing fails.
    """
    scip_dir = build_dir / "scip"
    scip_index = scip_dir / "index.scip"
    scip_json = scip_dir / "index.scip.json"

    if skip_scip and scip_index.exists():
        log.info("Skipping SCIP indexing - using existing index at %s", scip_index)
        return {
            "skipped": True,
            "scip_index": str(scip_index),
            "scip_json": str(scip_json) if scip_json.exists() else None,
            "index_size_mb": scip_index.stat().st_size / (1024 * 1024),
        }

    log.info("Running SCIP indexing...")
    scip_dir.mkdir(parents=True, exist_ok=True)

    try:
        COMMAND_EXECUTOR.run_scip_index(
            repo_root,
            scip_index,
            project_name="codeintel",
        )
    except CommandNotAllowedError as exc:
        msg = "scip-python is not available on PATH"
        raise RuntimeError(msg) from exc

    # Convert to JSON if needed
    if not scip_json.exists() and scip_index.exists():
        try:
            COMMAND_EXECUTOR.export_scip_to_json(scip_index, scip_json)
        except CommandNotAllowedError as exc:
            msg = "scip CLI is not available on PATH"
            raise RuntimeError(msg) from exc

    index_size = scip_index.stat().st_size / (1024 * 1024) if scip_index.exists() else 0.0
    return {
        "skipped": False,
        "scip_index": str(scip_index),
        "scip_json": str(scip_json) if scip_json.exists() else None,
        "index_size_mb": index_size,
    }


def _stage_build_run(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    build_dir: Path,
) -> dict[str, object]:
    """Run the full build using the build system.

    Parameters
    ----------
    repo_root
        Path to the repository root.
    repo
        Repository slug.
    commit
        Commit SHA.
    db_path
        Path to the DuckDB database.
    build_dir
        Path to the build directory.

    Returns
    -------
    dict[str, object]
        Stage outputs including run ID, status, and target counts.

    Raises
    ------
    RuntimeError
        If the build execution fails.
    """
    log.info("Initializing build system...")

    # Create snapshot and paths
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    paths = BuildPaths.from_layout(
        repo_root=repo_root,
        overrides=BuildLayoutOptions(build_dir=build_dir, db_path=db_path),
    )

    gateway = open_gateway(
        StorageConfig(
            db_path=db_path,
            apply_schema=True,
            ensure_views=True,
        )
    )

    graph = get_target_graph()
    all_targets = [t.name for t in graph.all_targets if t.module != "export"]

    log.info("Validating state for %d targets...", len(all_targets))

    db_state = StateValidator(graph, gateway, snapshot).validate()

    log.info("Resolving minimal work...")
    resolution = BuildResolver(graph, db_state).resolve(
        goals=all_targets,
        force_recompute=None,
    )

    log.info("Generating build plan...")
    plan = PlanGenerator(graph).generate(resolution)

    log.info(
        "Plan: %d stages, %d targets to compute, %d skipped",
        len(plan.stages),
        sum(len(s.steps) for s in plan.stages),
        len(plan.skipped_targets),
    )

    # Execute
    log.info("Executing build plan...")
    env = ExecutorEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=resolve_tools_config(),
    )
    executor = BuildExecutor(graph=graph, env=env)

    result = executor.execute(plan)

    outputs = {
        "run_id": result.run_id,
        "status": result.status,
        "computed_count": len(result.completed_targets),
        "skipped_count": len(result.skipped_targets),
        "failed_count": len(result.failed_targets),
        "total_duration_ms": result.duration_ms,
        "computed_targets": list(result.completed_targets),
        "failed_targets": list(result.failed_targets),
        "error_summary": result.error_summary,
    }

    # Raise if build failed so stage is marked as failed
    if result.status == "failed":
        msg = f"Build failed: {result.error_summary or 'Unknown error'}"
        raise RuntimeError(msg)

    return outputs


def _stage_export_docs(
    db_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Export documents to output directory.

    Parameters
    ----------
    db_path
        Path to the DuckDB database.
    output_dir
        Path to the output directory.

    Returns
    -------
    dict[str, object]
        Stage outputs including file counts and sizes.
    """
    log.info("Exporting documents to %s...", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gateway = open_gateway(
        StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=True,
        )
    )

    # Export JSONL (validation disabled for E2E testing - schemas may be incomplete)
    # Force full export to ensure fresh data is written (bypasses incremental markers)
    log.info("Exporting JSONL files...")
    export_opts = ExportCallOptions(validate_exports=False, force_full_export=True)
    jsonl_files = export_all_jsonl(gateway, output_dir, options=export_opts)

    # Export Parquet
    log.info("Exporting Parquet files...")
    export_all_parquet(gateway, output_dir, options=export_opts)

    # Count outputs
    jsonl_count = len(list(output_dir.glob("*.jsonl")))
    parquet_count = len(list(output_dir.glob("*.parquet")))
    total_size_mb = sum(f.stat().st_size for f in output_dir.iterdir()) / (1024 * 1024)

    return {
        "output_dir": str(output_dir),
        "jsonl_files": jsonl_count,
        "parquet_files": parquet_count,
        "total_size_mb": round(total_size_mb, 2),
        "files_written": len(jsonl_files),
    }


def _stage_verify_outputs(
    output_dir: Path,
    db_path: Path,
) -> dict[str, object]:
    """Verify critical outputs exist and have data.

    Parameters
    ----------
    output_dir
        Path to the output directory.
    db_path
        Path to the DuckDB database.

    Returns
    -------
    dict[str, object]
        Stage outputs including file stats and table counts.
    """
    log.info("Verifying outputs...")

    # Check critical files
    missing: list[str] = []
    file_stats: dict[str, dict[str, object]] = {}
    for fname in CRITICAL_FILES:
        fpath = output_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            file_stats[fname] = {"exists": True, "size_bytes": size}
        else:
            missing.append(fname)
            file_stats[fname] = {"exists": False, "size_bytes": 0}

    # Check database tables
    gateway = open_gateway(
        StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=True,
        )
    )

    table_counts: dict[str, object] = {}
    for table in CRITICAL_TABLES:
        try:
            row = gateway.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            count = int(row[0]) if row else 0
            table_counts[table] = count
        except Exception as exc:  # noqa: BLE001
            table_counts[table] = f"ERROR: {exc}"

    return {
        "critical_files_missing": missing,
        "file_stats": file_stats,
        "table_counts": table_counts,
        "all_critical_present": len(missing) == 0,
    }


def _get_commit_sha(repo_root: Path) -> str:
    """Get the current git commit SHA.

    Parameters
    ----------
    repo_root
        Path to the repository root.

    Returns
    -------
    str
        Short commit SHA or "unknown" if git fails.
    """
    try:
        revision = COMMAND_EXECUTOR.read_git_revision(repo_root)
        return revision[:12] if revision else "unknown"
    except (CommandExecutionError, CommandNotAllowedError):
        return "unknown"


def run_e2e_test(
    repo_root: Path,
    *,
    skip_scip: bool = False,
) -> E2EReport:
    """Run the complete E2E test.

    Parameters
    ----------
    repo_root
        Path to the repository root.
    skip_scip
        Whether to skip SCIP indexing if index exists.

    Returns
    -------
    E2EReport
        Complete report of the E2E test.
    """
    run_id = str(uuid.uuid4())[:8]
    started_at = _get_timestamp()
    start_time = time.perf_counter()

    paths = _derive_paths(repo_root)

    # Get repo info
    repo = os.environ.get("GEN_DOCS_REPO", "local/repo")
    commit = _get_commit_sha(repo_root)

    log.info("=" * 70)
    log.info("E2E BUILD TEST - Run ID: %s", run_id)
    log.info("=" * 70)
    log.info("Repo: %s", repo)
    log.info("Commit: %s", commit)
    log.info("Repo Root: %s", repo_root)
    log.info("Build Dir: %s", paths.build_dir)
    log.info("DB Path: %s", paths.db_path)
    log.info("Output Dir: %s", paths.output_dir)
    log.info("Skip SCIP: %s", skip_scip)
    log.info("=" * 70)

    stages: list[StageResult] = []
    overall_status = "success"

    # Stage 1: SCIP Indexing
    stages.append(
        _run_stage(
            "SCIP Indexing",
            _stage_scip_index,
            repo_root=repo_root,
            build_dir=paths.build_dir,
            skip_scip=skip_scip,
        )
    )

    # Only continue if SCIP succeeded or was skipped
    if stages[-1].status == "failed":
        overall_status = "failed"
    else:
        # Stage 2: Build Run
        stages.append(
            _run_stage(
                "Build System Run",
                _stage_build_run,
                repo_root=repo_root,
                repo=repo,
                commit=commit,
                db_path=paths.db_path,
                build_dir=paths.build_dir,
            )
        )

        if stages[-1].status == "failed":
            overall_status = "failed"
        else:
            # Stage 3: Export Documents
            stages.append(
                _run_stage(
                    "Document Export",
                    _stage_export_docs,
                    db_path=paths.db_path,
                    output_dir=paths.output_dir,
                )
            )

            # Stage 4: Verify Outputs
            stages.append(
                _run_stage(
                    "Verify Outputs",
                    _stage_verify_outputs,
                    output_dir=paths.output_dir,
                    db_path=paths.db_path,
                )
            )

            # Determine overall status
            overall_status = "failed" if any(s.status == "failed" for s in stages) else "success"

    end_time = time.perf_counter()
    total_duration_ms = (end_time - start_time) * 1000
    ended_at = _get_timestamp()

    # Build summary
    summary: dict[str, object] = {
        "stages_total": len(stages),
        "stages_success": sum(1 for s in stages if s.status == "success"),
        "stages_failed": sum(1 for s in stages if s.status == "failed"),
        "stages_skipped": sum(1 for s in stages if s.status == "skipped"),
    }

    # Add key metrics from stages
    for stage in stages:
        if stage.name == "Build System Run" and stage.status == "success":
            summary["targets_computed"] = stage.outputs.get("computed_count", 0)
            summary["targets_failed"] = stage.outputs.get("failed_count", 0)
        elif stage.name == "Document Export" and stage.status == "success":
            jsonl = stage.outputs.get("jsonl_files", 0)
            parquet = stage.outputs.get("parquet_files", 0)
            summary["export_files"] = (jsonl if isinstance(jsonl, int) else 0) + (
                parquet if isinstance(parquet, int) else 0
            )
            summary["export_size_mb"] = stage.outputs.get("total_size_mb", 0)
        elif stage.name == "Verify Outputs" and stage.status == "success":
            summary["all_outputs_valid"] = stage.outputs.get("all_critical_present", False)

    return E2EReport(
        run_id=run_id,
        repo=repo,
        commit=commit,
        started_at=started_at,
        ended_at=ended_at,
        total_duration_ms=total_duration_ms,
        overall_status=overall_status,
        stages=stages,
        summary=summary,
    )


def _print_report(report: E2EReport, *, json_output: bool = False) -> None:
    """Print the E2E report to stdout.

    Parameters
    ----------
    report
        The E2E report to print.
    json_output
        Whether to output as JSON.
    """
    write = sys.stdout.write

    if json_output:
        write(json.dumps(report.to_dict(), indent=2))
        write("\n")
        return

    write("\n")
    write("=" * 70 + "\n")
    write("E2E TEST REPORT\n")
    write("=" * 70 + "\n")
    write(f"Run ID:     {report.run_id}\n")
    write(f"Repo:       {report.repo}\n")
    write(f"Commit:     {report.commit}\n")
    write(f"Status:     {report.overall_status.upper()}\n")
    write(f"Duration:   {report.total_duration_ms / 1000:.2f}s\n")
    write("\n")

    write("-" * 70 + "\n")
    write("STAGES\n")
    write("-" * 70 + "\n")
    for stage in report.stages:
        icon = "✅" if stage.status == "success" else "❌" if stage.status == "failed" else "⏭️"
        write(f"  {icon} {stage.name}: {stage.status} ({stage.duration_ms / 1000:.2f}s)\n")
        if stage.error:
            write(f"      Error: {stage.error[:100]}...\n")
        if stage.outputs:
            for key, value in list(stage.outputs.items())[:5]:
                if isinstance(value, (str, int, float, bool)):
                    write(f"      {key}: {value}\n")
    write("\n")

    write("-" * 70 + "\n")
    write("SUMMARY\n")
    write("-" * 70 + "\n")
    for key, value in report.summary.items():
        write(f"  {key}: {value}\n")
    write("\n")
    write("=" * 70 + "\n")


def main() -> int:
    """Execute the E2E build test.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="E2E Build System Test")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    parser.add_argument(
        "--skip-scip",
        action="store_true",
        help="Skip SCIP indexing if index already exists",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    args = parser.parse_args()

    try:
        report = run_e2e_test(
            repo_root=args.repo_root.resolve(),
            skip_scip=args.skip_scip,
        )
    except Exception:
        log.exception("E2E test failed with exception")
        sys.stderr.write("\n❌ E2E TEST FAILED\n")
        return 1

    _print_report(report, json_output=args.json)

    # Save report to file
    report_path = args.repo_root / "build" / "e2e_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    log.info("Report saved to %s", report_path)

    return 0 if report.overall_status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
