"""Cyclopts wiring for docs export commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from cyclopts import App, Parameter

from codeintel.cli.commands.docs import (
    DryRunMode,
    NxGpuMode,
    OutputFormat,
    _bundle_docs_export,
    docs_export_handler,
)
from codeintel.cli.cyclopts_common import JsonFlag, OutputFmt, ProjectRoot, Verbose

docs_app = App(
    name="docs",
    help="Document export utilities.",
)


@dataclass
class DocsProjectCli:
    """Project and storage selection for docs export."""

    project_root: ProjectRoot = None
    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'org/repo'). Uses project config if omitted.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Uses project config if omitted.",
        ),
    ] = None
    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database. Uses project config if omitted.",
        ),
    ] = None
    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory (default: build/).",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root (default: current directory).",
        ),
    ] = None
    document_output_dir: Annotated[
        Path | None,
        Parameter(
            name="--document-output-dir",
            help="Override Document Output/ directory.",
        ),
    ] = None


@dataclass
class DocsBackendCli:
    """Graph backend selection."""

    nx_backend: Annotated[
        str,
        Parameter(
            name="--nx-backend",
            help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
        ),
    ] = "auto"
    nx_gpu_mode: Annotated[
        NxGpuMode,
        Parameter(
            name="--nx-gpu-mode",
            help="GPU backend preference: disabled, enabled, or strict.",
        ),
    ] = NxGpuMode.DISABLED


@dataclass
class DocsExportCli:
    """Export and validation options."""

    validate: Annotated[
        bool,
        Parameter(
            name="--validate",
            help="Require validation for exports (exit code 1 on failures).",
            negative=(),
        ),
    ] = False
    require_normalized_macros: Annotated[
        bool,
        Parameter(
            name="--require-normalized-macros",
            help="Require normalized macros during export.",
            negative=(),
        ),
    ] = False
    schemas: Annotated[
        list[str] | None,
        Parameter(
            name="--schema",
            help="Schema name to validate (repeatable).",
        ),
    ] = None
    datasets: Annotated[
        list[str] | None,
        Parameter(
            name="--dataset",
            help="Dataset name to export (repeatable).",
        ),
    ] = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Show export plan without executing.",
            negative=(),
        ),
    ] = False
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False
    verbose: Verbose = 0


@docs_app.command(name="export")
def docs_export(
    project: Annotated[DocsProjectCli, Parameter(name="*")] | None = None,
    backend: Annotated[DocsBackendCli, Parameter(name="*")] | None = None,
    export: Annotated[DocsExportCli, Parameter(name="*")] | None = None,
) -> None:
    """Export datasets to Document Output/.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    project_cfg = project or DocsProjectCli()
    backend_cfg = backend or DocsBackendCli()
    export_cfg = export or DocsExportCli()

    output_format = export_cfg.output_format
    if export_cfg.json:
        output_format = OutputFormat.JSON

    cli_kwargs = {
        "project_root": project_cfg.project_root,
        "repo": project_cfg.repo,
        "commit": project_cfg.commit,
        "repo_root": project_cfg.repo_root,
        "db_path": project_cfg.db_path,
        "build_dir": project_cfg.build_dir,
        "document_output_dir": project_cfg.document_output_dir,
        "nx_backend": backend_cfg.nx_backend,
        "nx_gpu_mode": backend_cfg.nx_gpu_mode,
        "validation": export_cfg.validate,
        "macro_requirement": export_cfg.require_normalized_macros,
        "schemas": export_cfg.schemas,
        "datasets": export_cfg.datasets,
        "output_format": output_format,
        "run_mode": DryRunMode.DRY_RUN if export_cfg.dry_run else DryRunMode.EXECUTE,
        "prereq_mode": export_cfg.skip_prereqs,
    }
    bundled = _bundle_docs_export(cli_kwargs)
    try:
        docs_export_handler(
            bundled["project"],
            bundled["backend"],
            bundled["export_options"],
            export_cfg.verbose,
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


__all__ = ["docs_app"]
