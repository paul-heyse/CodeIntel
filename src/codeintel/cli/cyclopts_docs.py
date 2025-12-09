"""Cyclopts wiring for docs export commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.commands.docs import (
    BackendOptions,
    DocsExportOptions,
    DryRunMode,
    NxGpuMode,
    OutputFormat,
    ProjectOptions,
    _bundle_docs_export,
    docs_export_handler,
)
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    resolve_output_format,
    runtime_cli_to_options,
)

docs_app = App(
    name="docs",
    help="Document export utilities.",
)


@dataclass
class DocsProjectCli:
    """Project and storage selection for docs export."""

    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)


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
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)
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


@dataclass(frozen=True)
class DocsExportBundle:
    """Typed bundle returned by docs export option normalization."""

    project: ProjectOptions
    backend: BackendOptions
    export_options: DocsExportOptions
    verbose: int


@docs_app.command(name="export")
def docs_export(
    project: Annotated[DocsProjectCli, Parameter(name="*")] | None = None,
    backend: Annotated[DocsBackendCli, Parameter(name="*")] | None = None,
    export: Annotated[DocsExportCli, Parameter(name="*")] | None = None,
) -> None:
    """Export datasets to Document Output/."""
    project_cfg = project or DocsProjectCli()
    backend_cfg = backend or DocsBackendCli()
    export_cfg = export or DocsExportCli()

    runtime = runtime_cli_to_options(project_cfg.runtime)
    output_format = resolve_output_format(
        json_flag=export_cfg.output.json,
        explicit=export_cfg.output.output_format,
        default=OutputFormat.TEXT,
    )

    cli_kwargs = {
        "project_root": runtime.project_root,
        "repo": runtime.repo,
        "commit": runtime.commit,
        "repo_root": runtime.repo_root,
        "db_path": runtime.db_path,
        "build_dir": runtime.build_dir,
        "document_output_dir": runtime.document_output_dir,
        "nx_backend": backend_cfg.nx_backend,
        "nx_gpu_mode": backend_cfg.nx_gpu_mode,
        "validation": export_cfg.validate,
        "macro_requirement": export_cfg.require_normalized_macros,
        "schemas": export_cfg.schemas,
        "datasets": export_cfg.datasets,
        "output_format": output_format,
        "run_mode": DryRunMode.DRY_RUN if export_cfg.dry_run else DryRunMode.EXECUTE,
        "prereq_mode": export_cfg.skip_prereqs,
        "verbose": project_cfg.runtime.verbose,
    }
    bundled_mapping = _bundle_docs_export(cli_kwargs)
    bundle = DocsExportBundle(
        project=bundled_mapping["project"],
        backend=bundled_mapping["backend"],
        export_options=bundled_mapping["export_options"],
        verbose=bundled_mapping["verbose"],
    )
    invoke_with_typer_translation(
        docs_export_handler,
        bundle.project,
        bundle.backend,
        bundle.export_options,
        bundle.verbose,
    )


__all__ = ["docs_app"]
