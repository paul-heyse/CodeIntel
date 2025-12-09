"""Cyclopts wiring for docs export commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import ValidationError, run_handler
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    make_handler_context,
)
from codeintel.cli.docs_handlers import (
    BackendOptions,
    DocsExportOptions,
    DryRunMode,
    ExportValidationMode,
    MacroRequirement,
    NxGpuMode,
    PrereqMode,
    ProjectOptions,
    bundle_docs_export,
    docs_export_handler,
)

docs_app = App(
    name="docs",
    help="Document export utilities.",
)


@dataclass
class DocsBackendCli:
    """Graph backend selection."""

    class NxBackend(Enum):
        """NetworkX backend selection."""

        AUTO = "auto"
        CPU = "cpu"
        NX_CUGRAPH = "nx-cugraph"

    nx_backend: Annotated[
        NxBackend,
        Parameter(
            name="--nx-backend",
            help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
            show_choices=True,
        ),
    ] = NxBackend.AUTO
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

    validation_mode: Annotated[
        ExportValidationMode | None,
        Parameter(
            name="--validation-mode",
            help="Validation strategy: required or skip.",
            show_choices=True,
        ),
    ] = None
    validate: Annotated[
        bool,
        Parameter(
            name="--validate",
            help="Require validation for exports (exit code 1 on failures).",
            negative=(),
        ),
    ] = False
    macro_requirement: Annotated[
        MacroRequirement | None,
        Parameter(
            name="--macro-requirement",
            help="Normalized macro requirement policy: require_normalized or allow_partial.",
            show_choices=True,
        ),
    ] = None
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
    output: Annotated[OutputFormatCLI | None, Parameter(name="*")] = None
    run_mode: Annotated[
        DryRunMode | None,
        Parameter(
            name="--run-mode",
            help="Execution mode for docs export.",
            show_choices=True,
        ),
    ] = None
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Show export plan without executing.",
            negative=(),
        ),
    ] = False
    prereq_mode: Annotated[
        PrereqMode | None,
        Parameter(
            name="--prereq-mode",
            help="Prerequisite execution mode.",
            show_choices=True,
        ),
    ] = None
    skip_prereqs: Annotated[
        bool,
        Parameter(
            name="--skip-prereqs",
            help="Skip prerequisite pipeline execution.",
            negative=(),
        ),
    ] = False


DEFAULT_BACKEND = DocsBackendCli()
DEFAULT_EXPORT = DocsExportCli()


@dataclass
class DocsCli:
    """Grouped CLI surface for docs export."""

    runtime: Annotated[RuntimeCLI | None, Parameter(name="*")] = None
    backend: Annotated[DocsBackendCli | None, Parameter(name="*")] = None
    export: Annotated[DocsExportCli | None, Parameter(name="*")] = None


@dataclass(frozen=True)
class DocsExportBundle:
    """Typed bundle returned by docs export option normalization."""

    project: ProjectOptions
    backend: BackendOptions
    export_options: DocsExportOptions
    verbose: int


@docs_app.command(name="export")
@dataclass
class DocsExportCommand:
    """Export datasets to Document Output/.

    Enforces mutually exclusive pairs:
    - validation_mode vs --validate
    - run_mode vs --dry-run
    - prereq_mode vs --skip-prereqs
    """

    cfg: Annotated[DocsCli | None, Parameter(name="*")] = None

    def __call__(self) -> None:
        cfg = self.cfg
        if cfg is None:
            cfg = DocsCli()
        project_cfg = cfg.runtime if cfg.runtime is not None else RuntimeCLI()
        backend_cfg = cfg.backend if cfg.backend is not None else DocsBackendCli()
        export_cfg = cfg.export if cfg.export is not None else DocsExportCli()
        if export_cfg.validation_mode is not None and export_cfg.validate:
            message = "Provide either --validation-mode or --validate, not both."
            raise ValidationError(message)
        if export_cfg.run_mode is not None and export_cfg.dry_run:
            message = "Provide either --run-mode or --dry-run, not both."
            raise ValidationError(message)
        if export_cfg.prereq_mode is not None and export_cfg.skip_prereqs:
            message = "Provide either --prereq-mode or --skip-prereqs, not both."
            raise ValidationError(message)

        validation_mode = export_cfg.validation_mode
        if validation_mode is None:
            validation_mode = (
                ExportValidationMode.REQUIRED if export_cfg.validate else ExportValidationMode.SKIP
            )
        macro_requirement = export_cfg.macro_requirement
        if macro_requirement is None:
            macro_requirement = (
                MacroRequirement.REQUIRE_NORMALIZED
                if export_cfg.require_normalized_macros
                else MacroRequirement.ALLOW_PARTIAL
            )
        run_mode = export_cfg.run_mode
        if run_mode is None:
            run_mode = DryRunMode.DRY_RUN if export_cfg.dry_run else DryRunMode.EXECUTE
        prereq_mode = export_cfg.prereq_mode
        if prereq_mode is None:
            prereq_mode = PrereqMode.SKIP if export_cfg.skip_prereqs else PrereqMode.RUN

        runtime_opts, verbose, output_format = make_handler_context(
            project_cfg, export_cfg.output or OutputFormatCLI(), default_output=OutputFormat.TEXT
        )

        cli_kwargs = {
            "project_root": runtime_opts.project_root,
            "repo": runtime_opts.repo,
            "commit": runtime_opts.commit,
            "repo_root": runtime_opts.repo_root,
            "db_path": runtime_opts.db_path,
            "build_dir": runtime_opts.build_dir,
            "document_output_dir": runtime_opts.document_output_dir,
            "nx_backend": backend_cfg.nx_backend.value,
            "nx_gpu_mode": backend_cfg.nx_gpu_mode,
            "validation": validation_mode,
            "macro_requirement": macro_requirement,
            "schemas": export_cfg.schemas,
            "datasets": export_cfg.datasets,
            "output_format": output_format,
            "run_mode": run_mode,
            "prereq_mode": prereq_mode,
            "verbose": verbose,
        }
        try:
            bundled_mapping = bundle_docs_export(cli_kwargs)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        bundle = DocsExportBundle(
            project=bundled_mapping["project"],
            backend=bundled_mapping["backend"],
            export_options=bundled_mapping["export_options"],
            verbose=bundled_mapping["verbose"],
        )
        run_handler(
            docs_export_handler,
            bundle.project,
            bundle.backend,
            bundle.export_options,
            bundle.verbose,
        )


__all__ = ["docs_app"]
