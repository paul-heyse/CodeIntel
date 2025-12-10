"""Typer-free handlers for docs export commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError` or
:class:`~codeintel.cli.errors.DocsValidationError`.

.. deprecated:: 2.0
    This module is deprecated. Use codeintel.cli.handlers.docs instead.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from codeintel.build import get_target_graph

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
from codeintel.build.executor import BuildExecutor
from codeintel.build.plan import PlanGenerator
from codeintel.build.resolver import BuildResolver, ResolutionResult
from codeintel.build.state import StateValidator
from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.errors import DocsValidationError

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.project import (
    ProjectNotFoundError,
    detect_commit,
    find_project_root,
    load_project_config,
)
from codeintel.cli.results import CliResult
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.primitives import (
    BuildLayoutOptions,
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.export.export_jsonl import ExportCallOptions
from codeintel.export.runner import (
    ExportOptions,
    ExportRunner,
    run_validated_exports,
)
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.serving.backend.datasets import validate_dataset_registry
from codeintel.serving.services.errors import ExportError, log_problem
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

warnings.warn(
    "codeintel.cli.docs_handlers is deprecated. Use codeintel.cli.handlers.docs instead.",
    DeprecationWarning,
    stacklevel=2,
)

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class OutputFormat(Enum):
    """Output rendering format."""

    TEXT = "text"
    JSON = "json"


class ExportValidationMode(Enum):
    """Validation strategy for docs exports."""

    REQUIRED = "required"
    SKIP = "skip"


class MacroRequirement(Enum):
    """Requirement policy for normalized macros."""

    REQUIRE_NORMALIZED = "require_normalized"
    ALLOW_PARTIAL = "allow_partial"


class DryRunMode(Enum):
    """Execution mode for docs exports."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class NxGpuMode(Enum):
    """GPU backend mode for NetworkX."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    STRICT = "strict"


class PrereqMode(Enum):
    """Prerequisite execution strategy."""

    RUN = "run"
    SKIP = "skip"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DocsExportOptions:
    """Bundled options for docs export workflows."""

    validation: ExportValidationMode = ExportValidationMode.REQUIRED
    macro_requirement: MacroRequirement = MacroRequirement.REQUIRE_NORMALIZED
    datasets: list[str] | None = None
    schemas: list[str] | None = None
    output_format: OutputFormat = OutputFormat.TEXT
    run_mode: DryRunMode = DryRunMode.EXECUTE
    prereq_mode: PrereqMode = PrereqMode.RUN


@dataclass(frozen=True)
class ProjectOptions:
    """Project/runtime resolution inputs."""

    project_root: Path | None
    repo: str | None
    commit: str | None
    db_path: Path | None
    build_dir: Path | None
    repo_root: Path | None
    document_output_dir: Path | None


@dataclass(frozen=True)
class BackendOptions:
    """Graph backend selection."""

    nx_backend: str
    nx_gpu_mode: NxGpuMode


@dataclass(frozen=True)
class BackendFlags:
    """Backend preferences provided via CLI."""

    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RepoSelection:
    """Repository selection inputs."""

    project_root: Path | None
    repo: str | None
    commit: str | None
    repo_root: Path | None


@dataclass(frozen=True)
class StorageSelection:
    """Storage and build path inputs."""

    db_path: Path | None
    build_dir: Path | None
    document_output_dir: Path | None


@dataclass(frozen=True)
class DocsValidationOptions:
    """Validation toggles for docs exports."""

    validation: ExportValidationMode
    macro_requirement: MacroRequirement


@dataclass(frozen=True)
class DocsSelectionOptions:
    """Dataset/schema selection for docs exports."""

    schemas: list[str] | None
    datasets: list[str] | None


@dataclass(frozen=True)
class DocsExecutionOptions:
    """Execution and output options for docs exports."""

    output_format: OutputFormat
    run_mode: DryRunMode
    prereq_mode: PrereqMode


class DocsExportBundleMapping(TypedDict):
    """Typed bundle returned by CLI option normalization."""

    project: ProjectOptions
    backend: BackendOptions
    export_options: DocsExportOptions
    verbose: int


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _parse_env_flag(value: str | None, *, default: bool | None = None) -> bool | None:
    """Parse a boolean-ish environment string.

    Parameters
    ----------
    value
        Environment variable value.
    default
        Default value if parsing fails.

    Returns
    -------
    bool | None
        Parsed boolean or default.
    """
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_graph_backend_config(flags: BackendFlags) -> GraphBackendConfig:
    """Build graph backend configuration from CLI options.

    Parameters
    ----------
    flags
        Backend preferences collected from CLI flags.

    Returns
    -------
    GraphBackendConfig
        Configured graph backend settings.
    """
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if flags.backend == "cpu":
        backend = "cpu"
    elif flags.backend == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=flags.use_gpu,
        backend=backend,
        strict=flags.strict,
    )


def build_graph_feature_flags_from_env() -> GraphFeatureFlags:
    """Construct GraphFeatureFlags from CODEINTEL_* environment variables.

    Returns
    -------
    GraphFeatureFlags
        Feature flags derived from environment variables.
    """
    eager = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_EAGER"))
        if "CODEINTEL_GRAPH_EAGER" in os.environ
        else None
    )
    community_limit = (
        int(os.environ["CODEINTEL_GRAPH_COMMUNITY_LIMIT"])
        if "CODEINTEL_GRAPH_COMMUNITY_LIMIT" in os.environ
        else None
    )
    validation_strict = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_VALIDATION_STRICT"))
        if "CODEINTEL_GRAPH_VALIDATION_STRICT" in os.environ
        else None
    )
    return GraphFeatureFlags(
        eager_hydration=eager,
        community_detection_limit=community_limit,
        validation_strict=validation_strict,
    )


def open_gateway_from_config(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """Open a StorageGateway from CodeIntelConfig.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Opened gateway.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def _project_options(
    repo_selection: RepoSelection,
    storage_selection: StorageSelection,
) -> ProjectOptions:
    """Build project options from repo and storage selections.

    Returns
    -------
    ProjectOptions
        Combined project options.
    """
    return ProjectOptions(
        project_root=repo_selection.project_root,
        repo=repo_selection.repo,
        commit=repo_selection.commit,
        db_path=storage_selection.db_path,
        build_dir=storage_selection.build_dir,
        repo_root=repo_selection.repo_root,
        document_output_dir=storage_selection.document_output_dir,
    )


def _backend_options(
    nx_backend: str,
    nx_gpu_mode: NxGpuMode,
) -> BackendOptions:
    """Build backend options from CLI values.

    Returns
    -------
    BackendOptions
        Backend configuration.
    """
    return BackendOptions(nx_backend=nx_backend, nx_gpu_mode=nx_gpu_mode)


def _docs_validation_options(
    validation: ExportValidationMode,
    macro_requirement: MacroRequirement,
) -> DocsValidationOptions:
    """Build validation options bundle.

    Returns
    -------
    DocsValidationOptions
        Validation configuration.
    """
    return DocsValidationOptions(
        validation=validation,
        macro_requirement=macro_requirement,
    )


def _docs_selection_options(
    schemas: list[str] | None,
    datasets: list[str] | None,
) -> DocsSelectionOptions:
    """Build selection options bundle.

    Returns
    -------
    DocsSelectionOptions
        Selection configuration.
    """
    return DocsSelectionOptions(
        schemas=schemas,
        datasets=datasets,
    )


def _docs_execution_options(
    output_format: OutputFormat,
    run_mode: DryRunMode,
    prereq_mode: PrereqMode | None,
) -> DocsExecutionOptions:
    """Build execution options bundle.

    Returns
    -------
    DocsExecutionOptions
        Execution configuration.
    """
    prereq = prereq_mode or PrereqMode.RUN
    return DocsExecutionOptions(
        output_format=output_format,
        run_mode=run_mode,
        prereq_mode=prereq,
    )


def _docs_export_options(
    validation: DocsValidationOptions,
    selection: DocsSelectionOptions,
    execution: DocsExecutionOptions,
) -> DocsExportOptions:
    """Combine all options into a single bundle.

    Returns
    -------
    DocsExportOptions
        Combined export options.
    """
    return DocsExportOptions(
        validation=validation.validation,
        macro_requirement=validation.macro_requirement,
        datasets=selection.datasets,
        schemas=selection.schemas,
        output_format=execution.output_format,
        run_mode=execution.run_mode,
        prereq_mode=execution.prereq_mode,
    )


# -----------------------------------------------------------------------------
# Coercion Functions
# -----------------------------------------------------------------------------


def _coerce_export_validation(value: object) -> ExportValidationMode:
    """Coerce a value to ExportValidationMode.

    Returns
    -------
    ExportValidationMode
        Coerced validation mode.

    Raises
    ------
    ValueError
        When the value cannot be coerced.
    """
    if isinstance(value, ExportValidationMode):
        return value
    if isinstance(value, str):
        try:
            return ExportValidationMode(value)
        except ValueError as exc:
            message = f"Unknown validation mode: {value}"
            raise ValueError(message) from exc
    return ExportValidationMode.REQUIRED if bool(value) else ExportValidationMode.SKIP


def _coerce_macro_requirement(value: object) -> MacroRequirement:
    """Coerce a value to MacroRequirement.

    Returns
    -------
    MacroRequirement
        Coerced macro requirement.

    Raises
    ------
    ValueError
        When the value cannot be coerced.
    """
    if isinstance(value, MacroRequirement):
        return value
    if isinstance(value, str):
        try:
            return MacroRequirement(value)
        except ValueError as exc:
            message = f"Unknown macro requirement: {value}"
            raise ValueError(message) from exc
    return MacroRequirement.REQUIRE_NORMALIZED if bool(value) else MacroRequirement.ALLOW_PARTIAL


def _coerce_run_mode(value: object) -> DryRunMode:
    """Coerce a value to DryRunMode.

    Returns
    -------
    DryRunMode
        Coerced run mode.

    Raises
    ------
    ValueError
        When the value cannot be coerced.
    """
    if isinstance(value, DryRunMode):
        return value
    if isinstance(value, str):
        try:
            return DryRunMode(value)
        except ValueError as exc:
            message = f"Unknown run mode: {value}"
            raise ValueError(message) from exc
    return DryRunMode.DRY_RUN if bool(value) else DryRunMode.EXECUTE


def _coerce_prereq_mode(value: object) -> PrereqMode:
    """Coerce a value to PrereqMode.

    Returns
    -------
    PrereqMode
        Coerced prereq mode.

    Raises
    ------
    ValueError
        When the value cannot be coerced.
    """
    if isinstance(value, PrereqMode):
        return value
    if isinstance(value, str):
        try:
            return PrereqMode(value)
        except ValueError as exc:
            message = f"Unknown prerequisite mode: {value}"
            raise ValueError(message) from exc
    return PrereqMode.SKIP if bool(value) else PrereqMode.RUN


# -----------------------------------------------------------------------------
# Configuration Resolution
# -----------------------------------------------------------------------------


def _resolve_export_config(
    project: ProjectOptions,
    backend: BackendOptions,
) -> CodeIntelConfig:
    """Resolve export configuration from options.

    Parameters
    ----------
    project
        Project resolution options.
    backend
        Graph backend options.

    Returns
    -------
    CodeIntelConfig
        Resolved configuration.

    Raises
    ------
    ValidationError
        When required repository information is missing.
    """
    try:
        project_root_path = find_project_root(project.project_root)
        project_config = load_project_config(project_root_path)

        resolved = {
            "repo": project.repo or project_config.repo,
            "commit": project.commit or detect_commit(project_root_path),
            "db_path": project.db_path or (project_root_path / project_config.storage.db_path),
            "repo_root": project.repo_root or project_root_path,
            "build_dir": project.build_dir or (project_root_path / ".codeintel"),
        }
    except ProjectNotFoundError:
        if project.repo is None or project.commit is None:
            message = "No codeintel.yaml found. Provide --repo and --commit explicitly."
            raise ValidationError(message) from None
        resolved = {
            "repo": project.repo,
            "commit": project.commit,
            "db_path": project.db_path or Path("build/db/codeintel.duckdb"),
            "repo_root": project.repo_root or Path.cwd(),
            "build_dir": project.build_dir or Path("build"),
        }

    graph_backend = build_graph_backend_config(
        BackendFlags(
            use_gpu=backend.nx_gpu_mode in {NxGpuMode.ENABLED, NxGpuMode.STRICT},
            backend=backend.nx_backend,
            strict=backend.nx_gpu_mode is NxGpuMode.STRICT,
        )
    )
    graph_features = build_graph_feature_flags_from_env()

    paths_cfg = CliPathsInput(
        repo_root=resolved["repo_root"],
        build_dir=resolved["build_dir"],
        db_path=resolved["db_path"],
        document_output_dir=project.document_output_dir,
    )
    repo_cfg = RepoConfig(repo=resolved["repo"], commit=resolved["commit"])
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )


# -----------------------------------------------------------------------------
# Bundle Function
# -----------------------------------------------------------------------------


def bundle_docs_export(cli_kwargs: Mapping[str, object]) -> DocsExportBundleMapping:
    """Bundle CLI arguments into typed options.

    Parameters
    ----------
    cli_kwargs
        Raw CLI keyword arguments.

    Returns
    -------
    DocsExportBundleMapping
        Bundled and validated options.
    """
    project = _project_options(
        RepoSelection(
            project_root=cast("Path | None", cli_kwargs.get("project_root")),
            repo=cast("str | None", cli_kwargs.get("repo")),
            commit=cast("str | None", cli_kwargs.get("commit")),
            repo_root=cast("Path | None", cli_kwargs.get("repo_root")),
        ),
        StorageSelection(
            db_path=cast("Path | None", cli_kwargs.get("db_path")),
            build_dir=cast("Path | None", cli_kwargs.get("build_dir")),
            document_output_dir=cast("Path | None", cli_kwargs.get("document_output_dir")),
        ),
    )
    backend = _backend_options(
        nx_backend=cast("str", cli_kwargs.get("nx_backend", "auto")),
        nx_gpu_mode=cast("NxGpuMode", cli_kwargs.get("nx_gpu_mode", NxGpuMode.DISABLED)),
    )
    validation = _docs_validation_options(
        validation=_coerce_export_validation(cli_kwargs.get("validation")),
        macro_requirement=_coerce_macro_requirement(cli_kwargs.get("macro_requirement")),
    )
    selection = _docs_selection_options(
        schemas=cast("list[str] | None", cli_kwargs.get("schemas")),
        datasets=cast("list[str] | None", cli_kwargs.get("datasets")),
    )
    execution = _docs_execution_options(
        output_format=cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT)),
        run_mode=_coerce_run_mode(cli_kwargs.get("run_mode")),
        prereq_mode=_coerce_prereq_mode(cli_kwargs.get("prereq_mode")),
    )
    export_options = _docs_export_options(validation, selection, execution)
    return {
        "project": project,
        "backend": backend,
        "export_options": export_options,
        "verbose": int(cast("int | str | None", cli_kwargs.get("verbose", 0)) or 0),
    }


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------


def _build_export_resolution(
    cfg: CodeIntelConfig,
    gateway: StorageGateway,
) -> tuple[BuildResolver, ResolutionResult]:
    """Resolve export targets and their dependencies.

    Parameters
    ----------
    cfg
        Resolved configuration.
    gateway
        Open storage gateway.

    Returns
    -------
    tuple[BuildResolver, BuildResolver.Resolution]
        Resolver and resolution for export targets.
    """
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    graph = get_target_graph()
    validator = StateValidator(graph=graph, gateway=gateway, snapshot=snapshot)
    state = validator.validate()

    resolver = BuildResolver(graph=graph, state=state)
    resolution = resolver.resolve(
        goals=["export_jsonl", "export_parquet"],
        force_recompute=None,
    )
    return resolver, resolution


def run_docs_export_via_build_system(
    cfg: CodeIntelConfig,
    *,
    options: DocsExportOptions,
) -> None:
    """Execute docs export using the build system for dependency-aware execution.

    This function uses the build system to ensure all prerequisites are met
    before running the export. It will run any missing analytics/graph targets
    that the export depends on.

    Parameters
    ----------
    cfg
        Resolved configuration.
    options
        Export options bundle.

    Raises
    ------
    ValidationError
        When the build plan fails or execution errors occur.
    """
    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = open_gateway_from_config(cfg, read_only=False)

    if cfg.paths.document_output_dir is None:
        msg = "document_output_dir was not resolved"
        raise ValidationError(msg)

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = BuildPaths.from_layout(
        repo_root=cfg.paths.repo_root,
        overrides=BuildLayoutOptions(
            build_dir=cfg.paths.build_dir,
            db_path=cfg.paths.db_path,
            document_output_dir=cfg.paths.document_output_dir,
        ),
        check_collisions=True,
    )

    graph = get_target_graph()
    state_validator = StateValidator(graph=graph, gateway=gateway, snapshot=snapshot)
    state = state_validator.validate()

    # Resolve what needs to run for export targets
    resolver = BuildResolver(graph=graph, state=state)
    resolution = resolver.resolve(goals=["export_jsonl", "export_parquet"], force_recompute=None)

    if not resolution.to_compute:
        LOG.info("All export targets are up to date.")
        sys.stdout.write("Exports are up to date.\n")
        return

    # Generate and execute the build plan
    plan = PlanGenerator(graph=graph).generate(resolution)
    LOG.info(
        "Build system: %d targets to compute, %d to skip",
        len(resolution.to_compute),
        len(resolution.to_skip),
    )

    executor = BuildExecutor(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=cfg.tools,
        graph=graph,
    )
    executor.export_options = ExportCallOptions(
        validate_exports=options.validation is ExportValidationMode.REQUIRED,
        schemas=options.schemas,
        datasets=options.datasets,
        require_normalized_macros=options.macro_requirement is MacroRequirement.REQUIRE_NORMALIZED,
    )
    result = executor.execute(plan)

    if result.status == "failed":
        message = f"Export failed: {result.error_summary}"
        raise ValidationError(message)

    LOG.info("Export complete via build system.")
    if options.output_format is OutputFormat.JSON:
        payload = {
            "status": "ok",
            "validation": options.validation.value,
            "macro_requirement": options.macro_requirement.value,
            "datasets": options.datasets,
            "schemas": options.schemas,
            "mode": "build_system",
        }
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
    else:
        sys.stdout.write("Export complete.\n")


def run_docs_export(
    cfg: CodeIntelConfig,
    options: DocsExportOptions,
    validator: Callable[[StorageGateway], None],
    export_runner: ExportRunner,
) -> None:
    """Execute the docs export with provided configuration and callbacks (legacy).

    Parameters
    ----------
    cfg
        Resolved configuration.
    options
        Export options bundle.
    validator
        Dataset validation callback.
    export_runner
        Export runner callback.

    Raises
    ------
    ValidationError
        When required paths are missing.
    DocsValidationError
        When dataset validation or export validation fails.
    """
    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = open_gateway_from_config(cfg, read_only=True)
    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        msg = "document_output_dir was not resolved"
        raise ValidationError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    schemas_list = list(options.schemas) if options.schemas else None
    datasets_list = list(options.datasets) if options.datasets else None

    if options.run_mode is DryRunMode.DRY_RUN:
        payload = {
            "output_dir": str(out_dir),
            "schemas": schemas_list,
            "datasets": datasets_list,
            "validation": options.validation.value,
            "macro_requirement": options.macro_requirement.value,
            "mode": "dry_run",
        }
        if options.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload))
            sys.stdout.write("\n")
        else:
            sys.stdout.write("Dry run: exports planned, no files written.\n")
        return

    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=options.validation is ExportValidationMode.REQUIRED,
                    schemas=schemas_list,
                    datasets=datasets_list,
                    require_normalized_macros=options.macro_requirement
                    is MacroRequirement.REQUIRE_NORMALIZED,
                ),
                validator=validator,
            ),
        )
    except DocsValidationError:
        raise
    except ValueError as exc:
        raise DocsValidationError(str(exc)) from exc
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        message = str(exc.detail.detail or exc.detail.title or "Export validation failed")
        raise DocsValidationError(message) from exc

    LOG.info("Export complete.")
    if options.output_format is OutputFormat.JSON:
        payload = {
            "status": "ok",
            "validation": options.validation.value,
            "macro_requirement": options.macro_requirement.value,
            "datasets": datasets_list,
            "schemas": schemas_list,
            "mode": "direct",
        }
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
    else:
        sys.stdout.write("Export complete.\n")


def docs_export_handler(
    project: ProjectOptions,
    backend: BackendOptions,
    export_options: DocsExportOptions,
    verbose: int,
) -> None:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    By default, uses the build system for dependency-aware export, which
    ensures all prerequisites (analytics, profiles) are computed first.

    Use --skip-prereqs to skip prerequisite computation if analytics are
    already complete.

    Parameters
    ----------
    project
        Project resolution options.
    backend
        Graph backend options.
    export_options
        Export configuration options.
    verbose
        Verbosity level.

    Raises
    ------
    DocsValidationError
        When export validation fails.
    """
    setup_logging(verbose)

    project_opts = project
    backend_opts = backend
    export_opts = export_options

    cfg = _resolve_export_config(project_opts, backend_opts)

    if export_opts.run_mode is DryRunMode.DRY_RUN:
        payload = {
            "mode": "dry_run",
            "prereq_mode": export_opts.prereq_mode.value,
            "validation": export_opts.validation.value,
            "macro_requirement": export_opts.macro_requirement.value,
            "datasets": export_opts.datasets,
            "schemas": export_opts.schemas,
            "backend": backend_opts.nx_backend,
            "gpu_mode": backend_opts.nx_gpu_mode.value,
        }
        if export_opts.output_format is OutputFormat.JSON:
            sys.stdout.write(json.dumps(payload))
            sys.stdout.write("\n")
        else:
            sys.stdout.write("Dry run: exports planned, no actions taken.\n")
        return

    try:
        if export_opts.prereq_mode is PrereqMode.SKIP:
            # Direct export without build system (legacy behavior)
            run_docs_export(
                cfg=cfg,
                options=export_opts,
                validator=validate_dataset_registry,
                export_runner=run_validated_exports,
            )
        else:
            # Use build system with all options
            run_docs_export_via_build_system(
                cfg,
                options=export_opts,
            )
    except DocsValidationError as exc:
        sys.stderr.write(f"Validation failed: {exc}\n")
        raise


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DocsExportResult:
    """Result from docs export operation.

    Parameters
    ----------
    status
        Export status (ok, dry_run, failed).
    validation
        Validation mode used.
    macro_requirement
        Macro requirement mode used.
    datasets
        Datasets exported (or None for all).
    schemas
        Schemas exported (or None for all).
    mode
        Execution mode (build_system, direct, dry_run).
    """

    status: str
    validation: str
    macro_requirement: str
    datasets: list[str] | None
    schemas: list[str] | None
    mode: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "status": self.status,
            "validation": self.validation,
            "macro_requirement": self.macro_requirement,
            "datasets": self.datasets,
            "schemas": self.schemas,
            "mode": self.mode,
        }


# -----------------------------------------------------------------------------
# ExecutionContext-based Handler
# -----------------------------------------------------------------------------


def _build_export_options_from_ctx(ctx: ExecutionContext) -> tuple[ProjectOptions, BackendOptions]:
    """Build project and backend options from execution context.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    tuple[ProjectOptions, BackendOptions]
        Project and backend options.
    """
    project_root_raw = ctx.params.get("project_root")
    repo_root_raw = ctx.params.get("repo_root")
    db_path_raw = ctx.params.get("db_path")
    build_dir_raw = ctx.params.get("build_dir")
    doc_output_raw = ctx.params.get("document_output_dir")

    project = ProjectOptions(
        project_root=Path(project_root_raw) if project_root_raw else None,
        repo=ctx.get_str_param("repo"),
        commit=ctx.get_str_param("commit"),
        db_path=Path(db_path_raw) if db_path_raw else None,
        build_dir=Path(build_dir_raw) if build_dir_raw else None,
        repo_root=Path(repo_root_raw) if repo_root_raw else None,
        document_output_dir=Path(doc_output_raw) if doc_output_raw else None,
    )

    nx_backend = ctx.get_str_param("nx_backend", "auto") or "auto"
    gpu_mode_raw = ctx.params.get("nx_gpu_mode", NxGpuMode.DISABLED)
    if isinstance(gpu_mode_raw, NxGpuMode):
        nx_gpu_mode = gpu_mode_raw
    else:
        nx_gpu_mode = NxGpuMode(str(gpu_mode_raw))

    backend = BackendOptions(nx_backend=nx_backend, nx_gpu_mode=nx_gpu_mode)
    return project, backend


def _build_docs_export_options_from_ctx(ctx: ExecutionContext) -> DocsExportOptions:
    """Build docs export options from execution context.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    DocsExportOptions
        Export options.
    """
    validation_raw = ctx.params.get("validation_mode", ExportValidationMode.SKIP)
    if isinstance(validation_raw, ExportValidationMode):
        validation = validation_raw
    else:
        validation = ExportValidationMode(str(validation_raw))

    macro_raw = ctx.params.get("macro_requirement", MacroRequirement.ALLOW_PARTIAL)
    if isinstance(macro_raw, MacroRequirement):
        macro_requirement = macro_raw
    else:
        macro_requirement = MacroRequirement(str(macro_raw))

    run_mode_raw = ctx.params.get("run_mode", DryRunMode.EXECUTE)
    if isinstance(run_mode_raw, DryRunMode):
        run_mode = run_mode_raw
    else:
        run_mode = DryRunMode(str(run_mode_raw))

    prereq_mode_raw = ctx.params.get("prereq_mode", PrereqMode.RUN)
    if isinstance(prereq_mode_raw, PrereqMode):
        prereq_mode = prereq_mode_raw
    else:
        prereq_mode = PrereqMode(str(prereq_mode_raw))

    datasets_raw = ctx.params.get("datasets")
    schemas_raw = ctx.params.get("schemas")

    return DocsExportOptions(
        validation=validation,
        macro_requirement=macro_requirement,
        datasets=list(datasets_raw) if datasets_raw else None,
        schemas=list(schemas_raw) if schemas_raw else None,
        output_format=OutputFormat(ctx.output_format.value),
        run_mode=run_mode,
        prereq_mode=prereq_mode,
    )


def docs_export_ctx(ctx: ExecutionContext) -> CliResult[DocsExportResult]:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    By default, uses the build system for dependency-aware export, which
    ensures all prerequisites (analytics, profiles) are computed first.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - repo: Repository slug.
        - commit: Commit SHA.
        - db_path: Database path.
        - build_dir: Build directory.
        - repo_root: Repository root.
        - document_output_dir: Document output directory.
        - nx_backend: NetworkX backend.
        - nx_gpu_mode: GPU mode (disabled, enabled, strict).
        - validation: Validation mode (required, skip).
        - macro_requirement: Macro requirement (require_normalized, allow_partial).
        - datasets: Dataset filter list.
        - schemas: Schema filter list.
        - run_mode: Run mode (execute, dry_run).
        - prereq_mode: Prerequisite mode (run, skip).

    Returns
    -------
    CliResult[DocsExportResult]
        Export result.

    Raises
    ------
    RuntimeError
        If export fails.
    """
    setup_logging(ctx.verbosity)

    project, backend = _build_export_options_from_ctx(ctx)
    export_opts = _build_docs_export_options_from_ctx(ctx)

    try:
        cfg = _resolve_export_config(project, backend)
    except ValidationError as exc:
        msg = f"Configuration resolution failed: {exc}"
        raise RuntimeError(msg) from exc

    if export_opts.run_mode is DryRunMode.DRY_RUN:
        return CliResult.ok(
            DocsExportResult(
                status="dry_run",
                validation=export_opts.validation.value,
                macro_requirement=export_opts.macro_requirement.value,
                datasets=export_opts.datasets,
                schemas=export_opts.schemas,
                mode="dry_run",
            )
        )

    try:
        if export_opts.prereq_mode is PrereqMode.SKIP:
            run_docs_export(
                cfg=cfg,
                options=export_opts,
                validator=validate_dataset_registry,
                export_runner=run_validated_exports,
            )
            mode = "direct"
        else:
            run_docs_export_via_build_system(cfg, options=export_opts)
            mode = "build_system"
    except (DocsValidationError, ValidationError) as exc:
        msg = f"Export failed: {exc}"
        raise RuntimeError(msg) from exc

    return CliResult.ok(
        DocsExportResult(
            status="ok",
            validation=export_opts.validation.value,
            macro_requirement=export_opts.macro_requirement.value,
            datasets=export_opts.datasets,
            schemas=export_opts.schemas,
            mode=mode,
        )
    )


__all__ = [
    "BackendFlags",
    "BackendOptions",
    "DocsExecutionOptions",
    "DocsExportBundleMapping",
    "DocsExportOptions",
    "DocsExportResult",
    "DocsSelectionOptions",
    "DocsValidationOptions",
    "DryRunMode",
    "ExportValidationMode",
    "MacroRequirement",
    "NxGpuMode",
    "OutputFormat",
    "PrereqMode",
    "ProjectOptions",
    "RepoSelection",
    "StorageSelection",
    "build_graph_backend_config",
    "build_graph_feature_flags_from_env",
    "bundle_docs_export",
    "docs_export_ctx",
    "docs_export_handler",
    "open_gateway_from_config",
    "run_docs_export",
    "run_docs_export_via_build_system",
    "setup_logging",
]
