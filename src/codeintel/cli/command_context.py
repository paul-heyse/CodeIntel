"""Unified command context manager for Cyclopts commands.

This module provides command_context(), a context manager that standardizes
infrastructure for all Cyclopts commands:

- Configuration loading via ConfigService
- Runtime resolution via RuntimeResolver
- Logging setup based on verbosity
- Unified rendering via UnifiedRenderer
- Automatic resource cleanup
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.cli_types import BackendFlags, OutputFormat
from codeintel.cli.config import build_config_from_options, load_config
from codeintel.cli.handlers.base import setup_logging
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.cli.rendering.service import UnifiedRenderer
from codeintel.cli.rendering.types import OutputFormat as RenderOutputFormat
from codeintel.cli.rendering.types import RenderContext
from codeintel.cli.resolution.params import RuntimeParams
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.models import CliPathsInput
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig

if TYPE_CHECKING:
    from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI


class CommandContextError(Exception):
    """Raised when command context creation fails."""


def _resolve_output_format(
    output_cli: OutputFormatCLI | None,
    default: OutputFormat = OutputFormat.TEXT,
) -> OutputFormat:
    """Resolve output format from CLI options.

    Parameters
    ----------
    output_cli
        Optional OutputFormatCLI with output format flags.
    default
        Default output format.

    Returns
    -------
    OutputFormat
        Resolved output format.
    """
    if output_cli is None:
        return default
    if output_cli.json:
        return OutputFormat.JSON
    if output_cli.output_format is not None:
        return output_cli.output_format
    return default


def _build_runtime_params(runtime_cli: RuntimeCLI | None) -> RuntimeParams:
    """Build RuntimeParams from RuntimeCLI.

    Parameters
    ----------
    runtime_cli
        Optional RuntimeCLI with project/commit/db flags.

    Returns
    -------
    RuntimeParams
        Canonical runtime parameters.
    """
    if runtime_cli is None:
        return RuntimeParams()
    return RuntimeParams(
        project_root=runtime_cli.project_root,
        repo=runtime_cli.repo,
        commit=runtime_cli.commit,
        db_path=runtime_cli.db_path,
        build_dir=runtime_cli.build_dir,
        repo_root=runtime_cli.repo_root,
        document_output_dir=runtime_cli.document_output_dir,
    )


@contextmanager
def command_context(
    operation_id: str,
    runtime_cli: RuntimeCLI | None = None,
    output_cli: OutputFormatCLI | None = None,
    *,
    params: dict[str, object] | None = None,
    require_runtime: bool = True,
) -> Iterator[tuple[EnhancedHandlerContext, UnifiedRenderer]]:
    """Create unified context for Cyclopts command execution.

    This context manager provides:

    1. Configuration loading via ConfigService
    2. Runtime resolution via RuntimeResolver (when required)
    3. Logging setup based on verbosity
    4. Renderer creation for output
    5. Automatic resource cleanup

    Commands using this pattern delegate all infrastructure to this helper,
    keeping command classes focused on parameter extraction and delegation.

    Parameters
    ----------
    operation_id
        Unique identifier for the operation (e.g., "ide.hints").
    runtime_cli
        Optional RuntimeCLI with project/commit/db flags.
    output_cli
        Optional OutputFormatCLI with output format flags.
    params
        Additional operation-specific parameters.
    require_runtime
        If True (default), runtime resolution is required and will raise if
        no project config is found and no explicit repo/commit are provided.
        If False, a stub runtime is created for commands that don't need
        project resources.

    Yields
    ------
    tuple[EnhancedHandlerContext, UnifiedRenderer]
        Handler context and renderer for command execution.

    Notes
    -----
    If runtime resolution fails due to missing project file and missing required
    CLI parameters (--repo and --commit), a ``CommandContextError`` will be raised
    by the underlying ``_resolve_runtime_from_params`` helper.

    Examples
    --------
    >>> @dataclass
    ... class MyCommand:
    ...     runtime: RuntimeCLI = field(default_factory=RuntimeCLI)
    ...     output: OutputFormatCLI = field(default_factory=OutputFormatCLI)
    ...
    ...     def __call__(self) -> None:
    ...         with command_context("my.command", self.runtime, self.output) as (ctx, renderer):
    ...             result = my_handler(ctx)
    ...             renderer.render_result(result)
    """
    # Resolve params - need to handle None for runtime_cli
    resolved_params = params or {}

    # Extract verbosity (default to 0 if no runtime_cli)
    verbosity = runtime_cli.verbose if runtime_cli is not None else 0

    # Resolve output format
    output_format = _resolve_output_format(output_cli, default=OutputFormat.TEXT)

    # Map OutputFormat (cli_types) to OutputFormat (rendering.types)
    render_format = RenderOutputFormat(output_format.value)

    # Load configuration
    config = load_config(validate=False)

    # Setup logging
    setup_logging(verbosity, config=config)

    # Convert to RuntimeParams for resolution
    runtime_params = _build_runtime_params(runtime_cli)

    # Resolve runtime (or create stub if not required)
    if require_runtime:
        runtime = _resolve_runtime_from_params(runtime_params)
    else:
        runtime = _create_stub_runtime()

    # Create render context and renderer
    render_ctx = RenderContext.auto_detect(format_override=render_format)
    renderer = UnifiedRenderer(render_ctx)

    # Combine params with CLI values (runtime_params serve as defaults, resolved_params override)
    combined_params: dict[str, object] = {
        "project_root": runtime_params.project_root,
        "repo": runtime_params.repo,
        "commit": runtime_params.commit,
        "db_path": runtime_params.db_path,
        "build_dir": runtime_params.build_dir,
        "repo_root": runtime_params.repo_root,
        "document_output_dir": runtime_params.document_output_dir,
        **resolved_params,  # Command-specific params override runtime defaults
    }

    # Create enhanced context
    ctx = EnhancedHandlerContext(
        config=config,
        runtime=runtime,
        params=combined_params,
        verbosity=verbosity,
        _operation_name=operation_id,
    )

    try:
        yield ctx, renderer
    finally:
        ctx.close()


def _create_stub_runtime() -> ResolvedRuntime:
    """Create a stub runtime for commands that don't need project resources.

    Returns
    -------
    ResolvedRuntime
        Stub runtime with placeholder values.
    """
    cwd = Path.cwd()
    stub_project = ProjectConfig(repo="stub/repo")
    stub_snapshot = SnapshotRef(
        repo="stub/repo",
        commit="0000000",
        repo_root=cwd,
    )
    stub_paths = CliPathsInput(
        repo_root=cwd,
        build_dir=cwd / "build",
        db_path=cwd / "build" / "db" / "stub.duckdb",
    )

    # Build minimal config
    stub_cfg = build_config_from_options(
        repo="stub/repo",
        commit="0000000",
        paths_cfg=stub_paths,
        backend=BackendFlags(),
    )

    stub_serving = ServingConfig(
        mode="local_db",
        repo_root=cwd,
        repo="stub/repo",
        commit="0000000",
        db_path=cwd / "build" / "db" / "stub.duckdb",
        read_only=True,
    )

    return ResolvedRuntime(
        root=cwd,
        project=stub_project,
        snapshot=stub_snapshot,
        paths=stub_cfg.build_paths,
        config=stub_cfg,
        serving=stub_serving,
    )


def _resolve_runtime_from_params(params: RuntimeParams) -> ResolvedRuntime:
    """Resolve runtime from RuntimeParams.

    Parameters
    ----------
    params
        RuntimeParams instance.

    Returns
    -------
    ResolvedRuntime
        Fully resolved runtime.

    Raises
    ------
    CommandContextError
        If resolution fails.
    """
    # Try project file discovery first
    try:
        project_runtime = build_project_runtime(params.project_root)
        return ResolvedRuntime(
            root=project_runtime.root,
            project=project_runtime.project,
            snapshot=project_runtime.snapshot,
            paths=project_runtime.paths,
            config=project_runtime.cfg,
            serving=project_runtime.serving,
        )
    except ProjectNotFoundError:
        pass

    # Fall back to explicit params
    if params.repo is None or params.commit is None:
        msg = (
            "No codeintel.yaml found. Provide --repo and --commit explicitly, "
            "or create a project file."
        )
        raise CommandContextError(msg)

    resolved_repo_root = params.repo_root or Path.cwd()
    resolved_db_path = params.db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = params.build_dir or Path("build")

    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=params.document_output_dir,
    )

    # Build config using common handler
    cfg = build_config_from_options(
        repo=params.repo,
        commit=params.commit,
        paths_cfg=paths_cfg,
        backend=BackendFlags(),
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    project = ProjectConfig(
        repo=cfg.repo.repo,
        storage=StorageProjectConfig(db_path=paths.db_path),
    )

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    return ResolvedRuntime(
        root=resolved_repo_root,
        project=project,
        snapshot=snapshot,
        paths=paths,
        config=cfg,
        serving=serving,
    )


__all__ = [
    "CommandContextError",
    "command_context",
]
