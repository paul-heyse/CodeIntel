"""Runtime resolution - single source of truth for project/runtime resolution.

This module consolidates all runtime resolution logic previously scattered across:
- cyclopts_common.py:build_runtime_from_cli
- common_handlers.py:build_runtime_from_cli
- datasets_handlers.py:build_runtime_from_cli
- subsystem_handlers.py:build_runtime_from_cli
- ide_handlers.py:build_runtime_from_cli
- build_handlers.py:build_runtime_from_cli
- ops_handlers.py:_build_runtime_or_error

The RuntimeResolver provides a single, unified implementation that handles:
1. Project file discovery (codeintel.yaml)
2. Fallback to explicit CLI parameters
3. Construction of ResolvedRuntime with all necessary configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.primitives import (
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.config.serving_models import ServingConfig

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


# Error message constants
_MSG_NO_PROJECT_NO_FALLBACK = "No codeintel.yaml found and fallback disabled"
_MSG_MISSING_PARAMS = "No codeintel.yaml found. Provide --repo and --commit explicitly"


@dataclass(frozen=True)
class _ConfigParams:
    """Internal dataclass for config building parameters."""

    repo: str
    commit: str
    repo_root: Path
    db_path: Path
    build_dir: Path
    document_output_dir: Path | None
    use_gpu: bool


class RuntimeResolver:
    """Resolve project runtime from ExecutionContext parameters.

    Resolution follows this order:
    1. Try project file discovery (codeintel.yaml) from project_root
    2. Fall back to explicit parameters (repo, commit, db_path, etc.)

    The resolver is stateless - all state lives in the ExecutionContext.

    Examples
    --------
    >>> resolver = RuntimeResolver()
    >>> runtime = resolver.resolve(ctx)  # doctest: +SKIP
    >>> runtime.db_path  # doctest: +SKIP
    PosixPath('build/db/codeintel.duckdb')
    """

    @staticmethod
    def resolve(
        ctx: ExecutionContext,
        *,
        allow_fallback: bool = True,
    ) -> ResolvedRuntime:
        """Resolve runtime from context parameters.

        Parameters
        ----------
        ctx
            Execution context with params.
        allow_fallback
            If True, attempt fallback to explicit params when no project file.
            If False, raise immediately when project file not found.

        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime.

        Raises
        ------
        ResolutionError
            If resolution fails (no project file and missing required params).
        """
        project_root = ctx.params.get("project_root")

        # Try project file discovery first
        try:
            return _resolve_from_project(project_root)
        except ProjectNotFoundError as exc:
            if not allow_fallback:
                raise ResolutionError(_MSG_NO_PROJECT_NO_FALLBACK) from exc

        # Fall back to explicit params
        return _resolve_from_params(ctx)


def _resolve_from_project(project_root: Path | None) -> ResolvedRuntime:
    """Resolve from project file (codeintel.yaml).

    Parameters
    ----------
    project_root
        Optional explicit project root. If None, searches from cwd.

    Returns
    -------
    ResolvedRuntime
        Runtime resolved from project file.

    Notes
    -----
    This function propagates ProjectNotFoundError from build_project_runtime
    when no project file is found.
    """
    # build_project_runtime handles discovery and construction
    # May raise ProjectNotFoundError if no project file found
    project_runtime = build_project_runtime(project_root)

    return ResolvedRuntime(
        root=project_runtime.root,
        project=project_runtime.project,
        snapshot=project_runtime.snapshot,
        paths=project_runtime.paths,
        config=project_runtime.cfg,
        serving=project_runtime.serving,
    )


def _resolve_from_params(ctx: ExecutionContext) -> ResolvedRuntime:
    """Resolve from explicit CLI parameters.

    Parameters
    ----------
    ctx
        Execution context with params.

    Returns
    -------
    ResolvedRuntime
        Runtime resolved from explicit parameters.

    Notes
    -----
    Propagates ResolutionError from _extract_required_params if required
    parameters (repo, commit) are missing.
    """
    repo, commit = _extract_required_params(ctx)
    repo_root_raw = ctx.params.get("repo_root")
    repo_root = Path(repo_root_raw) if repo_root_raw else Path.cwd()

    # Normalize other path params from string or Path
    db_path_raw = ctx.params.get("db_path")
    db_path = Path(db_path_raw) if db_path_raw else Path("build/db/codeintel.duckdb")

    build_dir_raw = ctx.params.get("build_dir")
    build_dir = Path(build_dir_raw) if build_dir_raw else Path("build")

    doc_output_raw = ctx.params.get("document_output_dir")
    document_output_dir = Path(doc_output_raw) if doc_output_raw else None

    # Build configuration from context params
    config = _build_config(
        _ConfigParams(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            db_path=db_path,
            build_dir=build_dir,
            document_output_dir=document_output_dir,
            use_gpu=ctx.params.get("use_gpu", False),
        )
    )

    # Ensure database directory exists
    config.build_paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    return ResolvedRuntime(
        root=repo_root,
        project=ProjectConfig(
            repo=repo,
            storage=StorageProjectConfig(db_path=config.build_paths.db_path),
        ),
        snapshot=SnapshotRef(repo=repo, commit=commit, repo_root=repo_root),
        paths=config.build_paths,
        config=config,
        serving=ServingConfig(
            mode="local_db",
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            db_path=config.build_paths.db_path,
            read_only=True,
        ),
    )


def _extract_required_params(ctx: ExecutionContext) -> tuple[str, str]:
    """Extract and validate required repo and commit params.

    Parameters
    ----------
    ctx
        Execution context with params.

    Returns
    -------
    tuple[str, str]
        Tuple of (repo, commit) strings.

    Raises
    ------
    ResolutionError
        If required parameters are missing.
    """
    repo = ctx.params.get("repo")
    commit = ctx.params.get("commit")

    missing: list[str] = []
    if repo is None:
        missing.append("repo")
    if commit is None:
        missing.append("commit")

    if missing:
        raise ResolutionError(_MSG_MISSING_PARAMS, missing_params=missing)

    return str(repo), str(commit)


def _build_config(params: _ConfigParams) -> CodeIntelConfig:
    """Build CodeIntelConfig from parameters.

    Parameters
    ----------
    params
        Configuration parameters dataclass.

    Returns
    -------
    CodeIntelConfig
        Constructed configuration.
    """
    paths_cfg = CliPathsInput(
        repo_root=params.repo_root,
        build_dir=params.build_dir,
        db_path=params.db_path,
        document_output_dir=params.document_output_dir,
    )

    repo_cfg = RepoConfig(repo=params.repo, commit=params.commit)

    backend = GraphBackendConfig(use_gpu=params.use_gpu)

    options = CliConfigOptions(
        graph_backend=backend,
        graph_features=GraphFeatureFlags(),
    )

    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=options,
    )


def resolve_runtime(
    ctx: ExecutionContext,
    *,
    allow_fallback: bool = True,
) -> ResolvedRuntime:
    """Resolve runtime from context (module-level convenience function).

    Parameters
    ----------
    ctx
        Execution context with params.
    allow_fallback
        If True, attempt fallback to explicit params.

    Returns
    -------
    ResolvedRuntime
        Fully resolved runtime.

    Notes
    -----
    Propagates ResolutionError from RuntimeResolver.resolve if resolution fails
    due to missing project file and missing required CLI parameters.
    """
    return RuntimeResolver.resolve(ctx, allow_fallback=allow_fallback)


__all__ = [
    "RuntimeResolver",
    "resolve_runtime",
]
