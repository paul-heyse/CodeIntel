"""Runtime resolution - single source of truth for project/runtime resolution.

This module consolidates all runtime resolution logic with a single,
unified implementation that handles:
1. Project file discovery (codeintel.yaml)
2. Fallback to explicit CLI parameters
3. Construction of ResolvedRuntime with all necessary configuration

The primary API is `resolve_from_params()` which takes a params dict directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    StorageProjectConfig,
    detect_commit,
    find_project_root,
    load_project_config,
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

LOG = logging.getLogger(__name__)


def _to_path_or_none(value: object) -> Path | None:
    """Convert value to Path or None.

    Parameters
    ----------
    value
        Value to convert (may be str, Path, or None).

    Returns
    -------
    Path | None
        Converted path or None.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _to_path_with_default(value: object, default: Path) -> Path:
    """Convert value to Path with default.

    Parameters
    ----------
    value
        Value to convert (may be str, Path, or None).
    default
        Default path if value is None.

    Returns
    -------
    Path
        Converted path or default.
    """
    if value is None:
        return default
    if isinstance(value, Path):
        return value
    return Path(str(value))


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


def resolve_from_params(
    params: Mapping[str, object] | Mapping[str, str],
    *,
    allow_fallback: bool = True,
) -> ResolvedRuntime:
    """Resolve runtime from params dict directly.

    This is the primary API for runtime resolution. It tries project file
    discovery first, then falls back to explicit parameters.

    Parameters
    ----------
    params
        Parameters dict with keys like project_root, repo, commit, db_path, etc.
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

    Examples
    --------
    >>> runtime = resolve_from_params({"project_root": Path(".")})  # doctest: +SKIP
    >>> runtime.db_path  # doctest: +SKIP
    PosixPath('build/db/codeintel.duckdb')
    """
    project_root_raw = params.get("project_root")
    project_root = _to_path_or_none(project_root_raw)

    # Try project file discovery first
    try:
        return _resolve_from_project(project_root)
    except ProjectNotFoundError as exc:
        if not allow_fallback:
            raise ResolutionError(_MSG_NO_PROJECT_NO_FALLBACK) from exc

    # Fall back to explicit params
    return _resolve_from_params_dict(params)


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
    This function propagates ProjectNotFoundError from find_project_root
    when no project file is found.
    """
    # Discover project root and load config
    # May raise ProjectNotFoundError if no project file found
    resolved_root = find_project_root(project_root)
    project = load_project_config(resolved_root)

    commit = detect_commit(resolved_root)
    repo_cfg = RepoConfig(repo=project.repo, commit=commit)

    db_path = resolved_root / project.storage.db_path
    paths_cfg = CliPathsInput(
        repo_root=resolved_root,
        build_dir=resolved_root / ".codeintel",
        db_path=db_path,
        document_output_dir=None,
    )

    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=None,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths

    # Ensure database directory exists
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    LOG.info(
        "project.runtime repo=%s commit=%s root=%s db=%s",
        project.repo,
        commit,
        resolved_root,
        paths.db_path,
    )

    return ResolvedRuntime(
        root=resolved_root,
        project=project,
        snapshot=snapshot,
        paths=paths,
        config=cfg,
        serving=serving,
    )


def _resolve_from_params_dict(params: Mapping[str, object] | Mapping[str, str]) -> ResolvedRuntime:
    """Resolve from explicit CLI parameters.

    Parameters
    ----------
    params
        Parameters dict with keys like repo, commit, db_path, etc.

    Returns
    -------
    ResolvedRuntime
        Runtime resolved from explicit parameters.

    Notes
    -----
    Propagates ResolutionError from _extract_required_params_dict if required
    parameters (repo, commit) are missing.
    """
    repo, commit = _extract_required_params_dict(params)
    repo_root = _to_path_with_default(params.get("repo_root"), Path.cwd())

    # Normalize other path params from string or Path
    db_path = _to_path_with_default(params.get("db_path"), Path("build/db/codeintel.duckdb"))
    build_dir = _to_path_with_default(params.get("build_dir"), Path("build"))
    document_output_dir = _to_path_or_none(params.get("document_output_dir"))

    # Build configuration from params
    use_gpu_raw = params.get("use_gpu", False)
    use_gpu = bool(use_gpu_raw) if use_gpu_raw is not None else False
    config = _build_config(
        _ConfigParams(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            db_path=db_path,
            build_dir=build_dir,
            document_output_dir=document_output_dir,
            use_gpu=use_gpu,
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


def _extract_required_params_dict(
    params: Mapping[str, object] | Mapping[str, str],
) -> tuple[str, str]:
    """Extract and validate required repo and commit params.

    Parameters
    ----------
    params
        Parameters dict.

    Returns
    -------
    tuple[str, str]
        Tuple of (repo, commit) strings.

    Raises
    ------
    ResolutionError
        If required parameters are missing.
    """
    repo = params.get("repo")
    commit = params.get("commit")

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


__all__ = [
    "resolve_from_params",
]
