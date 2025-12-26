"""Runtime resolution - single source of truth for project/runtime resolution.

This module consolidates all runtime resolution logic with a single,
unified implementation that handles:
1. Project file discovery (codeintel.yaml)
2. Fallback to explicit CLI parameters
3. Construction of ResolvedRuntime with all necessary configuration

The primary API is `resolve_from_params()` which takes a params dict directly.
"""

from __future__ import annotations

import configparser
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from codeintel.cli.project._project import (
    ProjectConfig,
    ProjectNotFoundError,
    StorageProjectConfig,
    detect_commit,
    find_project_root,
    load_project_config,
)
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.models import (
    CliConfigOptions,
    CliPathsInput,
    CodeIntelConfig,
    RepoConfig,
    ToolsConfig,
)
from codeintel.config.primitives import (
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.core.runtime.loader import RuntimeInputs, build_runtime_primitives
from codeintel.serving.config import ServingConfig

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FallbackSelection:
    source: Literal["git", "src"]
    repo_root: Path


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


def _normalize_repo_slug(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.endswith(".git"):
        stripped = stripped[:-4]
    stripped = stripped.lstrip("/")
    return stripped or None


def _repo_from_remote_url(url: str) -> str | None:
    if "://" in url:
        parsed = urlparse(url)
        return _normalize_repo_slug(parsed.path)

    if ":" in url:
        prefix, _, path = url.partition(":")
        if "/" not in prefix and "\\" not in prefix:
            return _normalize_repo_slug(path)

    return _normalize_repo_slug(url)


def _resolve_git_dir(repo_root: Path) -> Path | None:
    git_path = repo_root / ".git"
    if git_path.is_dir():
        return git_path
    if not git_path.is_file():
        return None

    try:
        content = git_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    prefix = "gitdir:"
    if not content.startswith(prefix):
        return None

    git_dir = Path(content[len(prefix) :].strip())
    if git_dir.is_absolute():
        return git_dir
    return (repo_root / git_dir).resolve()


def _find_git_root(start: Path) -> Path | None:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        git_path = candidate / ".git"
        if git_path.is_dir() or git_path.is_file():
            return candidate
    return None


def _select_fallback_repo_root(base: Path) -> _FallbackSelection | None:
    git_root = _find_git_root(base)
    if git_root is not None:
        return _FallbackSelection(source="git", repo_root=git_root)

    if base.name == "src" and base.is_dir():
        return _FallbackSelection(source="src", repo_root=base.resolve())

    src_root = base / "src"
    if src_root.is_dir():
        return _FallbackSelection(source="src", repo_root=src_root.resolve())

    return None


def _infer_repo_from_git_remote(repo_root: Path) -> str | None:
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None

    git_config = git_dir / "config"
    if not git_config.is_file():
        return None

    parser = configparser.ConfigParser()
    try:
        parser.read(git_config, encoding="utf-8")
    except (OSError, configparser.Error):
        return None

    origin_section = 'remote "origin"'
    url = None
    if parser.has_section(origin_section):
        url = parser.get(origin_section, "url", fallback=None)
    if not url:
        for section_name in parser.sections():
            if section_name.startswith('remote "'):
                url = parser.get(section_name, "url", fallback=None)
                if url:
                    break

    if not url:
        return None

    return _repo_from_remote_url(url)


def _apply_default_scip_project_name(
    config: CodeIntelConfig,
    repo: str,
) -> CodeIntelConfig:
    default_name = ToolsConfig.default().scip_project_name
    if config.tools.scip_project_name != default_name:
        return config
    tools = config.tools.model_copy(update={"scip_project_name": repo})
    return config.model_copy(update={"tools": tools})


_MSG_NO_PROJECT_NO_FALLBACK = "No codeintel.yaml found and fallback disabled"
_MSG_NO_PROJECT_NO_SOURCE = "No codeintel.yaml found and no git repo or src/ directory detected."
_MSG_MISSING_PARAMS = (
    "No codeintel.yaml found. Provide --repo explicitly or set project.repo/project.name."
)


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
    allow_fallback: bool | None = None,
) -> ResolvedRuntime:
    """Resolve runtime from params dict directly.

    This is the primary API for runtime resolution. It tries project file
    discovery first, then falls back to explicit parameters.

    Parameters
    ----------
    params
        Parameters dict with keys like project_root, repo, commit, db_path, etc.
    allow_fallback
        When True, attempt fallback to explicit params when no project file.
        When False, raise immediately when project file is missing.
        When None, fallback is enabled only if explicit repo/commit/db_path params are set.

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
    >>> runtime = resolve_from_params({"project_root": Path(".")})
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    """
    project_root_raw = params.get("project_root")
    project_root = _to_path_or_none(project_root_raw)
    fallback_enabled = _should_allow_fallback(params) if allow_fallback is None else allow_fallback

    missing_project_error: ProjectNotFoundError | None = None
    try:
        return _resolve_from_project(project_root)
    except ProjectNotFoundError as exc:
        if not fallback_enabled:
            raise ResolutionError(_MSG_NO_PROJECT_NO_FALLBACK) from exc
        missing_project_error = exc

    resolved_params = dict(params)
    selection: _FallbackSelection | None = None
    if resolved_params.get("repo_root") is None:
        base = project_root or Path.cwd()
        selection = _select_fallback_repo_root(base)
        if selection is None:
            raise ResolutionError(_MSG_NO_PROJECT_NO_SOURCE) from missing_project_error
        resolved_params["repo_root"] = selection.repo_root

    runtime = _resolve_from_params_dict(resolved_params)
    if selection is not None:
        _log_fallback_selection(selection, runtime)
    return runtime


def _should_allow_fallback(params: Mapping[str, object] | Mapping[str, str]) -> bool:
    if params.get("repo") or params.get("commit") or params.get("db_path"):
        return True
    base = _to_path_or_none(params.get("project_root")) or _to_path_or_none(params.get("repo_root"))
    base = base or Path.cwd()
    return _select_fallback_repo_root(base) is not None


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
    cfg = _apply_default_scip_project_name(cfg, cfg.repo.repo)

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths

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
        primitives=build_runtime_primitives(
            RuntimeInputs(
                snapshot=snapshot,
                paths=paths,
                tools=cfg.tools.to_binaries(),
                graph_backend=cfg.graph_backend,
                graph_features=cfg.graph_features,
                profiles=None,
            )
        ),
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
    repo_root = _to_path_with_default(params.get("repo_root"), Path.cwd())
    repo, commit = _extract_required_params_dict(params, repo_root=repo_root)

    db_path = _to_path_with_default(params.get("db_path"), Path("build/db/codeintel.duckdb"))
    build_dir = _to_path_with_default(params.get("build_dir"), Path("build"))
    document_output_dir = _to_path_or_none(params.get("document_output_dir"))

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
    config = _apply_default_scip_project_name(config, repo)

    config.build_paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    return ResolvedRuntime(
        root=repo_root,
        project=ProjectConfig(
            repo=repo,
            storage=StorageProjectConfig(db_path=config.build_paths.db_path),
        ),
        primitives=build_runtime_primitives(
            RuntimeInputs(
                snapshot=SnapshotRef(repo=repo, commit=commit, repo_root=repo_root),
                paths=config.build_paths,
                tools=config.tools.to_binaries(),
                graph_backend=config.graph_backend,
                graph_features=config.graph_features,
                profiles=None,
            )
        ),
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
    *,
    repo_root: Path,
) -> tuple[str, str]:
    """Extract and validate required repo and commit params.

    Parameters
    ----------
    params
        Parameters dict.
    repo_root
        Repository root used for fallback inference.

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
    resolved_root = repo_root.resolve()

    if repo is None:
        env_repo = os.environ.get("CODEINTEL_REPO")
        if env_repo is not None:
            repo = env_repo.strip() or None
    if repo is None:
        repo = _infer_repo_from_git_remote(resolved_root) or (resolved_root.name or None)

    if commit is None:
        commit = detect_commit(resolved_root)

    if repo is None:
        raise ResolutionError(_MSG_MISSING_PARAMS, missing_params=["repo"])

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


def _log_fallback_selection(
    selection: _FallbackSelection,
    runtime: ResolvedRuntime,
) -> None:
    LOG.warning(
        "Selection flag: auto-selected %s root=%s (repo=%s, commit=%s). "
        "To override, pass --root or set project.root/project.repo/project.commit "
        "in codeintel.toml, ~/.codeintel/config.yaml, or codeintel.yaml.",
        selection.source,
        runtime.repo_root,
        runtime.repo,
        runtime.commit,
    )


__all__ = [
    "resolve_from_params",
]
