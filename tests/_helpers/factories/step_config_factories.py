"""Factory functions for step configurations in tests.

This module provides factory functions for creating step configurations with
standard test defaults, reducing boilerplate in plugin tests.

Example
-------
>>> from tests._helpers.factories import make_step_config
>>> from codeintel.config.steps_analytics import FunctionContractsStepConfig
>>> config = make_step_config(FunctionContractsStepConfig, tmp_path)
>>> assert config.snapshot.repo == "demo/repo"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.config.primitives import SnapshotRef
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO

if TYPE_CHECKING:
    from collections.abc import Callable


def make_snapshot(
    repo_root: Path | None = None,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> SnapshotRef:
    """Create a standard test snapshot.

    Parameters
    ----------
    repo_root
        Optional repo root path; defaults to Path.cwd() if not provided.
    repo
        Repository identifier; defaults to DEFAULT_REPO.
    commit
        Commit identifier; defaults to DEFAULT_COMMIT.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference.
    """
    return SnapshotRef(
        repo=repo,
        commit=commit,
        repo_root=repo_root if repo_root is not None else Path.cwd(),
    )


def make_step_config[T](
    config_type: type[T],
    repo_root: Path | None = None,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
) -> T:
    """Create a step config with standard snapshot.

    This factory creates step configuration objects with a properly initialized
    SnapshotRef, using sensible test defaults for repo and commit identifiers.

    The config is created with only the required snapshot parameter. If you need
    to set additional config parameters, construct the config directly:

        snapshot = make_snapshot(tmp_path)
        config = MyStepConfig(snapshot=snapshot, my_param=value)

    Parameters
    ----------
    config_type
        The step config class to instantiate (must accept `snapshot` kwarg).
    repo_root
        Optional repo root path; defaults to Path.cwd() if not provided.
    repo
        Repository identifier; defaults to DEFAULT_REPO.
    commit
        Commit identifier; defaults to DEFAULT_COMMIT.

    Returns
    -------
    T
        Configured step config instance.

    Example
    -------
    >>> from codeintel.config.steps_analytics import FunctionContractsStepConfig
    >>> config = make_step_config(FunctionContractsStepConfig, tmp_path)
    >>> assert config.snapshot.repo == "demo/repo"
    """
    snapshot = make_snapshot(repo_root, repo=repo, commit=commit)
    # All step configs accept snapshot as their first/required parameter.
    # Cast the config class constructor to a callable that takes just snapshot,
    # since different step configs have varying optional parameters beyond snapshot.
    ctor = cast("Callable[..., T]", config_type)
    return ctor(snapshot=snapshot)


__all__ = [
    "make_snapshot",
    "make_step_config",
]
