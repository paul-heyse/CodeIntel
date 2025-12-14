r"""Path normalization utilities.

This module provides utilities for normalizing file paths,
computing relative paths, and resolving repository roots.

Examples
--------
>>> from codeintel.core.paths import normalize_path, repo_relpath
>>>
>>> normalize_path("src\\module\\file.py")
'src/module/file.py'
>>> repo_relpath(Path("/project"), Path("/project/src/file.py"))
'src/file.py'
"""

from __future__ import annotations

from pathlib import Path


def normalize_path(path: str | Path) -> str:
    r"""Normalize a file path.

    Converts the path to a consistent forward-slash format
    and removes redundant components.

    Parameters
    ----------
    path
        Path to normalize.

    Returns
    -------
    str
        Normalized path string.

    Examples
    --------
    >>> normalize_path("src\\module\\file.py")
    'src/module/file.py'
    >>> normalize_path("./src/../src/file.py")
    'src/file.py'
    """
    path_obj = Path(path)

    try:
        normalized = path_obj.resolve()
    except (OSError, ValueError):
        normalized = path_obj

    result = str(normalized).replace("\\", "/")

    if result.startswith("./"):
        result = result[2:]

    return result


def ensure_repo_root(repo_root: Path | str) -> Path:
    """Resolve a repo root to an absolute, expanded Path.

    Expands user home directory (~) and resolves to an absolute path.

    Parameters
    ----------
    repo_root
        Repository root path (may be relative or contain ~).

    Returns
    -------
    Path
        Absolute repository root.

    Examples
    --------
    >>> ensure_repo_root("~/projects/myrepo")
    PosixPath('/home/user/projects/myrepo')
    >>> ensure_repo_root(".")
    PosixPath('/current/working/dir')
    """
    return Path(repo_root).expanduser().resolve()


def repo_relpath(repo_root: Path, path: Path | str) -> str:
    """Compute a repository-relative POSIX path for a file under repo_root.

    Parameters
    ----------
    repo_root
        Repository root path.
    path
        Absolute or relative file path.

    Returns
    -------
    str
        Relative path with forward slashes.

    Examples
    --------
    >>> from pathlib import Path
    >>> repo_relpath(Path("/project"), Path("/project/src/file.py"))
    'src/file.py'

    Notes
    -----
    Raises ``ValueError`` if path is not under repo_root (propagated from
    ``Path.relative_to``).
    """
    return Path(path).relative_to(repo_root).as_posix()


def safe_relpath(path: str | Path, base: str | Path) -> str:
    """Compute a safe relative path.

    Returns the relative path from base to path, or the
    normalized absolute path if not possible.

    Parameters
    ----------
    path
        Target path.
    base
        Base path.

    Returns
    -------
    str
        Relative path or normalized absolute path.

    Examples
    --------
    >>> safe_relpath("/project/src/file.py", "/project")
    'src/file.py'
    """
    path_obj = Path(path)
    base_obj = Path(base)

    try:
        rel = path_obj.relative_to(base_obj)
        return str(rel).replace("\\", "/")
    except ValueError:
        return normalize_path(path)


__all__ = [
    "ensure_repo_root",
    "normalize_path",
    "repo_relpath",
    "safe_relpath",
]
