"""Path utilities to normalize repository-relative paths and modules."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ensure_repo_root",
    "normalize_rel_path",
    "relpath_to_module",
    "repo_relpath",
    "safe_relpath",
]


def ensure_repo_root(repo_root: Path | str) -> Path:
    """Resolve a repo root to an absolute, expanded Path.

    Returns
    -------
    Path
        Absolute repository root.
    """
    return Path(repo_root).expanduser().resolve()


def normalize_rel_path(path: str | Path) -> str:
    """Return a POSIX-style relative path (keeps subdirs, strips backslashes).

    Returns
    -------
    str
        Normalized path with forward slashes.
    """
    return Path(path).as_posix()


def repo_relpath(repo_root: Path, path: Path | str) -> str:
    """Compute a repository-relative POSIX path for a file under repo_root.

    Returns
    -------
    str
        Relative path with forward slashes.
    """
    return Path(path).relative_to(repo_root).as_posix()


def relpath_to_module(rel_path: str | Path) -> str:
    """
    Convert a repository-relative Python path to a dotted module name.

    Examples
    --------
    >>> relpath_to_module("pkg/sub/module.py")
    'pkg.sub.module'

    Returns
    -------
    str
        Dotted module path for the given relative path.
    """
    return ".".join(Path(rel_path).with_suffix("").parts)


def safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    """Safely compute repository-relative path, returning None on failure.

    Handle both absolute and relative file paths, normalizing them to
    a repository-relative POSIX path.

    Parameters
    ----------
    repo_root
        Repository root path.
    file_path
        Absolute or relative file path.

    Returns
    -------
    str | None
        Normalized relative path or None on failure.
    """
    try:
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None
