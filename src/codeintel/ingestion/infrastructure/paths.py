"""Path utilities to normalize repository-relative paths and modules.

.. deprecated:: 1.0
    Import from ``codeintel.core.paths`` instead.
    This module will be removed in a future version.

Examples
--------
Instead of:

>>> from codeintel.ingestion.infrastructure.paths import normalize_rel_path

Use:

>>> from codeintel.core.paths import normalize_path
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.paths import (
    ensure_repo_root,
    path_to_module,
    repo_relpath,
)

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.ingestion.infrastructure.paths is deprecated. "
        "Import from codeintel.core.paths instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def normalize_rel_path(path: str | Path) -> str:
    """Return a POSIX-style relative path (keeps subdirs, strips backslashes).

    .. deprecated:: 1.0
        Use ``codeintel.core.paths.normalize_path`` instead.

    Parameters
    ----------
    path
        Path to normalize.

    Returns
    -------
    str
        Normalized path with forward slashes.

    Examples
    --------
    >>> normalize_rel_path("src/module/file.py")
    'src/module/file.py'
    """
    return Path(path).as_posix()


def relpath_to_module(rel_path: str | Path) -> str:
    """Convert a repository-relative Python path to a dotted module name.

    .. deprecated:: 1.0
        Use ``codeintel.core.paths.path_to_module`` instead.

    Parameters
    ----------
    rel_path
        Repository-relative file path.

    Returns
    -------
    str
        Dotted module path for the given relative path.

    Examples
    --------
    >>> relpath_to_module("pkg/sub/module.py")
    'pkg.sub.module'
    """
    return ".".join(Path(rel_path).with_suffix("").parts)


def safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    """Safely compute repository-relative path, returning None on failure.

    Handle both absolute and relative file paths, normalizing them to
    a repository-relative POSIX path.

    .. deprecated:: 1.0
        Use ``codeintel.core.paths.safe_relpath`` instead.
        Note: The core version has a different signature and returns the
        normalized absolute path instead of None on failure.

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

    Examples
    --------
    >>> from pathlib import Path
    >>> safe_relpath(Path("/project"), Path("/project/src/file.py"))
    'src/file.py'
    """
    try:
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None


# Re-export from core (these have compatible signatures)
__all__ = [
    "ensure_repo_root",
    "normalize_rel_path",
    "path_to_module",
    "relpath_to_module",
    "repo_relpath",
    "safe_relpath",
]
