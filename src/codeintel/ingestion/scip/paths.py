"""Path helpers for SCIP indexing."""

from __future__ import annotations

from pathlib import Path


def resolve_target_base(repo_root: Path, target_dir: Path | None) -> Path:
    """Resolve the base directory passed to scip-python.

    Parameters
    ----------
    repo_root
        Repository root path.
    target_dir
        Optional override for the scip-python project root.

    Returns
    -------
    Path
        Directory passed as the scip-python project root.
    """
    if target_dir is not None:
        return target_dir
    src_dir = repo_root / "src"
    return src_dir if src_dir.is_dir() else repo_root


def scip_relative_path(
    *,
    repo_root: Path,
    target_base: Path,
    rel_path: str,
) -> str | None:
    """Return a path relative to the scip-python project root.

    Parameters
    ----------
    repo_root
        Repository root path.
    target_base
        scip-python project root (same value passed to scip-python index).
    rel_path
        Path relative to repo_root.

    Returns
    -------
    str | None
        Path relative to target_base using POSIX separators, or None if the
        module is outside the target_base.
    """
    abs_path = repo_root / rel_path
    try:
        relative = abs_path.relative_to(target_base)
    except ValueError:
        return None
    return relative.as_posix()


__all__ = ["resolve_target_base", "scip_relative_path"]
