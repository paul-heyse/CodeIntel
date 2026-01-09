"""Shared utilities for lint/guardrail file discovery."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from rpygrep import RipGrepSearch


def list_python_files(root: Path, rel_roots: Sequence[str]) -> tuple[Path, ...]:
    """Return Python files under the given relative roots.

    Parameters
    ----------
    root
        Repository root for path resolution.
    rel_roots
        Relative directory roots to scan.

    Returns
    -------
    tuple[Path, ...]
        Python file paths in traversal order.
    """
    return _list_python_files_cached(root, tuple(rel_roots))


@lru_cache(maxsize=32)
def _list_python_files_cached(root: Path, rel_roots: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for rel_root in rel_roots:
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            paths.append(path)
    return tuple(paths)


def find_literal_candidates(
    root: Path,
    *,
    patterns: Sequence[str],
    include_globs: Sequence[str],
) -> set[Path]:
    """Return files that contain any of the literal patterns.

    Parameters
    ----------
    root
        Repository root for the search.
    patterns
        Literal patterns to search for.
    include_globs
        Glob patterns limiting search scope.

    Returns
    -------
    set[Path]
        Candidate file paths that matched the search.
    """
    if not patterns:
        return set()

    search = RipGrepSearch(working_directory=root).patterns_are_not_regex()
    for pattern in patterns:
        search = search.add_pattern(pattern)
    if include_globs:
        search = search.include_globs(list(include_globs))
    search = search.include_types(["py"]).max_count(1).add_extra_options(["--no-config"])
    candidates: set[Path] = set()
    for result in search.run():
        path = result.path
        if not path.is_absolute():
            path = (root / path).resolve()
        candidates.add(path)
    return candidates
