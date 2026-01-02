"""Worktree file listing helpers aligned with .gitignore."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec
from pathspec.util import normalize_file


@dataclass(frozen=True)
class WorktreeFilter:
    """Represents gitignore-based filtering for a repository."""

    repo_root: Path
    ignore_spec: GitIgnoreSpec | None

    @classmethod
    def from_repo(cls, repo_root: Path) -> WorktreeFilter:
        """Load a gitignore spec from the repository root.

        Returns
        -------
        WorktreeFilter
            Worktree filter with compiled gitignore rules, if present.
        """
        ignore_file = repo_root / ".gitignore"
        spec = None
        if ignore_file.exists():
            lines = ignore_file.read_text(encoding="utf-8").splitlines()
            spec = GitIgnoreSpec.from_lines(lines)
        return cls(repo_root=repo_root, ignore_spec=spec)

    def is_ignored(self, rel_path: Path, *, is_dir: bool) -> bool:
        """Return True if a repo-relative path is ignored by gitignore.

        Parameters
        ----------
        rel_path:
            Repo-relative path to test.
        is_dir:
            Whether the path is a directory.

        Returns
        -------
        bool
            True when the path should be ignored.
        """
        if self.ignore_spec is None:
            return False
        text = rel_path.as_posix()
        if is_dir and not text.endswith("/"):
            text = f"{text}/"
        return bool(self.ignore_spec.match_file(normalize_file(text)))


def iter_worktree_files(
    repo_root: Path,
    *,
    scope_paths: Iterable[str] | None = None,
    max_depth: int = 0,
) -> Iterator[str]:
    """Yield repo-relative files obeying .gitignore semantics.

    Parameters
    ----------
    repo_root:
        Root of the repository to scan.
    scope_paths:
        Optional list of repo-relative paths to constrain the scan.
    max_depth:
        Maximum recursion depth relative to the scope root. Use 0 for unlimited.

    Yields
    ------
    str
        Repo-relative POSIX path for each file.
    """
    filter_spec = WorktreeFilter.from_repo(repo_root)
    for base in _scope_roots(repo_root, scope_paths):
        if base.is_file():
            rel_path = base.relative_to(repo_root)
            if not filter_spec.is_ignored(rel_path, is_dir=False):
                yield rel_path.as_posix()
            continue
        yield from _walk_root(
            repo_root=repo_root,
            base=base,
            filter_spec=filter_spec,
            max_depth=max_depth,
        )


def list_worktree_files(
    repo_root: Path,
    *,
    scope_paths: Iterable[str] | None = None,
    max_depth: int = 0,
) -> list[str]:
    """Return sorted repo-relative files obeying .gitignore semantics.

    Returns
    -------
    list[str]
        Sorted repo-relative files.
    """
    return sorted(set(iter_worktree_files(repo_root, scope_paths=scope_paths, max_depth=max_depth)))


def list_python_files(
    repo_root: Path,
    *,
    scope_paths: Iterable[str] | None = None,
    max_depth: int = 0,
) -> list[str]:
    """Return sorted Python files obeying .gitignore semantics.

    Returns
    -------
    list[str]
        Sorted Python files.
    """
    files = list_worktree_files(repo_root, scope_paths=scope_paths, max_depth=max_depth)
    return [path for path in files if path.endswith(".py")]


def _scope_roots(repo_root: Path, scope_paths: Iterable[str] | None) -> list[Path]:
    if not scope_paths:
        return [repo_root]
    roots: list[Path] = []
    repo_root = repo_root.resolve()
    for value in scope_paths:
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            candidate = candidate.resolve()
        except FileNotFoundError:
            continue
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            continue
        if candidate.exists():
            roots.append(candidate)
    return roots or [repo_root]


def _walk_root(
    *,
    repo_root: Path,
    base: Path,
    filter_spec: WorktreeFilter,
    max_depth: int,
) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = Path(dirpath).relative_to(repo_root)
        depth = _relative_depth(Path(dirpath), base)
        _prune_dirnames(rel_dir, dirnames, filter_spec, max_depth, depth)
        for filename in filenames:
            rel_path = rel_dir / filename
            if filter_spec.is_ignored(rel_path, is_dir=False):
                continue
            if rel_path.parts and rel_path.parts[0] == ".git":
                continue
            yield rel_path.as_posix()


def _prune_dirnames(
    rel_dir: Path,
    dirnames: list[str],
    filter_spec: WorktreeFilter,
    max_depth: int,
    depth: int,
) -> None:
    if max_depth and depth >= max_depth:
        dirnames[:] = []
        return
    pruned: list[str] = []
    for dirname in dirnames:
        if dirname == ".git":
            continue
        candidate = rel_dir / dirname
        if filter_spec.is_ignored(candidate, is_dir=True):
            continue
        pruned.append(dirname)
    dirnames[:] = pruned


def _relative_depth(current: Path, base: Path) -> int:
    if current == base:
        return 0
    rel = current.relative_to(base)
    if rel == Path():
        return 0
    return len(rel.parts)


__all__ = [
    "WorktreeFilter",
    "iter_worktree_files",
    "list_python_files",
    "list_worktree_files",
]
