"""Dulwich-based snapshot alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    from dulwich.repo import Repo as _DulwichRepo
except ImportError:
    _DulwichRepo = None

from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from dulwich.repo import Repo


def _discover_repo(start_path: Path) -> Repo | None:
    if _DulwichRepo is None:
        return None
    try:
        return _DulwichRepo.discover(start_path)
    except (OSError, ValueError):
        return None


def _coerce_commit(head: object) -> str | None:
    commit = head.decode("ascii", errors="ignore") if isinstance(head, bytes) else str(head)
    commit = commit.strip()
    return commit or None


def resolve_head_commit(repo_root: Path) -> str | None:
    """Resolve the current HEAD commit for the repo rooted at repo_root.

    Returns
    -------
    str | None
        Commit SHA string when available, otherwise None.
    """
    repo = _discover_repo(repo_root)
    if repo is None:
        return None
    return _coerce_commit(repo.head())


def snapshot_from_dulwich(start_path: Path | None = None) -> SnapshotRef | None:
    """Discover the current repo and build a SnapshotRef from HEAD.

    Returns
    -------
    SnapshotRef | None
        Snapshot reference when discoverable, otherwise None.
    """
    root = start_path or Path.cwd()
    repo = _discover_repo(root)
    if repo is None:
        return None
    repo_root = Path(repo.path).resolve()
    head = _coerce_commit(repo.head())
    if head is None:
        return None
    repo_name = repo_root.name or "repo"
    return SnapshotRef.from_args(
        repo=repo_name,
        commit=head,
        repo_root=repo_root,
    )


def ensure_snapshot_matches_head(snapshot: SnapshotRef) -> SnapshotRef:
    """Validate that a snapshot commit matches the repo HEAD.

    Returns
    -------
    SnapshotRef
        Snapshot reference when it matches repo HEAD.

    Raises
    ------
    RuntimeError
        If the repo HEAD cannot be resolved from the snapshot repo_root.
    ValueError
        If the snapshot commit does not match the repo HEAD.
    """
    head = resolve_head_commit(snapshot.repo_root)
    if head is None:
        msg = "Unable to resolve HEAD commit from repo_root"
        raise RuntimeError(msg)
    if head != snapshot.commit:
        msg = f"Snapshot commit does not match repo HEAD: snapshot={snapshot.commit} head={head}"
        raise ValueError(msg)
    return snapshot


__all__ = [
    "ensure_snapshot_matches_head",
    "resolve_head_commit",
    "snapshot_from_dulwich",
]
