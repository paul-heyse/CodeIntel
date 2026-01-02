"""Repository file cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.advanced_query_engine.util.line_index import LineIndex


@dataclass
class RepoCache:
    """Cache file bytes and line indexes for a repository."""

    repo_root: Path
    _bytes: dict[str, bytes]
    _line_index: dict[str, LineIndex]

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._bytes = {}
        self._line_index = {}

    def read_bytes(self, rel_path: str) -> bytes:
        """Read bytes for a repo-relative path with caching.

        Parameters
        ----------
        rel_path:
            Repo-relative path to read.

        Returns
        -------
        bytes
            File contents as bytes.
        """
        if rel_path not in self._bytes:
            self._bytes[rel_path] = (self.repo_root / rel_path).read_bytes()
        return self._bytes[rel_path]

    def line_index(self, rel_path: str) -> LineIndex:
        """Return a cached LineIndex for a repo-relative path.

        Parameters
        ----------
        rel_path:
            Repo-relative path to index.

        Returns
        -------
        LineIndex
            Cached line index for the file.
        """
        if rel_path not in self._line_index:
            self._line_index[rel_path] = LineIndex.build(self.read_bytes(rel_path))
        return self._line_index[rel_path]


__all__ = ["RepoCache"]
