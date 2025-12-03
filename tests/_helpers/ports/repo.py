"""Repository port protocol for filesystem operations.

This module defines the RepoPort protocol that abstracts filesystem repository
operations. Tests code to this protocol while using real filesystem
implementations per the Testing Charter.

Design Notes
------------
- Protocol methods cover common repo file operations
- Real adapter uses pathlib.Path with temp directories
- No in-memory filesystem fakes; tests use real temp directories
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class RepoPort(Protocol):
    """Protocol for repository filesystem operations.

    Defines the interface tests use for repository file access.
    Implementations use real filesystem operations with temp directories.

    Attributes
    ----------
    root : Path
        Root path of the repository.
    """

    @property
    def root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Absolute path to repository root.
        """
        ...

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write content to a file in the repository.

        Creates parent directories as needed.

        Parameters
        ----------
        relative_path
            Path relative to repo root.
        content
            File content to write.

        Returns
        -------
        Path
            Absolute path to created file.
        """
        ...

    def read_file(self, relative_path: str) -> str:
        """Read content from a file in the repository.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        str
            File content.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        """
        ...

    def exists(self, relative_path: str) -> bool:
        """Check if a path exists in the repository.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        bool
            True if path exists.
        """
        ...

    def mkdir(self, relative_path: str) -> Path:
        """Create a directory in the repository.

        Creates parent directories as needed.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        Path
            Absolute path to created directory.
        """
        ...

    def list_files(self, relative_path: str = "", pattern: str = "*.py") -> list[Path]:
        """List files matching a pattern.

        Parameters
        ----------
        relative_path
            Starting directory relative to repo root.
        pattern
            Glob pattern to match.

        Returns
        -------
        list[Path]
            Matching file paths (relative to repo root).
        """
        ...


class FileSystemRepo:
    """Real filesystem implementation of RepoPort.

    Uses pathlib.Path operations on a temp directory. This is the production
    adapter for tests per the Testing Charter requirement for real technology.

    Attributes
    ----------
    root : Path
        Root path of the repository.
    """

    def __init__(self, root: Path) -> None:
        """Initialize with repository root path.

        Parameters
        ----------
        root
            Directory to use as repo root.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Absolute path to repository root.
        """
        return self._root

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write content to a file in the repository.

        Creates parent directories as needed.

        Parameters
        ----------
        relative_path
            Path relative to repo root.
        content
            File content to write.

        Returns
        -------
        Path
            Absolute path to created file.
        """
        full_path = self._root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    def read_file(self, relative_path: str) -> str:
        """Read content from a file in the repository.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        str
            File content.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        """
        full_path = self._root / relative_path
        if not full_path.exists():
            message = f"File not found: {relative_path}"
            raise FileNotFoundError(message)
        return full_path.read_text()

    def exists(self, relative_path: str) -> bool:
        """Check if a path exists in the repository.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        bool
            True if path exists.
        """
        return (self._root / relative_path).exists()

    def mkdir(self, relative_path: str) -> Path:
        """Create a directory in the repository.

        Creates parent directories as needed.

        Parameters
        ----------
        relative_path
            Path relative to repo root.

        Returns
        -------
        Path
            Absolute path to created directory.
        """
        full_path = self._root / relative_path
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    def list_files(self, relative_path: str = "", pattern: str = "*.py") -> list[Path]:
        """List files matching a pattern.

        Parameters
        ----------
        relative_path
            Starting directory relative to repo root.
        pattern
            Glob pattern to match.

        Returns
        -------
        list[Path]
            Matching file paths (relative to repo root).
        """
        start_dir = self._root / relative_path if relative_path else self._root
        return [p.relative_to(self._root) for p in start_dir.rglob(pattern)]


__all__ = ["FileSystemRepo", "RepoPort"]
