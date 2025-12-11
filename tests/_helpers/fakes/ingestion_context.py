"""Utility helpers for ingestion plugin tests.

This module provides simple utility functions for ingestion tests.
For execution context building, use ``ExecutionContextBuilder`` from
``tests._helpers.fakes.contexts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def build_repo_tree(root: Path, files: Mapping[str, str]) -> Path:
    """Write a set of files relative to root and return the root path.

    Parameters
    ----------
    root
        Root directory to write files into.
    files
        Mapping of relative file paths to their contents.

    Returns
    -------
    Path
        Repository root containing the written files.
    """
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


__all__ = [
    "build_repo_tree",
]
