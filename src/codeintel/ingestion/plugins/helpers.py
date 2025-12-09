"""Shared helper functions for ingestion plugins."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

__all__ = ["get_module_paths", "paths_to_modules"]


def paths_to_modules(paths: Sequence[str], repo_root: Path) -> list[ModuleRecord]:
    """Convert relative paths to ModuleRecord objects with metadata.

    Returns
    -------
    list[ModuleRecord]
        Module metadata for each provided path in order.
    """
    total = len(paths)
    return [
        ModuleRecord(
            rel_path=path,
            module_name=path.replace("/", ".").removesuffix(".py"),
            file_path=repo_root / path,
            index=i + 1,
            total=total,
        )
        for i, path in enumerate(paths)
    ]


def get_module_paths(ctx: TargetExecutionContext) -> list[str]:
    """Fetch module paths from context resources or gateway.

    Returns
    -------
    list[str]
        Module paths derived from context resources or storage; empty when unavailable.
    """
    if ctx.resources.modules:
        return list(ctx.resources.modules)
    try:
        rows = ctx.gateway.con.execute(
            "SELECT path FROM core.modules WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except (RuntimeError, OSError):
        return []
