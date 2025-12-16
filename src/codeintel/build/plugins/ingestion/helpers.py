"""Shared helper functions for ingestion plugins."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.plugins._helpers import is_test_path
from codeintel.ingestion.infrastructure.scanning import ScanProfile, default_code_profile
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.ibis_types import ibis_bool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions

__all__ = [
    "build_scan_profile",
    "filter_modules",
    "get_module_paths",
    "paths_to_modules",
]

log = logging.getLogger(__name__)


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


def build_scan_profile(repo_root: Path, options: ModuleIngestOptions) -> ScanProfile:
    """Build a scan profile that respects configured scope and ignore settings.

    Returns
    -------
    ScanProfile
        Scan profile with option-driven roots and ignores applied.
    """
    base_profile = default_code_profile(repo_root)

    ignore_dirs = list(base_profile.ignore_dirs)
    if not options.include_tests and "tests" not in ignore_dirs:
        ignore_dirs.append("tests")
    if not options.include_generated:
        for name in ("generated", ".generated"):
            if name not in ignore_dirs:
                ignore_dirs.append(name)

    source_roots = base_profile.source_roots
    if options.scope_paths:
        resolved_roots: list[Path] = []
        for scope in options.scope_paths:
            scope_path = Path(scope)
            resolved = scope_path if scope_path.is_absolute() else repo_root / scope_path
            if resolved.is_file():
                resolved = resolved.parent
            if resolved.is_dir() and resolved not in resolved_roots:
                resolved_roots.append(resolved)
        if resolved_roots:
            source_roots = tuple(resolved_roots)

    return ScanProfile(
        repo_root=repo_root,
        source_roots=source_roots,
        include_globs=base_profile.include_globs,
        ignore_dirs=tuple(ignore_dirs),
        log_every=base_profile.log_every,
        log_interval=base_profile.log_interval,
    )


def _is_generated_path(rel_path: str) -> bool:
    """Return True when the path looks like a generated artifact.

    Returns
    -------
    bool
        True when path appears generated.
    """
    path = Path(rel_path)
    lower_parts = [part.lower() for part in path.parts]
    if any(part in {"generated", ".generated"} for part in lower_parts):
        return True
    filename = path.name.lower()
    return filename.endswith(("_generated.py", ".generated.py", "_pb2.py", "_pb2.pyi"))


def _is_in_scope(rel_path: str, scope_paths: Sequence[str]) -> bool:
    """Check whether a path is contained within any configured scope prefix.

    Returns
    -------
    bool
        True when the path resides under a configured scope.
    """
    if not scope_paths:
        return True
    rel_parts = Path(rel_path).parts
    for scope in scope_paths:
        scope_parts = Path(scope).parts
        if scope_parts and rel_parts[: len(scope_parts)] == scope_parts:
            return True
    return False


def filter_modules(
    modules: Sequence[ModuleRecord],
    options: ModuleIngestOptions,
) -> list[ModuleRecord]:
    """Apply scope, test, generated, and size filters to discovered modules.

    Returns
    -------
    list[ModuleRecord]
        Modules that satisfy configured options.
    """
    filtered: list[ModuleRecord] = []
    for module in modules:
        rel_path = module.rel_path
        if options.scope_paths and not _is_in_scope(rel_path, options.scope_paths):
            continue
        if not options.include_tests and is_test_path(rel_path):
            continue
        if not options.include_generated and _is_generated_path(rel_path):
            continue
        if options.max_file_size_kb > 0:
            try:
                size_bytes = module.file_path.stat().st_size
            except OSError:
                continue
            if size_bytes > options.max_file_size_kb * 1024:
                continue
        filtered.append(module)
    return filtered


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
        table = ctx.gateway.ibis.table("core.modules")
        df = (
            table.filter(
                [
                    ibis_bool(table.repo == ctx.repo),
                    ibis_bool(table.commit == ctx.commit),
                ]
            )
            .select("path")
            .execute()
        )
        return [str(path) for path in df["path"].tolist()]
    except (RuntimeError, OSError) as exc:
        log.warning("gateway error fetching module paths: %s", exc)
        return []
