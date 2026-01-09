"""Runtime helpers for entrypoint detection that read module sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.entrypoints.core import EntrypointModuleSource
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.paths import normalize_path, safe_relpath
from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

    from codeintel.ingestion.infrastructure.scanning import ScanProfile

log = logging.getLogger(__name__)


def _iter_sources(
    module_map: dict[str, str],
    repo_root: Path,
    scan_profile: ScanProfile | None,
) -> Iterator[EntrypointModuleSource]:
    for record in FilesystemDiscoveryAdapter.iter_modules(
        module_map,
        repo_root,
        logger=log,
        scan_profile=scan_profile,
    ):
        source = FilesystemDiscoveryAdapter.read_module_source(record)
        if source is None:
            continue
        yield EntrypointModuleSource(
            rel_path=record.rel_path,
            module=record.module_name,
            source=source,
        )


def _module_from_values(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if item is not None]
        items = [item for item in items if item]
        if not items:
            return None
        return sorted(items)[0]
    return None


def _module_map_from_worklist(worklist: pa.Table, repo_root: Path) -> dict[str, str]:
    module_map: dict[str, str] = {}
    if not {"path", "modules"}.issubset(worklist.column_names):
        return module_map
    for path, modules in iter_tuples(worklist.to_reader(), columns=("path", "modules")):
        if not isinstance(path, str) or not path.strip():
            continue
        module = _module_from_values(modules)
        if module is None:
            continue
        rel_path = normalize_path(safe_relpath(path, repo_root))
        module_map[rel_path] = module
    return module_map


def iter_entrypoint_module_sources(
    module_map: Mapping[str, str],
    repo_root: Path,
    *,
    module_worklist: pa.Table | None = None,
    scan_profile: ScanProfile | None = None,
) -> Iterable[EntrypointModuleSource]:
    """Iterate in-memory module sources for entrypoint detection.

    Returns
    -------
    Iterable[EntrypointModuleSource]
        Iterable of module sources for entrypoint scanning.
    """
    module_map_dict: dict[str, str] = {}
    if module_worklist is not None:
        module_map_dict = _module_map_from_worklist(module_worklist, repo_root)
    if not module_map_dict:
        module_map_dict = dict(module_map)
    if not module_map_dict:
        return iter(())
    return _iter_sources(module_map_dict, repo_root, scan_profile)


def load_entrypoint_module_sources(
    module_map: Mapping[str, str],
    repo_root: Path,
    *,
    module_worklist: pa.Table | None = None,
    scan_profile: ScanProfile | None = None,
) -> list[EntrypointModuleSource]:
    """Load all module sources needed for entrypoint detection.

    Returns
    -------
    list[EntrypointModuleSource]
        Loaded module sources for entrypoint scanning.
    """
    return list(
        iter_entrypoint_module_sources(
            module_map,
            repo_root,
            module_worklist=module_worklist,
            scan_profile=scan_profile,
        )
    )


__all__ = [
    "iter_entrypoint_module_sources",
    "load_entrypoint_module_sources",
]
