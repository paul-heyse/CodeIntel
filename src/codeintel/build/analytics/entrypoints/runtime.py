"""Runtime helpers for entrypoint detection that read module sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.build.analytics.entrypoints.core import EntrypointModuleSource
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


def iter_entrypoint_module_sources(
    module_map: Mapping[str, str],
    repo_root: Path,
    *,
    scan_profile: ScanProfile | None = None,
) -> Iterable[EntrypointModuleSource]:
    """Iterate in-memory module sources for entrypoint detection.

    Returns
    -------
    Iterable[EntrypointModuleSource]
        Iterable of module sources for entrypoint scanning.
    """
    module_map_dict = dict(module_map)
    if not module_map_dict:
        return iter(())
    return _iter_sources(module_map_dict, repo_root, scan_profile)


def load_entrypoint_module_sources(
    module_map: Mapping[str, str],
    repo_root: Path,
    *,
    scan_profile: ScanProfile | None = None,
) -> list[EntrypointModuleSource]:
    """Load all module sources needed for entrypoint detection.

    Returns
    -------
    list[EntrypointModuleSource]
        Loaded module sources for entrypoint scanning.
    """
    return list(iter_entrypoint_module_sources(module_map, repo_root, scan_profile=scan_profile))


__all__ = [
    "iter_entrypoint_module_sources",
    "load_entrypoint_module_sources",
]
