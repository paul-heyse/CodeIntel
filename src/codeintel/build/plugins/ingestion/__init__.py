"""Ingestion plugin helpers for the build system.

This package provides helper utilities for ingestion targets.
The plugin implementations have been migrated to native Hamilton modules
in Phase 2 of the Hamilton Native Implementation Plan.

See: codeintel.build.hamilton.native.ingestion for native implementations.

Utilities
---------
- helpers: Shared helper functions for module paths and filtering.

Options have been migrated to:
    codeintel.build.hamilton.native.options.ingestion
"""

from __future__ import annotations

from codeintel.build.hamilton.native.options.ingestion import (
    ModuleIngestOptions,
    ScipIngestOptions,
)
from codeintel.build.plugins.ingestion.helpers import (
    build_scan_profile,
    filter_modules,
    get_module_paths,
    paths_to_modules,
)

__all__ = [
    "ModuleIngestOptions",
    "ScipIngestOptions",
    "build_scan_profile",
    "filter_modules",
    "get_module_paths",
    "paths_to_modules",
]
