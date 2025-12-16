"""Ingestion plugin helpers for the build system.

This package provides helper utilities for ingestion targets.
The plugin implementations have been migrated to native Hamilton modules
in Phase 2 of the Hamilton Native Implementation Plan.

See: codeintel.build.hamilton.native.ingestion for native implementations.

Utilities
---------
- helpers: Shared helper functions for module paths and filtering.
- modules_options: Options model for module ingestion.
- scip_options: Options model for SCIP ingestion.
"""

from __future__ import annotations

from codeintel.build.plugins.ingestion.helpers import (
    build_scan_profile,
    filter_modules,
    get_module_paths,
    paths_to_modules,
)
from codeintel.build.plugins.ingestion.modules_options import ModuleIngestOptions
from codeintel.build.plugins.ingestion.scip_options import ScipIngestOptions

__all__ = [
    "ModuleIngestOptions",
    "ScipIngestOptions",
    "build_scan_profile",
    "filter_modules",
    "get_module_paths",
    "paths_to_modules",
]
