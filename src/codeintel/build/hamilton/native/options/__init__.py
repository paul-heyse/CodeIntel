"""Configuration option dataclasses for Hamilton native modules.

This package contains configuration option dataclasses that were migrated
from the legacy plugin infrastructure.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.options.graphs import (
    CallGraphOptions,
    CfgDfgOptions,
    GoidBuilderOptions,
    ImportGraphOptions,
    SymbolUsesOptions,
)
from codeintel.build.hamilton.native.options.ingestion import (
    ModuleIngestOptions,
    ScipIngestOptions,
)

__all__ = [
    "CallGraphOptions",
    "CfgDfgOptions",
    "GoidBuilderOptions",
    "ImportGraphOptions",
    "ModuleIngestOptions",
    "ScipIngestOptions",
    "SymbolUsesOptions",
]
