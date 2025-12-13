"""Build plugins for the CodeIntel build system.

This package contains all build plugins organized by domain:

- ingestion: Data ingestion plugins (AST, CST, SCIP, etc.)
- graphs: Graph construction plugins (callgraph, CFG, DFG, etc.)
- analytics: Analytics computation plugins (metrics, coverage, etc.)

All plugins implement the TargetPlugin protocol from codeintel.build.plugin.
Plugins are registered with the build registry in codeintel.build.plugin_registry.
"""

from __future__ import annotations

__all__: list[str] = []
