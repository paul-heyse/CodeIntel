"""Ingestion plugins for the build system.

This package provides plugin implementations for ingestion targets.
All plugins implement the TargetPlugin protocol from codeintel.build.plugin.

Class-Based Plugins
-------------------
The following plugins are available:

- RepoScanPlugin: Scan repository modules and build change-tracker state.
- AstExtractPlugin: Parse Python AST and persist rows + metrics.
- CstExtractPlugin: Parse CST via LibCST and write rows.
- ScipIngestPlugin: Run scip-python and persist symbols.
- TypingIngestPlugin: Populate typedness and static diagnostics.
- CoverageIngestPlugin: Load coverage.py data.
- TestsIngestPlugin: Ingest pytest JSON report.
- DocstringsIngestPlugin: Extract docstrings and persist structured rows.
- ConfigIngestPlugin: Flatten config files into config_values.

Example
-------
>>> from codeintel.build.plugins.ingestion import AstExtractPlugin
>>> from codeintel.build.plugin_registry import get_plugin_for_target
>>> plugin = get_plugin_for_target("ast")
>>>
"""

from __future__ import annotations

from codeintel.build.plugins.ingestion.ast_extract import AstExtractPlugin
from codeintel.build.plugins.ingestion.config_plugin import ConfigIngestPlugin
from codeintel.build.plugins.ingestion.coverage_plugin import CoverageIngestPlugin
from codeintel.build.plugins.ingestion.cst_extract import CstExtractPlugin
from codeintel.build.plugins.ingestion.docstrings_plugin import DocstringsIngestPlugin
from codeintel.build.plugins.ingestion.repo_scan import RepoScanPlugin
from codeintel.build.plugins.ingestion.scip_plugin import ScipIngestPlugin
from codeintel.build.plugins.ingestion.tests_plugin import TestsIngestPlugin
from codeintel.build.plugins.ingestion.typing_plugin import TypingIngestPlugin

__all__ = [
    "AstExtractPlugin",
    "ConfigIngestPlugin",
    "CoverageIngestPlugin",
    "CstExtractPlugin",
    "DocstringsIngestPlugin",
    "RepoScanPlugin",
    "ScipIngestPlugin",
    "TestsIngestPlugin",
    "TypingIngestPlugin",
]
