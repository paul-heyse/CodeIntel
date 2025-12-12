"""Ingestion plugins for the build system.

This package provides plugin implementations for ingestion targets.
All plugins implement the `TargetPlugin` protocol from `codeintel.build.plugin`.

Class-Based Plugins
-------------------
The following plugins are available:

- `RepoScanPlugin`: Scan repository modules and build change-tracker state.
- `AstExtractPlugin`: Parse Python AST and persist rows + metrics.
- `CstExtractPlugin`: Parse CST via LibCST and write rows.
- `ScipIngestPlugin`: Run scip-python and persist symbols.
- `TypingIngestPlugin`: Populate typedness and static diagnostics.
- `CoverageIngestPlugin`: Load coverage.py data.
- `TestsIngestPlugin`: Ingest pytest JSON report.
- `DocstringsIngestPlugin`: Extract docstrings and persist structured rows.
- `ConfigIngestPlugin`: Flatten config files into config_values.

Example
-------
>>> from codeintel.ingestion.plugins import AstExtractPlugin
>>> from codeintel.build.plugin_registry import get_plugin_for_target
>>> plugin = get_plugin_for_target("ast")
>>>
"""

from __future__ import annotations

from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
from codeintel.ingestion.plugins.config_plugin import ConfigIngestPlugin
from codeintel.ingestion.plugins.coverage_plugin import CoverageIngestPlugin
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin
from codeintel.ingestion.plugins.modules_plugin import ModuleIngestPlugin
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin
from codeintel.ingestion.plugins.scip_plugin import ScipIngestPlugin
from codeintel.ingestion.plugins.tests_plugin import TestsIngestPlugin
from codeintel.ingestion.plugins.typing_plugin import TypingIngestPlugin
from codeintel.ingestion.validation import (
    ColumnConstraint,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestContractValidator,
    foreign_key_contract,
    not_null_contract,
    row_count_contract,
)

__all__ = [
    "AstExtractPlugin",
    "ColumnConstraint",
    "ConfigIngestPlugin",
    "ContractValidationResult",
    "ContractViolation",
    "CoverageIngestPlugin",
    "CstExtractPlugin",
    "DocstringsIngestPlugin",
    "ForeignKeyConstraint",
    "IngestContractSpec",
    "IngestContractValidator",
    "ModuleIngestPlugin",
    "RepoScanPlugin",
    "ScipIngestPlugin",
    "TestsIngestPlugin",
    "TypingIngestPlugin",
    "foreign_key_contract",
    "not_null_contract",
    "row_count_contract",
]
