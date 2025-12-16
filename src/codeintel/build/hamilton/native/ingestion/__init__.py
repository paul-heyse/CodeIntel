"""Native ingestion targets with tool execution subgraphs.

Phase 2: All ingestion domain plugins migrated to native Hamilton modules
with @check_output_custom validators and @schema.output documentation.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.ingestion.ast import (
    AstExtractResult,
    t__ast,
    t__ast__extract,
)
from codeintel.build.hamilton.native.ingestion.config import (
    ConfigIngestResult,
    ConfigScanResult,
    t__config_ingest,
    t__config_ingest__ingest,
    t__config_ingest__scan,
)
from codeintel.build.hamilton.native.ingestion.coverage import (
    CoverageIngestResult,
    t__coverage_ingest,
    t__coverage_ingest__ingest,
)
from codeintel.build.hamilton.native.ingestion.cst import (
    CstExtractResult,
    t__cst,
    t__cst__extract,
)
from codeintel.build.hamilton.native.ingestion.docstrings import (
    DocstringsExtractResult,
    t__docstrings,
    t__docstrings__extract,
)
from codeintel.build.hamilton.native.ingestion.modules import (
    ModuleScanResult,
    RepoMapWriteResult,
    t__modules,
    t__modules__scan,
    t__modules__write_repo_map,
)
from codeintel.build.hamilton.native.ingestion.scip import (
    parse__scip,
    t__scip,
    tool__scip,
)
from codeintel.build.hamilton.native.ingestion.tests import (
    TestsIngestResult,
    t__tests_ingest,
    t__tests_ingest__ingest,
)
from codeintel.build.hamilton.native.ingestion.typing import (
    parse__typing,
    t__typing,
    tool__typing__pyrefly,
    tool__typing__pyright,
    tool__typing__ruff,
)

__all__: list[str] = [
    # ast target
    "AstExtractResult",
    "t__ast",
    "t__ast__extract",
    # config target
    "ConfigIngestResult",
    "ConfigScanResult",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__scan",
    # coverage target
    "CoverageIngestResult",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
    # cst target
    "CstExtractResult",
    "t__cst",
    "t__cst__extract",
    # docstrings target
    "DocstringsExtractResult",
    "t__docstrings",
    "t__docstrings__extract",
    # modules target
    "ModuleScanResult",
    "RepoMapWriteResult",
    "t__modules",
    "t__modules__scan",
    "t__modules__write_repo_map",
    # scip target
    "parse__scip",
    "t__scip",
    "tool__scip",
    # tests target
    "TestsIngestResult",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    # typing target
    "parse__typing",
    "t__typing",
    "tool__typing__pyrefly",
    "tool__typing__pyright",
    "tool__typing__ruff",
]
