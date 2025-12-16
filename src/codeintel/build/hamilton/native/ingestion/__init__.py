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
    "AstExtractResult",
    "ConfigIngestResult",
    "ConfigScanResult",
    "CoverageIngestResult",
    "CstExtractResult",
    "DocstringsExtractResult",
    "ModuleScanResult",
    "RepoMapWriteResult",
    "TestsIngestResult",
    "parse__scip",
    "parse__typing",
    "t__ast",
    "t__ast__extract",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__scan",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
    "t__cst",
    "t__cst__extract",
    "t__docstrings",
    "t__docstrings__extract",
    "t__modules",
    "t__modules__scan",
    "t__modules__write_repo_map",
    "t__scip",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__typing",
    "tool__scip",
    "tool__typing__pyrefly",
    "tool__typing__pyright",
    "tool__typing__ruff",
]
