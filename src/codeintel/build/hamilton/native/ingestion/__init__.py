"""Native ingestion targets with tool execution subgraphs.

Phase 2: All ingestion domain plugins migrated to native Hamilton modules
with @check_output_custom validators and @schema.output documentation.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.ingestion.extraction_targets import (
    AstExtractResult,
    CstExtractResult,
    DocstringsExtractResult,
    t__ast,
    t__ast__extract,
    t__cst,
    t__cst__extract,
    t__docstrings,
    t__docstrings__extract,
)
from codeintel.build.hamilton.native.ingestion.ingest_targets import (
    ConfigIngestResult,
    ConfigScanResult,
    CoverageIngestResult,
    ModuleScanResult,
    RepoMapWriteResult,
    TestsIngestResult,
    TypingIngestResult,
    t__config_ingest,
    t__config_ingest__ingest,
    t__config_ingest__scan,
    t__coverage_ingest,
    t__coverage_ingest__ingest,
    t__modules,
    t__modules__scan,
    t__modules__write_repo_map,
    t__tests_ingest,
    t__tests_ingest__ingest,
    t__typing,
    t__typing__ingest,
)
from codeintel.build.hamilton.native.ingestion.scip import (
    ScipRunResult,
    scip__index_artifact,
    scip__json_artifact,
    t__scip,
    t__scip__run,
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
    "ScipRunResult",
    "TestsIngestResult",
    "TypingIngestResult",
    "scip__index_artifact",
    "scip__json_artifact",
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
    "t__scip__run",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__typing",
    "t__typing__ingest",
]
