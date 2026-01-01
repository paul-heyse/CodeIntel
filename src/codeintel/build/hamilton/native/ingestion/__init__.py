"""Native ingestion targets with tool execution subgraphs.

Phase 2: All ingestion domain targets migrated from legacy wrappers to native
Hamilton modules with boundary validation handled at storage I/O.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.ingestion.extraction_targets import (
    t__ast,
    t__ast__ingest,
    t__ast__run,
    t__cst,
    t__cst__ingest,
    t__cst__run,
    t__docstrings,
    t__docstrings__ingest,
    t__docstrings__run,
    t__syntax_index,
    t__syntax_index__ingest,
    t__syntax_index__run,
)
from codeintel.build.hamilton.native.ingestion.file_line_index import (
    t__file_line_index,
)
from codeintel.build.hamilton.native.ingestion.ingest_targets import (
    ConfigScanResult,
    ConfigToolOutput,
    CoverageToolOutput,
    ModuleToolOutput,
    TestsToolOutput,
    TypingToolOutput,
    t__config_ingest,
    t__config_ingest__ingest,
    t__config_ingest__run,
    t__config_ingest__scan,
    t__coverage_ingest,
    t__coverage_ingest__ingest,
    t__coverage_ingest__run,
    t__modules,
    t__modules__ingest,
    t__modules__run,
    t__tests_ingest,
    t__tests_ingest__ingest,
    t__tests_ingest__run,
    t__typing,
    t__typing__ingest,
    t__typing__run,
)
from codeintel.build.hamilton.native.ingestion.scip import (
    ScipRunResult,
    scip__index_artifact,
    t__scip,
    t__scip__ingest,
    t__scip__run,
)
from codeintel.build.hamilton.native.ingestion.scip_proto import (
    ScipProtoRunResult,
    scip__proto_module_path,
    t__scip_proto,
    t__scip_proto__run,
)

__all__: list[str] = [
    "ConfigScanResult",
    "ConfigToolOutput",
    "CoverageToolOutput",
    "ModuleToolOutput",
    "ScipProtoRunResult",
    "ScipRunResult",
    "TestsToolOutput",
    "TypingToolOutput",
    "scip__index_artifact",
    "scip__proto_module_path",
    "t__ast",
    "t__ast__ingest",
    "t__ast__run",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__run",
    "t__config_ingest__scan",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
    "t__coverage_ingest__run",
    "t__cst",
    "t__cst__ingest",
    "t__cst__run",
    "t__docstrings",
    "t__docstrings__ingest",
    "t__docstrings__run",
    "t__file_line_index",
    "t__modules",
    "t__modules__ingest",
    "t__modules__run",
    "t__scip",
    "t__scip__ingest",
    "t__scip__run",
    "t__scip_proto",
    "t__scip_proto__run",
    "t__syntax_index",
    "t__syntax_index__ingest",
    "t__syntax_index__run",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__tests_ingest__run",
    "t__typing",
    "t__typing__ingest",
    "t__typing__run",
]
