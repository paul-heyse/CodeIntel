"""Native ingestion targets with tool execution subgraphs.

Phase 2: All ingestion domain targets migrated from legacy wrappers to native
Hamilton modules with boundary validation handled at storage I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

if TYPE_CHECKING:
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
        ModuleToolOutput,
        TestsToolOutput,
        TypingToolOutput,
        t__config_ingest,
        t__config_ingest__ingest,
        t__config_ingest__run,
        t__config_ingest__scan,
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
    from codeintel.build.hamilton.native.ingestion.scip_resolution import (
        t__scip_resolution,
    )
    from codeintel.build.hamilton.native.ingestion.tree_sitter import (
        t__tree_sitter_index,
        t__tree_sitter_index__ingest,
        t__tree_sitter_index__run,
    )

_LAZY_IMPORTS = {
    "ConfigScanResult": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "ConfigToolOutput": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "ModuleToolOutput": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "ScipProtoRunResult": "codeintel.build.hamilton.native.ingestion.scip_proto",
    "ScipRunResult": "codeintel.build.hamilton.native.ingestion.scip",
    "TestsToolOutput": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "TypingToolOutput": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "scip__index_artifact": "codeintel.build.hamilton.native.ingestion.scip",
    "scip__proto_module_path": "codeintel.build.hamilton.native.ingestion.scip_proto",
    "t__ast": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__ast__ingest": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__ast__run": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__config_ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__config_ingest__ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__config_ingest__run": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__config_ingest__scan": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__cst": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__cst__ingest": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__cst__run": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__docstrings": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__docstrings__ingest": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__docstrings__run": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__file_line_index": "codeintel.build.hamilton.native.ingestion.file_line_index",
    "t__modules": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__modules__ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__modules__run": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__scip": "codeintel.build.hamilton.native.ingestion.scip",
    "t__scip__ingest": "codeintel.build.hamilton.native.ingestion.scip",
    "t__scip__run": "codeintel.build.hamilton.native.ingestion.scip",
    "t__scip_proto": "codeintel.build.hamilton.native.ingestion.scip_proto",
    "t__scip_proto__run": "codeintel.build.hamilton.native.ingestion.scip_proto",
    "t__scip_resolution": "codeintel.build.hamilton.native.ingestion.scip_resolution",
    "t__syntax_index": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__syntax_index__ingest": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__syntax_index__run": "codeintel.build.hamilton.native.ingestion.extraction_targets",
    "t__tests_ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__tests_ingest__ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__tests_ingest__run": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__tree_sitter_index": "codeintel.build.hamilton.native.ingestion.tree_sitter",
    "t__tree_sitter_index__ingest": "codeintel.build.hamilton.native.ingestion.tree_sitter",
    "t__tree_sitter_index__run": "codeintel.build.hamilton.native.ingestion.tree_sitter",
    "t__typing": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__typing__ingest": "codeintel.build.hamilton.native.ingestion.ingest_targets",
    "t__typing__run": "codeintel.build.hamilton.native.ingestion.ingest_targets",
}


def __getattr__(name: str) -> object:
    """Lazy import module exports to avoid circular import chains.

    Returns
    -------
    object
        Requested attribute from the lazily imported module.

    Raises
    ------
    AttributeError
        If the attribute is not defined by this module.
    """
    if name in _LAZY_IMPORTS:
        module = lazy_import(_LAZY_IMPORTS[name])
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

__all__: list[str] = [
    "ConfigScanResult",
    "ConfigToolOutput",
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
    "t__scip_resolution",
    "t__syntax_index",
    "t__syntax_index__ingest",
    "t__syntax_index__run",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__tests_ingest__run",
    "t__tree_sitter_index",
    "t__tree_sitter_index__ingest",
    "t__tree_sitter_index__run",
    "t__typing",
    "t__typing__ingest",
    "t__typing__run",
]
