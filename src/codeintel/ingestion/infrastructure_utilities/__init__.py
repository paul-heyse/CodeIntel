"""Infrastructure utilities for ingestion pipelines.

This package provides foundational utilities used across the ingestion system:

- `paths`: Repository-relative path normalization and module name conversion
- `source_scanner`: File discovery with glob patterns and ignore lists
- `tool_runner`: Structured execution of external tools (pyright, ruff, etc.)
- `workers`: Worker pool infrastructure for parallel processing
- `cst_utils`: CST visitor helpers for LibCST-based parsing
- `ast_utils`: AST parsing and span lookup utilities
- `_scip_resolver`: SCIP ingestion input resolution
"""

from __future__ import annotations

# SCIP resolver utilities
from codeintel.ingestion.infrastructure_utilities._scip_resolver import (
    ResolvedScipConfig,
    resolve_scip_inputs,
)

# AST utilities
from codeintel.ingestion.infrastructure_utilities.ast_utils import (
    AstSpanIndex,
    parse_python_module,
    timed_parse,
)

# CST utilities
from codeintel.ingestion.infrastructure_utilities.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
    LineIndexedSource,
)

# Path utilities
from codeintel.ingestion.infrastructure_utilities.paths import (
    ensure_repo_root,
    normalize_rel_path,
    relpath_to_module,
    repo_relpath,
)

# Source scanning utilities
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    DEFAULT_IGNORE_DIRS,
    IGNORES,
    ScanProfile,
    SourceScanner,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)

# Tool runner utilities
from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolResult,
    ToolRunner,
    ToolRunResult,
)

# Worker pool utilities
from codeintel.ingestion.infrastructure_utilities.workers import (
    AST_WORKER_CONFIG,
    CST_WORKER_CONFIG,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)

__all__ = [
    "AST_WORKER_CONFIG",
    "CST_WORKER_CONFIG",
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MIN_WORKERS",
    "IGNORES",
    "AstSpanIndex",
    "CstCaptureConfig",
    "CstCaptureVisitor",
    "LineIndexedSource",
    "ResolvedScipConfig",
    "ScanProfile",
    "SourceScanner",
    "ToolExecutionError",
    "ToolName",
    "ToolNotFoundError",
    "ToolResult",
    "ToolRunResult",
    "ToolRunner",
    "WorkerConfig",
    "create_executor",
    "default_code_profile",
    "default_config_profile",
    "ensure_repo_root",
    "executor_factory",
    "normalize_rel_path",
    "parse_python_module",
    "profile_from_env",
    "relpath_to_module",
    "repo_relpath",
    "resolve_scip_inputs",
    "resolve_worker_count",
    "timed_parse",
    "worker_pool",
]
