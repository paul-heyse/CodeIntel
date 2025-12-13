"""Infrastructure utilities for ingestion pipelines.

This package provides foundational infrastructure used across the ingestion system:

- `paths`: Repository-relative path normalization and module name conversion
- `scanning`: File discovery with glob patterns and ignore lists
- `workers`: Worker pool infrastructure for parallel processing
- `cst_utils`: CST visitor helpers for LibCST-based parsing
- `ast_utils`: AST parsing and span lookup utilities
- `db_queries`: Safe database query helpers

NOTE: This package was renamed from 'utilities' to 'infrastructure' for alignment
with the graphs package structure (compute/ vs infrastructure concerns).
"""

from __future__ import annotations

from codeintel.ingestion.engine._scip_resolver import (
    ResolvedScipConfig,
    ScipPathConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
from codeintel.ingestion.infrastructure.ast_utils import (
    AstSpanIndex,
    parse_python_module,
    timed_parse,
)
from codeintel.ingestion.infrastructure.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
    LineIndexedSource,
)
from codeintel.ingestion.infrastructure.db_queries import (
    DUCKDB_QUERY_ERRORS,
    ColumnNotFoundError,
    QueryError,
    TableNotFoundError,
    safe_count,
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
    safe_table_exists,
)
from codeintel.ingestion.infrastructure.paths import (
    ensure_repo_root,
    normalize_rel_path,
    relpath_to_module,
    repo_relpath,
)
from codeintel.ingestion.infrastructure.scanning import (
    DEFAULT_IGNORE_DIRS,
    IGNORES,
    ScanProfile,
    SourceScanner,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.infrastructure.workers import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)

__all__ = [
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MIN_WORKERS",
    "DUCKDB_QUERY_ERRORS",
    "IGNORES",
    "AstSpanIndex",
    "ColumnNotFoundError",
    "CstCaptureConfig",
    "CstCaptureVisitor",
    "LineIndexedSource",
    "QueryError",
    "ResolvedScipConfig",
    "ScanProfile",
    "ScipPathConfig",
    "ScipResolverInput",
    "SourceScanner",
    "TableNotFoundError",
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
    "safe_count",
    "safe_count_duplicates",
    "safe_count_non_positive",
    "safe_count_nulls",
    "safe_count_orphan_refs",
    "safe_count_with_scope",
    "safe_get_columns",
    "safe_max_value",
    "safe_min_value",
    "safe_not_null_fraction",
    "safe_table_exists",
    "timed_parse",
    "worker_pool",
]
