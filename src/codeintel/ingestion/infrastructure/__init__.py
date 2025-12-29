"""Infrastructure utilities for ingestion pipelines.

This package provides foundational infrastructure used across the ingestion system:

- `scanning`: File discovery with glob patterns and ignore lists
- `cst_utils`: CST visitor helpers for LibCST-based parsing

Parsing utilities are now consolidated in core modules:

- Parsing utilities: ``codeintel.core.parsing``
- Path utilities: ``codeintel.core.paths``
- Worker utilities: ``codeintel.core.concurrency``

Database query helpers are available at ``codeintel.storage.queries.safe``.
"""

from __future__ import annotations

from codeintel.core.concurrency import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    WorkerConfig,
    create_executor,
    executor_factory,
    resolve_worker_count,
    worker_pool,
)
from codeintel.core.parsing import AstSpanIndex
from codeintel.core.paths import ensure_repo_root, repo_relpath
from codeintel.ingestion.engine._scip_resolver import (
    ResolvedScipConfig,
    ScipPathConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
from codeintel.ingestion.infrastructure.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
    LineIndexedSource,
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

__all__ = [
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
    "ScipPathConfig",
    "ScipResolverInput",
    "SourceScanner",
    "WorkerConfig",
    "create_executor",
    "default_code_profile",
    "default_config_profile",
    "ensure_repo_root",
    "executor_factory",
    "profile_from_env",
    "repo_relpath",
    "resolve_scip_inputs",
    "resolve_worker_count",
    "worker_pool",
]
