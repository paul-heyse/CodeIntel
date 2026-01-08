"""Configuration options for ingestion Hamilton native modules.

These dataclasses configure the behavior of ingestion targets such as
module scanning, SCIP indexing, and related operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "AstExtractOptions",
    "BytecodeExtractOptions",
    "InspectExtractOptions",
    "ModuleIngestOptions",
    "ScipIngestOptions",
    "SymtableExtractOptions",
    "SyntaxAugmentOptions",
    "SyntaxIndexOptions",
    "TreeSitterIndexOptions",
]


@dataclass(frozen=True)
class ModuleIngestOptions:
    """Configuration options for module ingestion.

    Attributes
    ----------
    scope_paths
        If set, only ingest modules within these paths.
    include_tests
        Whether to include test modules.
    include_generated
        Whether to include generated files.
    max_file_size_kb
        Maximum file size to ingest.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True
    include_generated: bool = False
    max_file_size_kb: int = 102400


@dataclass(frozen=True)
class ScipIngestOptions:
    """Configuration options for SCIP indexing.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    environment_json
        Optional scip-python environment JSON file.
    pyright_config_path
        Optional pyrightconfig.json to stage during indexing.
    project_version_mode
        Project version mode: unset, commit, or constant.
    project_version_value
        Project version value when using constant mode.
    project_namespace
        Optional namespace prefix for project symbols.
    include_references
        Whether to include symbol references in output.
    include_implementations
        Whether to include implementation relationships.
    max_file_size_kb
        Maximum file size to process.
    timeout_seconds
        Timeout for SCIP indexing operation.
    scip_node_max_old_space_mb
        Optional Node.js max old space size (MB) for scip-python.
    scip_output_dir
        Directory to write SCIP index files.
    batch_size
        Target number of modules per incremental shard batch.
    batch_max_bytes
        Optional maximum total size of a batch in bytes.
    full_rebuild_threshold_count
        Count threshold that triggers a full rebuild when exceeded.
    full_rebuild_threshold_ratio
        Ratio threshold (changed/total) that triggers a full rebuild.
    full_rebuild_ratio_min_modules
        Minimum total modules before ratio thresholds are applied.
    full_rebuild_ratio_min_changed
        Minimum changed modules before ratio thresholds are applied.
    """

    scope_paths: list[str] | None = None
    environment_json: Path | None = None
    pyright_config_path: Path | None = None
    project_version_mode: str = "unset"
    project_version_value: str | None = None
    project_namespace: str | None = None
    include_references: bool = True
    include_implementations: bool = True
    max_file_size_kb: int = 102400
    timeout_seconds: int = 300
    scip_node_max_old_space_mb: int | None = None
    scip_output_dir: Path | None = None
    batch_size: int = 200
    batch_max_bytes: int = 50_000_000
    full_rebuild_threshold_count: int = 1000
    full_rebuild_threshold_ratio: float = 0.3
    full_rebuild_ratio_min_modules: int = 200
    full_rebuild_ratio_min_changed: int = 25

    def should_include_references(self) -> bool:
        """Check if references should be included.

        Returns
        -------
        bool
            True when references should be emitted.
        """
        return self.include_references

    def should_include_implementations(self) -> bool:
        """Check if implementations should be included.

        Returns
        -------
        bool
            True when implementations should be emitted.
        """
        return self.include_implementations


@dataclass(frozen=True)
class SyntaxIndexOptions:
    """Configuration options for syntax index extraction.

    Attributes
    ----------
    emit_ast_nodes
        Whether to merge CPython AST facts into syntax nodes.
    batch_size
        Target row count per Arrow RecordBatch when streaming.
    """

    emit_ast_nodes: bool = True
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True)
class AstExtractOptions:
    """Configuration options for AST extraction.

    Attributes
    ----------
    batch_size
        Target row count per Arrow RecordBatch when streaming.
    """

    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True)
class SymtableExtractOptions:
    """Configure symtable extraction behavior.

    Attributes
    ----------
    enable
        Whether to enable symtable extraction.
    batch_size
        Target row count per Arrow RecordBatch when streaming.
    """

    enable: bool = True
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True)
class BytecodeExtractOptions:
    """Configuration options for bytecode extraction.

    Attributes
    ----------
    enable
        Whether to enable bytecode extraction.
    optimize
        Optimization level passed to compile() (0, 1, or 2).
    show_caches
        Whether to request inline cache metadata from dis.
    adaptive
        Whether to emit adaptive instruction variants.
    include_exception_table
        Whether to parse the exception table.
    include_cfg
        Whether to derive CFG blocks/edges.
    include_defuse
        Whether to emit def/use events.
    dont_inherit
        Whether to ignore future flags from the runtime environment during compile.
    compile_flags
        Optional flags passed to compile() for bytecode extraction.
    max_module_bytes
        Maximum module size in bytes to process (None disables the limit).
    max_module_seconds
        Optional per-module wall-clock budget for bytecode extraction.
    max_workers
        Number of worker threads to use for bytecode extraction.
    batch_size
        Target row count per Arrow RecordBatch when streaming.
    enable_cache
        Whether to cache compiled code objects for reuse across runs.
    cache_dir
        Optional directory for compiled bytecode cache files.
    """

    enable: bool = True
    optimize: int = 0
    show_caches: bool = True
    adaptive: bool = False
    include_exception_table: bool = True
    include_cfg: bool = True
    include_defuse: bool = True
    dont_inherit: bool = True
    compile_flags: int = 0
    max_module_bytes: int | None = None
    max_module_seconds: float | None = None
    max_workers: int = 1
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    enable_cache: bool = True
    cache_dir: Path | None = None


@dataclass(frozen=True)
class InspectExtractOptions:
    """Configuration options for inspect extraction.

    Attributes
    ----------
    enable
        Whether to enable inspect extraction (disabled by default).
    module_allowlist
        Explicit module names allowed for import/inspection.
    use_subprocess
        Whether to isolate inspect extraction in a subprocess.
    timeout_seconds
        Timeout for inspect extraction (applies to subprocess runs).
    max_modules
        Maximum number of modules to inspect per run (None disables the limit).
    max_module_bytes
        Maximum module size in bytes to inspect (None disables the limit).
    max_module_seconds
        Optional per-module wall-clock budget for inspect extraction.
    max_memory_mb
        Optional per-run memory ceiling for inspect extraction (megabytes).
    max_objects
        Maximum number of objects to inspect.
    batch_size
        Target row count per Arrow RecordBatch when streaming.
    follow_wrapped
        Whether to follow wrapper chains for signatures.
    eval_str
        Whether to evaluate string annotations.
    """

    enable: bool = False
    module_allowlist: list[str] = field(default_factory=list)
    use_subprocess: bool = True
    timeout_seconds: int = 30
    max_modules: int | None = None
    max_module_bytes: int | None = None
    max_module_seconds: float | None = None
    max_memory_mb: int | None = None
    max_objects: int = 5000
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    follow_wrapped: bool = True
    eval_str: bool = False


@dataclass(frozen=True)
class TreeSitterIndexOptions:
    """Configuration options for tree-sitter indexing.

    Attributes
    ----------
    emit_nodes_edges
        Whether to emit full tree-sitter CST nodes/edges.
    emit_tokens
        Whether to emit token-level captures.
    emit_trivia
        Whether to emit trivia-level captures (comments).
    emit_language_metadata
        Whether to emit language ABI metadata rows.
    enable_incremental
        Whether to enable incremental parsing with cached trees when available.
    match_limit
        Match limit for query execution.
    allow_non_local_patterns
        Whether to allow non-local query patterns.
    """

    emit_nodes_edges: bool = True
    emit_tokens: bool = True
    emit_trivia: bool = True
    emit_language_metadata: bool = True
    enable_incremental: bool = False
    match_limit: int = 10000
    allow_non_local_patterns: bool = False


@dataclass(frozen=True)
class SyntaxAugmentOptions:
    """Configuration options for syntax augmentation.

    Attributes
    ----------
    emit_ts_xref
        Whether to emit tree-sitter to syntax-node xref rows.
    fallback_on_libcst_failure
        Whether to use tree-sitter nodes/edges for LibCST failures.
    """

    emit_ts_xref: bool = True
    fallback_on_libcst_failure: bool = True
