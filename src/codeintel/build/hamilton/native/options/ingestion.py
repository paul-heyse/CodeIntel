"""Configuration options for ingestion Hamilton native modules.

These dataclasses configure the behavior of ingestion targets such as
module scanning, SCIP indexing, and related operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "BytecodeExtractOptions",
    "InspectExtractOptions",
    "ModuleIngestOptions",
    "ScipIngestOptions",
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
    max_file_size_kb: int = 1024


@dataclass(frozen=True)
class ScipIngestOptions:
    """Configuration options for SCIP indexing.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_references
        Whether to include symbol references in output.
    include_implementations
        Whether to include implementation relationships.
    max_file_size_kb
        Maximum file size to process.
    timeout_seconds
        Timeout for SCIP indexing operation.
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
    include_references: bool = True
    include_implementations: bool = True
    max_file_size_kb: int = 1024
    timeout_seconds: int = 300
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
    """

    emit_ast_nodes: bool = True


@dataclass(frozen=True)
class BytecodeExtractOptions:
    """Configuration options for bytecode extraction.

    Attributes
    ----------
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
    """

    optimize: int = 0
    show_caches: bool = True
    adaptive: bool = False
    include_exception_table: bool = True
    include_cfg: bool = True
    include_defuse: bool = True


@dataclass(frozen=True)
class InspectExtractOptions:
    """Configuration options for inspect extraction.

    Attributes
    ----------
    enable
        Whether to enable inspect extraction (disabled by default).
    module_allowlist
        Explicit module names allowed for import/inspection.
    max_objects
        Maximum number of objects to inspect.
    follow_wrapped
        Whether to follow wrapper chains for signatures.
    eval_str
        Whether to evaluate string annotations.
    """

    enable: bool = False
    module_allowlist: list[str] = field(default_factory=list)
    max_objects: int = 5000
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
