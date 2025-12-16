"""Configuration options for ingestion Hamilton native modules.

These dataclasses configure the behavior of ingestion targets such as
module scanning, SCIP indexing, and related operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ModuleIngestOptions",
    "ScipIngestOptions",
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
    """

    scope_paths: list[str] | None = None
    include_references: bool = True
    include_implementations: bool = True
    max_file_size_kb: int = 1024
    timeout_seconds: int = 300
    scip_output_dir: Path | None = None

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
