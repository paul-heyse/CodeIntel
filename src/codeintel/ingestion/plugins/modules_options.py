"""Module ingest plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleIngestOptions:
    """Configuration options for module ingestion.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only ingest modules within these paths.
    include_tests : bool
        Whether to include test modules.
    include_generated : bool
        Whether to include generated files.
    max_file_size_kb : int
        Maximum file size to ingest.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True
    include_generated: bool = False
    max_file_size_kb: int = 1024


__all__ = ["ModuleIngestOptions"]
