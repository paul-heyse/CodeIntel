"""SCIP ingest plugin options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScipIngestOptions:
    """Configuration options for SCIP indexing."""

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


__all__ = ["ScipIngestOptions"]
