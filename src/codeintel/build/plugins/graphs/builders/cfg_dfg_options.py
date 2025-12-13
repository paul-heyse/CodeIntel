"""CFG/DFG builder plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CfgDfgOptions:
    """Configuration options for CFG/DFG construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_test_files
        Whether to include test files when building graphs.
    """

    scope_paths: list[str] | None = None
    include_test_files: bool = True


__all__ = ["CfgDfgOptions"]
