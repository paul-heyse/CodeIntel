"""Symbol uses plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolUsesOptions:
    """Configuration options for symbol use graph construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_tests
        Whether to include test files when building symbol use edges.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True


__all__ = ["SymbolUsesOptions"]
