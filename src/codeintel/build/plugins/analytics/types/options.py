"""Type coverage plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeCoverageOptions:
    """Configuration options for type coverage analysis.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only analyze files within these paths.
    include_private : bool
        Whether to include private functions in coverage.
    strictness : str
        Type checking strictness ("strict", "standard", "lenient").
    """

    scope_paths: list[str] | None = None
    include_private: bool = True
    strictness: str = "standard"


__all__ = ["TypeCoverageOptions"]
