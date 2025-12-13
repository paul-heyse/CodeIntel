"""GOID builder plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoidBuilderOptions:
    """Configuration options for GOID construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_tests
        Whether to include test files.
    include_private
        Whether to include symbols whose names start with an underscore.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True
    include_private: bool = True


__all__ = ["GoidBuilderOptions"]
