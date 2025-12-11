"""Import graph plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportGraphOptions:
    """Configuration options for import graph construction.

    Attributes
    ----------
    scope_paths : list[str] | None
        If set, only process files within these paths.
    include_stdlib : bool
        Whether to include stdlib imports in the graph.
    include_third_party : bool
        Whether to include third-party imports.
    resolve_dynamic : bool
        Whether to attempt resolution of dynamic imports.
    """

    scope_paths: list[str] | None = None
    include_stdlib: bool = False
    include_third_party: bool = True
    resolve_dynamic: bool = False


__all__ = ["ImportGraphOptions"]
