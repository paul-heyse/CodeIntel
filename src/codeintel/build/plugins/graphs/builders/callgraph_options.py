"""Call graph plugin options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallGraphOptions:
    """Configuration options for call graph construction."""

    scope_paths: list[str] | None = None
    max_edges_per_file: int = 10000
    include_external_calls: bool = True
    resolve_imports: bool = True
    use_libcst: bool = True


__all__ = ["CallGraphOptions"]
