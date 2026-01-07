"""Shared types for CFG/DFG analytics.

This module provides base types and protocols used by both cfg_core.py
and dfg_core.py for type-safe context handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from codeintel.build.graphs.rx.algos import GraphInput


@runtime_checkable
class FnContextProtocol(Protocol):
    """Protocol defining common fields for CFG and DFG function contexts.

    Both CfgFnContext and DfgFnContext implement this protocol, allowing
    code to work with either context type when only the common fields
    are needed.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    fn_goid
        Function global object identifier.
    rel_path
        Relative file path.
    module
        Module name (may be None).
    qualname
        Qualified name of the function.
    graph
        Directed graph representing the flow.
    sccs
        Strongly connected components in the graph.
    now
        Timestamp for created_at fields.
    """

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...

    @property
    def fn_goid(self) -> int:
        """Function global object identifier."""
        ...

    @property
    def rel_path(self) -> str:
        """Relative file path."""
        ...

    @property
    def module(self) -> str | None:
        """Module name (may be None)."""
        ...

    @property
    def qualname(self) -> str | None:
        """Qualified name of the function."""
        ...

    @property
    def graph(self) -> GraphInput:
        """Directed graph representing the flow."""
        ...

    @property
    def sccs(self) -> list[set[int]]:
        """Strongly connected components in the graph."""
        ...

    @property
    def now(self) -> datetime:
        """Timestamp for created_at fields."""
        ...


__all__ = ["FnContextProtocol"]
