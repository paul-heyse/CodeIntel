"""Internal module attachment helpers for dynamic node generation."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol, cast


class _NamedCallable(Protocol):
    """Protocol for callables with mutable naming metadata."""

    __name__: str
    __module__: str


def attach_node(module: ModuleType, *, node_name: str, fn: object) -> None:
    """Attach a callable to a module under a stable node name."""
    named = cast("_NamedCallable", fn)
    named.__name__ = node_name
    named.__module__ = module.__name__
    setattr(module, node_name, fn)


__all__ = ["attach_node"]
