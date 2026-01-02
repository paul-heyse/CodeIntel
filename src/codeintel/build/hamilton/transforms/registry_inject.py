"""Registry-only inject helpers for Hamilton wiring."""

from __future__ import annotations

from hamilton.function_modifiers import inject, source
from hamilton.function_modifiers.base import NodeTransformLifecycle


def inject_from_registry(*, param_name: str, node_name: str) -> NodeTransformLifecycle:
    """Return an inject decorator that wires a parameter to a registry node.

    Returns
    -------
    NodeTransformLifecycle
        Decorator that maps the parameter to the registry node.
    """
    return inject(**{param_name: source(node_name)})


__all__ = ["inject_from_registry"]
