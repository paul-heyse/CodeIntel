"""Template Hamilton modules for the build DAG.

This package contains **template** Hamilton nodes that provide a fallback
implementation for all build targets. Native target modules override these
templates via Hamilton's module override semantics.
"""

from __future__ import annotations

from types import ModuleType

from codeintel.build.hamilton.templates.all_targets import get_template_module

__all__ = [
    "ModuleType",
    "get_template_module",
]
