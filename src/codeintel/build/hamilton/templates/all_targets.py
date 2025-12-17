"""Provide template Hamilton nodes for all build targets.

Native Hamilton modules override these templates via Hamilton's module override
semantics (see `driver.Builder().allow_module_overrides()`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.nodes.support_factory import (
    SupportGenerationOptions,
    get_support_module,
)

if TYPE_CHECKING:
    from types import ModuleType


def get_template_module() -> ModuleType:
    """Return the module containing template nodes for all build targets.

    Returns
    -------
    ModuleType
        A Python module object containing template Hamilton nodes for the full
        build graph.
    """
    options = SupportGenerationOptions(
        include_target_stubs=True,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    return get_support_module(options=options)


__all__ = [
    "get_template_module",
]
