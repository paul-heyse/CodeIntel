"""Hamilton support-node generation.

The build system uses native Hamilton `t__*` target nodes. This package
provides *mechanically derived* support nodes (datasets/loaders/artifacts)
generated from target contracts.
"""

from __future__ import annotations

from codeintel.build.hamilton.nodes.support_factory import (
    SupportGenerationOptions,
    build_support_module,
    clear_support_module_cache,
    get_support_module,
)

__all__ = [
    "SupportGenerationOptions",
    "build_support_module",
    "clear_support_module_cache",
    "get_support_module",
]
