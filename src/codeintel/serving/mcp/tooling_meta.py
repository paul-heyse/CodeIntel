"""Compatibility shim for tooling metadata helpers.

The canonical implementation lives under `codeintel.serving.meta.tooling`.
"""

from codeintel.serving.meta.tooling import runtime_versions, tooling_mismatch_warnings

__all__ = ["runtime_versions", "tooling_mismatch_warnings"]
