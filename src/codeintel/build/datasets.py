"""Compatibility re-exports for build dataset references.

Historically, some build submodules imported :class:`DatasetRef` from
``codeintel.build.datasets``. The canonical type now lives under the
Hamilton IO layer.
"""

from __future__ import annotations

from codeintel.build.hamilton.io.dataset_ref import (
    DatasetRef,
    refs_from_target_result,
    refs_to_tuple,
)

__all__ = [
    "DatasetRef",
    "refs_from_target_result",
    "refs_to_tuple",
]
