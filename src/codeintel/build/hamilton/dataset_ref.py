"""Compatibility re-exports for Hamilton dataset references."""

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
