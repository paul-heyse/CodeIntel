"""Deprecated: Use codeintel.analytics.compute.functions.typedness instead.

This module is a re-export shim for backward compatibility. All functionality
has been moved to the compute layer at `codeintel.analytics.compute.functions.typedness`.

The new location provides the same API but is organized as part of the
pure computation layer that separates business logic from I/O concerns.
"""

from __future__ import annotations

import warnings

from codeintel.analytics.compute.functions.typedness import (
    SKIP_PARAM_NAMES,
    ParamStats,
    TypednessFlags,
    compute_param_stats,
    compute_typedness_flags,
)

warnings.warn(
    "codeintel.analytics.functions.typedness is deprecated. "
    "Use codeintel.analytics.compute.functions.typedness instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SKIP_PARAM_NAMES",
    "ParamStats",
    "TypednessFlags",
    "compute_param_stats",
    "compute_typedness_flags",
]
