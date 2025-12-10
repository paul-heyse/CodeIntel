"""Compatibility shim for pipelines module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.project`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.pipelines import PipelineConfig, execute_batch

    # New (preferred):
    from codeintel.cli.project import PipelineConfig, execute_batch
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.pipelines' is deprecated. "
    "Use 'codeintel.cli.project' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.project.pipelines import (
    BatchItemResult,
    BatchOperation,
    BatchResult,
    PipelineConfig,
    StreamingRenderer,
    execute_batch,
    load_batch,
    stream_results,
)

__all__ = [
    "BatchItemResult",
    "BatchOperation",
    "BatchResult",
    "PipelineConfig",
    "StreamingRenderer",
    "execute_batch",
    "load_batch",
    "stream_results",
]
