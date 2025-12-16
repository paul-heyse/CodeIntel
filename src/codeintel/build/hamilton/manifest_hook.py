"""Backward compatibility alias for manifest_hook.

This module has been moved to codeintel.build.hamilton.hooks.manifest_hook.
All imports are re-exported for backward compatibility.

.. deprecated::
    Import from codeintel.build.hamilton.hooks instead.
"""

from __future__ import annotations

# Re-export everything from the new location for backward compatibility
from codeintel.build.hamilton.hooks.manifest_hook import (
    ManifestSaveRequest,
    SkipCheckRequest,
    TargetRunRecord,
    compute_target_input_hash,
    compute_target_input_hash_with_deps,
    compute_target_options_hash,
    save_manifest,
    should_skip,
)

__all__ = [
    "ManifestSaveRequest",
    "SkipCheckRequest",
    "TargetRunRecord",
    "compute_target_input_hash",
    "compute_target_input_hash_with_deps",
    "compute_target_options_hash",
    "save_manifest",
    "should_skip",
]
