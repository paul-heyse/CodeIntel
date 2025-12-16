"""Backward compatibility alias for enforcement_hook.

This module has been moved to codeintel.build.hamilton.hooks.contract_hook.
All imports are re-exported for backward compatibility.

.. deprecated::
    Import from codeintel.build.hamilton.hooks instead.
"""

from __future__ import annotations

# Re-export everything from the new location for backward compatibility
from codeintel.build.hamilton.hooks.contract_hook import ContractEnforcementHook

__all__ = ["ContractEnforcementHook"]
