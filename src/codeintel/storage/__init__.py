"""Shared storage abstractions and constants for CodeIntel persistence backends.

Row models and serializers are now defined in codeintel.config.dataset_contract.
This module was previously a re-export facade, but all row model imports should
now go directly to codeintel.config.dataset_contract for explicit sourcing.
"""

from __future__ import annotations
