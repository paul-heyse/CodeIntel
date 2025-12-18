"""Typed facade for Ibis gateway access.

This module provides a small, storage-owned wrapper around ``gateway.ibis.table(...)``.

Why this exists
---------------
Even though ``StorageGateway.ibis.table(...)`` is already typed, many higher-level call sites
end up repeating table acquisition patterns or introducing casts to satisfy type checkers.
Centralizing the "table acquisition" seam here:

- Keeps table access uniform across storage/build/analytics call sites.
- Avoids scattering "table typing" casts outside of the canonical typing seams.
- Preserves storage ownership of gateway semantics (core must not depend on storage).
"""

from __future__ import annotations

import ibis.expr.types as ir

from codeintel.storage.gateway.protocol import MinimalGateway, StorageGateway

__all__ = ["table"]


def table(gateway: MinimalGateway | StorageGateway, table_key: str) -> ir.Table:
    """Return an Ibis table expression for a fully qualified table key.

    Parameters
    ----------
    gateway
        Storage gateway providing the Ibis adapter.
    table_key
        Fully qualified table/view key (e.g., ``analytics.function_metrics``).

    Returns
    -------
    ir.Table
        Ibis table expression for the requested object.
    """
    return gateway.ibis.table(table_key)
