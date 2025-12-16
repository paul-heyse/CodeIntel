"""Deprecated serving-layer pool shims.

Serving no longer owns connection pooling primitives. Import pool types from
``codeintel.storage.gateway.pool`` instead.
"""

from __future__ import annotations

from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse

DuckDBPoolConfig = PoolConfig
DuckDBReadPool = ReadPoolWarehouse

__all__ = ["DuckDBPoolConfig", "DuckDBReadPool", "PoolConfig", "ReadPoolWarehouse"]
