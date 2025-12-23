"""Semantic-first serving surfaces.

The serving layer exposes CodeIntel's immutable, published DuckDB snapshots via:

- HTTP (FastAPI): `codeintel.serving.http.app:create_serving_app`
- MCP (FastMCP): `codeintel.serving.mcp.server:main`

Runtime state is derived from the atomic `current.json` pointer under
`CODEINTEL_SERVE_DIR` (default: `.codeintel/serve`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.imports.lazy import lazy_import

__all__ = [
    "PoolConfig",
    "ReadPoolWarehouse",
    "SemanticQueryKernel",
    "ServingDBManager",
    "ServingSettings",
    "ServingSnapshotPointer",
    "build_mcp_app",
    "create_mcp_server",
    "create_serving_app",
]

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.http.app import create_serving_app
    from codeintel.serving.mcp.app import build_mcp_app
    from codeintel.serving.mcp.server import create_mcp_server
    from codeintel.serving.semantic.kernel import SemanticQueryKernel
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PoolConfig": ("codeintel.storage.gateway.pool", "PoolConfig"),
    "ReadPoolWarehouse": ("codeintel.storage.gateway.pool", "ReadPoolWarehouse"),
    "SemanticQueryKernel": ("codeintel.serving.semantic.kernel", "SemanticQueryKernel"),
    "ServingDBManager": ("codeintel.serving.db.manager", "ServingDBManager"),
    "ServingSettings": ("codeintel.serving.settings", "ServingSettings"),
    "ServingSnapshotPointer": ("codeintel.serving.db.pointer", "ServingSnapshotPointer"),
    "build_mcp_app": ("codeintel.serving.mcp.app", "build_mcp_app"),
    "create_mcp_server": ("codeintel.serving.mcp.server", "create_mcp_server"),
    "create_serving_app": ("codeintel.serving.http.app", "create_serving_app"),
}


def __getattr__(name: str) -> object:
    """Lazily import serving symbols to avoid import-time cycles.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = lazy_import(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
