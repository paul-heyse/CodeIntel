"""Semantic-first serving surfaces.

The serving layer exposes CodeIntel's immutable, published DuckDB snapshots via:

- HTTP (FastAPI): `codeintel.serving.http.app:create_serving_app`
- MCP (FastMCP): `codeintel.serving.mcp.server:main`

Runtime state is derived from the atomic `current.json` pointer under
`CODEINTEL_SERVE_DIR` (default: `.codeintel/serve`).
"""

from __future__ import annotations

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.db.pool import DuckDBPoolConfig, DuckDBReadPool
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.mcp.server import create_mcp_server
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings

__all__ = [
    "DuckDBPoolConfig",
    "DuckDBReadPool",
    "SemanticQueryKernel",
    "ServingDBManager",
    "ServingSettings",
    "ServingSnapshotPointer",
    "build_mcp_app",
    "create_mcp_server",
    "create_serving_app",
]
