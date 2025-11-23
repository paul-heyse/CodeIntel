"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import duckdb
from mcp.server.fastmcp import (
    FastMCP,  # official Python SDK quickstart :contentReference[oaicite:7]{index=7}
)

from codeintel.mcp.backend import QueryBackend, create_backend
from codeintel.mcp.config import McpServerConfig
from codeintel.mcp.registry import register_tools


def _env_path(name: str, default: str) -> Path:
    v = os.environ.get(name, default)
    return Path(v).expanduser().resolve()


def _build_backend(cfg: McpServerConfig) -> tuple[QueryBackend, Callable[[], None]]:
    """
    Build a QueryBackend from config, opening a DuckDB connection when local.

    Parameters
    ----------
    cfg:
        Server configuration derived from environment variables.

    Returns
    -------
    tuple[QueryBackend, Callable[[], None]]
        Backend instance and a shutdown hook that closes resources.

    Raises
    ------
    ValueError
        If required configuration such as db_path is missing.
    """
    if cfg.mode == "local_db":
        if cfg.db_path is None:
            message = "db_path is required for local_db mode"
            raise ValueError(message)
        connection = duckdb.connect(str(cfg.db_path), read_only=True)
        backend = create_backend(cfg, con=connection)

        def _close() -> None:
            connection.close()

        return backend, _close
    backend = create_backend(cfg)
    close = getattr(backend, "close", lambda: None)
    return backend, close


cfg = McpServerConfig.from_env()
backend, _close = _build_backend(cfg)

# Create the MCP server; json_response=True returns plain JSON in results. :contentReference[oaicite:8]{index=8}
mcp = FastMCP("CodeIntel", json_response=True)
register_tools(mcp, backend)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    """
    Run the CodeIntel MCP server.

    By default this uses stdio transport, which is what Cursor and the
    OpenAI CLI expect for local MCP servers. :contentReference[oaicite:14]{index=14}
    """
    try:
        mcp.run()  # stdio by default
    finally:
        _close()


if __name__ == "__main__":
    main()
