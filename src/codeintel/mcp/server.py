"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

from collections.abc import Callable

import duckdb
from mcp.server.fastmcp import FastMCP

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.backend import QueryBackend, create_backend
from codeintel.mcp.registry import register_tools


def _build_backend(cfg: ServingConfig) -> tuple[QueryBackend, Callable[[], None]]:
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


def create_mcp_server(cfg: ServingConfig | None = None) -> tuple[FastMCP, Callable[[], None]]:
    """
    Create the MCP server instance plus shutdown hook.

    Parameters
    ----------
    cfg:
        Optional pre-loaded ServingConfig. When omitted, environment variables are used.

    Returns
    -------
    tuple[FastMCP, Callable[[], None]]
        Configured MCP server and shutdown callback.
    """
    config = cfg or ServingConfig.from_env()
    backend, close = _build_backend(config)
    server = FastMCP("CodeIntel", json_response=True)
    service = getattr(backend, "service", None)
    register_tools(server, service or backend)
    return server, close


def main() -> None:
    """
    Run the CodeIntel MCP server.

    By default this uses stdio transport, which is what Cursor and the
    OpenAI CLI expect for local MCP servers. :contentReference[oaicite:14]{index=14}
    """
    server, close = create_mcp_server()
    try:
        server.run()  # stdio by default
    finally:
        close()


if __name__ == "__main__":
    main()
