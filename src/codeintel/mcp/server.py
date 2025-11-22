"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from mcp.server.fastmcp import (
    FastMCP,  # official Python SDK quickstart :contentReference[oaicite:7]{index=7}
)

from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.registry import register_tools


def _env_path(name: str, default: str) -> Path:
    v = os.environ.get(name, default)
    return Path(v).expanduser().resolve()


# --------------------------------------------------------------------------------------
# Server initialization
# --------------------------------------------------------------------------------------

# Basic env-driven config; you can tighten this later or hook it into CodeIntelConfig.
REPO_ROOT = _env_path("CODEINTEL_REPO_ROOT", ".")
DB_PATH = _env_path("CODEINTEL_DB_PATH", str(REPO_ROOT / "build" / "db" / "codeintel.duckdb"))
REPO_SLUG = os.environ.get("CODEINTEL_REPO", REPO_ROOT.name)
COMMIT_SHA = os.environ.get("CODEINTEL_COMMIT", "HEAD")

# One read-only DuckDB connection per server process
_con = duckdb.connect(str(DB_PATH), read_only=True)

backend = DuckDBBackend(con=_con, repo=REPO_SLUG, commit=COMMIT_SHA)

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
    mcp.run()  # stdio by default


if __name__ == "__main__":
    main()
