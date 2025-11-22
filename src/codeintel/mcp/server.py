"""MCP server exposing CodeIntel datasets and tools."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from mcp.server.fastmcp import (
    FastMCP,  # official Python SDK quickstart :contentReference[oaicite:7]{index=7}
)

from codeintel.mcp.backend import DuckDBBackend


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


# --------------------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------------------


@mcp.tool()
def get_function_summary(
    urn: str | None = None,
    goid_h128: int | None = None,
    rel_path: str | None = None,
    qualname: str | None = None,
) -> dict:
    """
    Look up a Python function/method in this repo and return a rich summary.

    You MUST provide either:
      - urn (preferred), or
      - goid_h128, or
      - rel_path AND qualname.

    The returned object includes risk, coverage, and type info aggregated
    from docs.v_function_summary and analytics tables.

    Returns
    -------
    dict
        Response payload containing a `found` flag and `summary` row.
    """
    row = backend.get_function_summary(
        urn=urn,
        goid_h128=goid_h128,
        rel_path=rel_path,
        qualname=qualname,
    )
    return {"found": bool(row), "summary": row}


@mcp.tool()
def list_high_risk_functions(
    min_risk: float = 0.7,
    limit: int = 50,
    tested_only: bool = False,
) -> list[dict]:
    """
    Return functions whose risk_score >= min_risk, sorted by risk_score descending.

    Use this to ask things like:
      - "What are the riskiest functions in this repo?"
      - "Where should I write tests first?"

    Fields include:
      - urn, rel_path, qualname
      - risk_score, risk_level
      - coverage_ratio, tested
      - complexity_bucket, typedness_bucket, hotspot_score.

    Returns
    -------
    list[dict]
        Function summaries ordered by risk_score.
    """
    return backend.list_high_risk_functions(
        min_risk=min_risk,
        limit=limit,
        tested_only=tested_only,
    )


@mcp.tool()
def get_callgraph_neighbors(
    goid_h128: int,
    direction: str = "both",  # "in" | "out" | "both"
    limit: int = 50,
) -> dict:
    """
    Get static call graph neighbors from `docs.v_call_graph_enriched`.

    direction:
      - "out": functions this function calls
      - "in":  functions that call this function
      - "both": both directions (default)

    Returns
    -------
    dict
        Neighboring call graph edges keyed by direction.

    Raises
    ------
    ValueError
        If `direction` is not one of {"in", "out", "both"}.
    """
    if direction not in ("in", "out", "both"):
        raise ValueError('direction must be "in", "out", or "both"')

    return backend.get_callgraph_neighbors(
        goid_h128=goid_h128,
        direction=direction,
        limit=limit,
    )


@mcp.tool()
def get_tests_for_function(
    goid_h128: int | None = None,
    urn: str | None = None,
) -> list[dict]:
    """
    List tests that exercise a given function from docs.v_test_to_function.

    Returns
    -------
    list[dict]
        Test rows covering the function.
    """
    return backend.get_tests_for_function(goid_h128=goid_h128, urn=urn)


@mcp.tool()
def get_file_summary(
    rel_path: str,
) -> dict:
    """
    Summarize a file and its functions using `docs.v_file_summary`.

    Useful for:
      - "Tell me about this file, its hotspot status, and key functions."
      - "Which functions in this file are high risk or poorly tested?"

    Returns
    -------
    dict
        Payload with `found` flag and file summary row when present.
    """
    row = backend.get_file_summary(rel_path=rel_path)
    return {"found": bool(row), "file": row}


# ---------- Dataset browsing as MCP tools (lightweight alternative to resources) -----


@mcp.tool()
def list_datasets() -> list[dict]:
    """
    List key datasets (tables/views) available for inspection.

    The resulting entries are intended for small samples fetched via
    `read_dataset_rows` rather than bulk export.

    Returns
    -------
    list[dict]
        Dataset descriptors suitable for sampling.
    """
    return backend.list_datasets()


@mcp.tool()
def read_dataset_rows(
    dataset_name: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return a small slice of a dataset as JSON rows.

    Use this sparingly; large scans should go through Parquet/JSONL artifacts
    under Document Output/, not through MCP.

    Returns
    -------
    list[dict]
        Rows from the requested dataset limited by `limit` and `offset`.
    """
    return backend.read_dataset_rows(dataset_name=dataset_name, limit=limit, offset=offset)


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
