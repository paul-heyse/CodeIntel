"""FastAPI server exposing DuckDB-backed CodeIntel metadata views."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.views import create_all_views

DB_PATH = os.getenv("CODEINTEL_DB_PATH", "build/db/codeintel.duckdb")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown lifecycle.

    Parameters
    ----------
    app:
        FastAPI application instance.

    Yields
    ------
    None
        Control back to the application runtime while the DuckDB connection is open.
    """
    con = duckdb.connect(DB_PATH, read_only=False)
    apply_all_schemas(con)
    create_all_views(con)
    app.state.con = con
    try:
        await asyncio.sleep(0)
        yield
    finally:
        con.close()


app = FastAPI(
    title="CodeIntel Metadata API",
    description="Thin API over DuckDB views for AI agents and tools.",
    version="0.1.0",
    lifespan=lifespan,
)


class FunctionSummary(BaseModel):
    """Summary fields from docs.v_function_summary."""

    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str

    loc: int | None = None
    logical_loc: int | None = None
    cyclomatic_complexity: int | None = None
    complexity_bucket: str | None = None

    param_count: int | None = None
    positional_params: int | None = None
    keyword_only_params: int | None = None
    has_varargs: bool | None = None
    has_varkw: bool | None = None
    is_async: bool | None = None
    is_generator: bool | None = None
    return_count: int | None = None
    yield_count: int | None = None
    raise_count: int | None = None

    typedness_bucket: str | None = None
    typedness_source: str | None = None

    hotspot_score: float | None = None
    file_typed_ratio: float | None = None
    static_error_count: int | None = None
    has_static_errors: bool | None = None

    executable_lines: int | None = None
    covered_lines: int | None = None
    coverage_ratio: float | None = None
    tested: bool | None = None
    test_count: int | None = None
    failing_test_count: int | None = None
    last_test_status: str | None = None

    risk_score: float | None = None
    risk_level: str | None = None

    tags: list[str] | None = None
    owners: list[str] | None = None

    created_at: str | None = None


class CallGraphEdge(BaseModel):
    """Call graph edge enriched with caller/callee metadata and risk."""

    caller_goid_h128: int | None = None
    caller_urn: str | None = None
    caller_rel_path: str | None = None
    caller_qualname: str | None = None
    caller_risk_level: str | None = None
    caller_risk_score: float | None = None

    callee_goid_h128: int | None = None
    callee_urn: str | None = None
    callee_rel_path: str | None = None
    callee_qualname: str | None = None
    callee_risk_level: str | None = None
    callee_risk_score: float | None = None

    callsite_path: str | None = None
    callsite_line: int | None = None
    callsite_col: int | None = None
    language: str | None = None
    kind: str | None = None
    resolved_via: str | None = None
    confidence: float | None = None


class TestToFunctionEdge(BaseModel):
    """Test-to-function coverage edge."""

    test_id: str
    test_goid_h128: int | None = None
    test_urn: str | None = None
    test_repo: str | None = None
    test_commit: str | None = None
    test_rel_path: str | None = None
    test_qualname: str | None = None
    test_kind: str | None = None
    test_status: str | None = None
    duration_ms: float | None = None
    markers: list[str] | None = None
    parametrized: bool | None = None
    flaky: bool | None = None

    function_goid_h128: int | None = None
    function_urn: str | None = None
    function_rel_path: str | None = None
    function_qualname: str | None = None
    function_language: str | None = None
    function_kind: str | None = None
    function_risk_score: float | None = None
    function_risk_level: str | None = None

    covered_lines: int | None = None
    executable_lines: int | None = None
    coverage_ratio: float | None = None
    edge_last_status: str | None = None
    edge_created_at: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_connection() -> duckdb.DuckDBPyConnection:
    """
    Retrieve the shared DuckDB connection from the FastAPI app state.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Active DuckDB connection.

    Raises
    ------
    RuntimeError
        If the connection has not been initialized.
    """
    connection = getattr(app.state, "con", None)
    if connection is None:
        msg = "DuckDB connection has not been initialized"
        raise RuntimeError(msg)
    return connection


def _decode_json_field(value: object) -> object | None:
    """
    Decode JSON stored as a string into native Python structures.

    Parameters
    ----------
    value:
        Value that may be a JSON-encoded string.

    Returns
    -------
    object | None
        Parsed JSON if decodable, otherwise the original value.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        with suppress(json.JSONDecodeError):
            return json.loads(value)
    return value


def _to_function_summary(row: dict[str, Any]) -> FunctionSummary:
    """
    Convert a DuckDB row dictionary into a FunctionSummary model.

    Parameters
    ----------
    row:
        Dictionary returned from DuckDB.

    Returns
    -------
    FunctionSummary
        Pydantic model with normalized types.
    """
    # DuckDB may return DECIMAL as Python Decimal; Pydantic will happily coerce,
    # but you can normalize explicitly if you prefer.
    for key in ("function_goid_h128",):
        raw_value = row.get(key)
        if raw_value is not None:
            with suppress(TypeError, ValueError):
                row[key] = int(raw_value)

    for key in ("tags", "owners"):
        row[key] = _decode_json_field(row.get(key))

    return FunctionSummary(**row)


def _resolve_goid_from_core(
    con: duckdb.DuckDBPyConnection,
    urn: str | None,
    goid_h128: int | None,
) -> int | None:
    """
    Resolve a GOID from core.goids using either URN or numeric hash.

    Parameters
    ----------
    con:
        Active DuckDB connection.
    urn:
        GOID URN to resolve.
    goid_h128:
        Numeric GOID hash to resolve.

    Returns
    -------
    int | None
        Resolved numeric GOID if found, otherwise None.
    """
    if goid_h128 is not None:
        return goid_h128
    if urn is None:
        return None
    row = con.execute("SELECT goid_h128 FROM core.goids WHERE urn = ? LIMIT 1;", [urn]).fetchone()
    if row is None:
        return None
    return int(row[0])


def _resolve_goid_from_docs(
    con: duckdb.DuckDBPyConnection,
    urn: str | None,
    goid_h128: int | None,
) -> int | None:
    """
    Resolve a GOID from docs.v_function_summary using URN or numeric hash.

    Parameters
    ----------
    con:
        Active DuckDB connection.
    urn:
        GOID URN to resolve.
    goid_h128:
        Numeric GOID hash to resolve.

    Returns
    -------
    int | None
        Resolved numeric GOID if found, otherwise None.
    """
    if goid_h128 is not None:
        return goid_h128
    if urn is None:
        return None
    row = con.execute(
        "SELECT function_goid_h128 FROM docs.v_function_summary WHERE urn = ? LIMIT 1;", [urn]
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/function/summary/by-urn",
    response_model=FunctionSummary,
    summary="Get function summary by GOID URN",
)
def function_summary_by_urn(urn: str = Query(..., description="GOID URN")) -> FunctionSummary:
    """
    Fetch a function summary by GOID URN.

    Parameters
    ----------
    urn:
        GOID URN identifying the function.

    Returns
    -------
    FunctionSummary
        Summary row for the requested function.

    Raises
    ------
    HTTPException
        If the function is not found.
    """
    con = _get_connection()
    df = con.execute(
        """
        SELECT *
        FROM docs.v_function_summary
        WHERE urn = ?
        LIMIT 1;
        """,
        [urn],
    ).fetch_df()

    if df.empty:
        raise HTTPException(status_code=404, detail="Function not found")

    row = df.iloc[0].to_dict()
    return _to_function_summary(row)


@app.get(
    "/function/summary/by-goid",
    response_model=FunctionSummary,
    summary="Get function summary by numeric GOID",
)
def function_summary_by_goid(
    goid_h128: int = Query(..., description="Numeric GOID (128-bit hash)"),
) -> FunctionSummary:
    """
    Fetch a function summary by numeric GOID.

    Parameters
    ----------
    goid_h128:
        Numeric GOID hash.

    Returns
    -------
    FunctionSummary
        Summary row for the requested function.

    Raises
    ------
    HTTPException
        If the function is not found.
    """
    con = _get_connection()
    df = con.execute(
        """
        SELECT *
        FROM docs.v_function_summary
        WHERE function_goid_h128 = ?
        LIMIT 1;
        """,
        [goid_h128],
    ).fetch_df()

    if df.empty:
        raise HTTPException(status_code=404, detail="Function not found")

    row = df.iloc[0].to_dict()
    return _to_function_summary(row)


@app.get(
    "/function/callgraph",
    response_model=list[CallGraphEdge],
    summary="Get call graph neighborhood for a function",
)
def function_callgraph(
    urn: str | None = Query(
        None,
        description="GOID URN of the root function (required if goid_h128 not provided)",
    ),
    goid_h128: int | None = Query(
        None,
        description="Numeric GOID (if URN not provided)",
    ),
    direction: Literal["both", "incoming", "outgoing"] = Query(
        "both",
        regex="^(both|incoming|outgoing)$",
        description="Edge direction relative to the function",
    ),
) -> list[CallGraphEdge]:
    """
    Return call graph edges around a function.

    Parameters
    ----------
    urn:
        Function GOID URN (required if goid_h128 not provided).
    goid_h128:
        Numeric GOID hash (preferred if available).
    direction:
        Whether to fetch incoming, outgoing, or both edge directions.

    Returns
    -------
    list[CallGraphEdge]
        Edges annotated with caller/callee metadata.

    Raises
    ------
    HTTPException
        If the GOID cannot be resolved or the query is invalid.
    """
    con = _get_connection()
    resolved_goid = _resolve_goid_from_core(con, urn, goid_h128)
    if resolved_goid is None:
        detail = (
            "Either urn or goid_h128 must be specified"
            if urn is None
            else "Function not found for urn"
        )
        status = 400 if urn is None else 404
        raise HTTPException(status_code=status, detail=detail)

    if direction == "incoming":
        sql = """
            SELECT
              caller_goid_h128,
              caller_urn,
              caller_rel_path,
              caller_qualname,
              caller_risk_level,
              caller_risk_score,
              callee_goid_h128,
              callee_urn,
              callee_rel_path,
              callee_qualname,
              callee_risk_level,
              callee_risk_score,
              callsite_path,
              callsite_line,
              callsite_col,
              language,
              kind,
              resolved_via,
              confidence
            FROM docs.v_call_graph_enriched
            WHERE callee_goid_h128 = ?;
        """
        params = [resolved_goid]
    elif direction == "outgoing":
        sql = """
            SELECT
              caller_goid_h128,
              caller_urn,
              caller_rel_path,
              caller_qualname,
              caller_risk_level,
              caller_risk_score,
              callee_goid_h128,
              callee_urn,
              callee_rel_path,
              callee_qualname,
              callee_risk_level,
              callee_risk_score,
              callsite_path,
              callsite_line,
              callsite_col,
              language,
              kind,
              resolved_via,
              confidence
            FROM docs.v_call_graph_enriched
            WHERE caller_goid_h128 = ?;
        """
        params = [resolved_goid]
    else:
        sql = """
            SELECT
              caller_goid_h128,
              caller_urn,
              caller_rel_path,
              caller_qualname,
              caller_risk_level,
              caller_risk_score,
              callee_goid_h128,
              callee_urn,
              callee_rel_path,
              callee_qualname,
              callee_risk_level,
              callee_risk_score,
              callsite_path,
              callsite_line,
              callsite_col,
              language,
              kind,
              resolved_via,
              confidence
            FROM docs.v_call_graph_enriched
            WHERE caller_goid_h128 = ? OR callee_goid_h128 = ?;
        """
        params = [resolved_goid, resolved_goid]

    df = con.execute(sql, params).fetch_df()

    records = df.to_dict("records")
    # Normalize number types if needed
    for rec in records:
        for key in ("caller_goid_h128", "callee_goid_h128"):
            if rec.get(key) is not None:
                with suppress(TypeError, ValueError):
                    rec[key] = int(rec[key])

    return [CallGraphEdge(**rec) for rec in records]


@app.get(
    "/tests/for-function",
    response_model=list[TestToFunctionEdge],
    summary="List tests that exercise a function",
)
def get_tests_for_function(
    urn: str | None = Query(
        None,
        description="GOID URN of the function (required if goid_h128 not provided)",
    ),
    goid_h128: int | None = Query(
        None,
        description="Numeric GOID (if URN not provided)",
    ),
) -> list[TestToFunctionEdge]:
    """
    List tests that exercise a given function.

    Parameters
    ----------
    urn:
        GOID URN of the target function.
    goid_h128:
        Numeric GOID hash of the target function.

    Returns
    -------
    list[TestToFunctionEdge]
        Coverage edges between tests and the target function.

    Raises
    ------
    HTTPException
        If the GOID cannot be resolved or the query is invalid.
    """
    con = _get_connection()
    resolved_goid = _resolve_goid_from_docs(con, urn, goid_h128)
    if resolved_goid is None:
        detail = (
            "Either urn or goid_h128 must be specified"
            if urn is None
            else "Function not found for urn"
        )
        status = 400 if urn is None else 404
        raise HTTPException(status_code=status, detail=detail)

    df = con.execute(
        """
        SELECT
            test_id,
            test_goid_h128,
            test_urn,
            test_repo,
            test_commit,
            test_rel_path,
            test_qualname,
            test_kind,
            test_status,
            duration_ms,
            markers,
            parametrized,
            flaky,
            function_goid_h128,
            function_urn,
            function_rel_path,
            function_qualname,
            function_language,
            function_kind,
            function_risk_score,
            function_risk_level,
            covered_lines,
            executable_lines,
            coverage_ratio,
            edge_last_status,
            edge_created_at
        FROM docs.v_test_to_function
        WHERE function_goid_h128 = ?;
        """,
        [resolved_goid],
    ).fetch_df()

    records = df.to_dict("records")

    for rec in records:
        for key in ("test_goid_h128", "function_goid_h128"):
            if rec.get(key) is not None:
                with suppress(TypeError, ValueError):
                    rec[key] = int(rec[key])
        for key in ("markers",):
            rec[key] = _decode_json_field(rec.get(key))

    return [TestToFunctionEdge(**rec) for rec in records]
