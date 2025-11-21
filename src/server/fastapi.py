# src/codeintel_core/server/api.py

from __future__ import annotations

import json
import os
from typing import List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from codeintel_core.config.schemas import create_all_schemas
from codeintel_core.config.views import create_all_views


DB_PATH = os.getenv("CODEINTEL_DUCKDB_PATH", "build/db/codeintel.duckdb")

app = FastAPI(
    title="CodeIntel Metadata API",
    description="Thin API over DuckDB views for AI agents and tools.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FunctionSummary(BaseModel):
    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str

    loc: Optional[int] = None
    logical_loc: Optional[int] = None
    cyclomatic_complexity: Optional[int] = None
    complexity_bucket: Optional[str] = None

    param_count: Optional[int] = None
    positional_params: Optional[int] = None
    keyword_only_params: Optional[int] = None
    has_varargs: Optional[bool] = None
    has_varkw: Optional[bool] = None
    is_async: Optional[bool] = None
    is_generator: Optional[bool] = None
    return_count: Optional[int] = None
    yield_count: Optional[int] = None
    raise_count: Optional[int] = None

    typedness_bucket: Optional[str] = None
    typedness_source: Optional[str] = None

    hotspot_score: Optional[float] = None
    file_typed_ratio: Optional[float] = None
    static_error_count: Optional[int] = None
    has_static_errors: Optional[bool] = None

    executable_lines: Optional[int] = None
    covered_lines: Optional[int] = None
    coverage_ratio: Optional[float] = None
    tested: Optional[bool] = None
    test_count: Optional[int] = None
    failing_test_count: Optional[int] = None
    last_test_status: Optional[str] = None

    risk_score: Optional[float] = None
    risk_level: Optional[str] = None

    tags: Optional[list[str]] = None
    owners: Optional[list[str]] = None

    created_at: Optional[str] = None


class CallGraphEdge(BaseModel):
    caller_goid_h128: Optional[int] = None
    caller_urn: Optional[str] = None
    caller_rel_path: Optional[str] = None
    caller_qualname: Optional[str] = None
    caller_risk_level: Optional[str] = None
    caller_risk_score: Optional[float] = None

    callee_goid_h128: Optional[int] = None
    callee_urn: Optional[str] = None
    callee_rel_path: Optional[str] = None
    callee_qualname: Optional[str] = None
    callee_risk_level: Optional[str] = None
    callee_risk_score: Optional[float] = None

    callsite_path: Optional[str] = None
    callsite_line: Optional[int] = None
    callsite_col: Optional[int] = None
    language: Optional[str] = None
    kind: Optional[str] = None
    resolved_via: Optional[str] = None
    confidence: Optional[float] = None


class TestToFunctionEdge(BaseModel):
    test_id: str
    test_goid_h128: Optional[int] = None
    test_urn: Optional[str] = None
    test_repo: Optional[str] = None
    test_commit: Optional[str] = None
    test_rel_path: Optional[str] = None
    test_qualname: Optional[str] = None
    test_kind: Optional[str] = None
    test_status: Optional[str] = None
    duration_ms: Optional[float] = None
    markers: Optional[list[str]] = None
    parametrized: Optional[bool] = None
    flaky: Optional[bool] = None

    function_goid_h128: Optional[int] = None
    function_urn: Optional[str] = None
    function_rel_path: Optional[str] = None
    function_qualname: Optional[str] = None
    function_language: Optional[str] = None
    function_kind: Optional[str] = None
    function_risk_score: Optional[float] = None
    function_risk_level: Optional[str] = None

    covered_lines: Optional[int] = None
    executable_lines: Optional[int] = None
    coverage_ratio: Optional[float] = None
    edge_last_status: Optional[str] = None
    edge_created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    con = duckdb.connect(DB_PATH, read_only=False)
    # Ensure schemas and views exist; if they already do, these are no-ops.
    create_all_schemas(con)
    create_all_views(con)
    app.state.con = con


@app.on_event("shutdown")
def shutdown() -> None:
    con = getattr(app.state, "con", None)
    if con is not None:
        con.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_json_field(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _to_function_summary(row: dict) -> FunctionSummary:
    # DuckDB may return DECIMAL as Python Decimal; Pydantic will happily coerce,
    # but you can normalize explicitly if you prefer.
    for key in ("function_goid_h128",):
        if key in row and row[key] is not None:
            try:
                row[key] = int(row[key])
            except Exception:
                pass

    for key in ("tags", "owners"):
        row[key] = _decode_json_field(row.get(key))

    return FunctionSummary(**row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/function/summary/by-urn",
    response_model=FunctionSummary,
    summary="Get function summary by GOID URN",
)
def function_summary_by_urn(urn: str = Query(..., description="GOID URN")):
    con = app.state.con
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
    goid_h128: int = Query(..., description="Numeric GOID (128-bit hash)")
):
    con = app.state.con
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
    response_model=List[CallGraphEdge],
    summary="Get call graph neighborhood for a function",
)
def function_callgraph(
    urn: Optional[str] = Query(
        None,
        description="GOID URN of the root function (required if goid_h128 not provided)",
    ),
    goid_h128: Optional[int] = Query(
        None,
        description="Numeric GOID (if URN not provided)",
    ),
    direction: str = Query(
        "both",
        regex="^(both|incoming|outgoing)$",
        description="Edge direction relative to the function",
    ),
):
    con = app.state.con

    # Resolve GOID if only URN is provided
    if goid_h128 is None:
        if urn is None:
            raise HTTPException(
                status_code=400, detail="Either urn or goid_h128 must be specified"
            )
        row = con.execute(
            "SELECT goid_h128 FROM core.goids WHERE urn = ? LIMIT 1;", [urn]
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Function not found for urn")
        goid_h128 = int(row[0])

    if direction == "incoming":
        where_clause = "callee_goid_h128 = ?"
    elif direction == "outgoing":
        where_clause = "caller_goid_h128 = ?"
    else:
        where_clause = "caller_goid_h128 = ? OR callee_goid_h128 = ?"

    params = [goid_h128] if direction != "both" else [goid_h128, goid_h128]

    df = con.execute(
        f"""
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
        WHERE {where_clause};
        """,
        params,
    ).fetch_df()

    records = df.to_dict("records")
    # Normalize number types if needed
    for rec in records:
        for key in ("caller_goid_h128", "callee_goid_h128"):
            if rec.get(key) is not None:
                try:
                    rec[key] = int(rec[key])
                except Exception:
                    pass

    return [CallGraphEdge(**rec) for rec in records]


@app.get(
    "/tests/for-function",
    response_model=List[TestToFunctionEdge],
    summary="List tests that exercise a function",
)
def tests_for_function(
    urn: Optional[str] = Query(
        None,
        description="GOID URN of the function (required if goid_h128 not provided)",
    ),
    goid_h128: Optional[int] = Query(
        None,
        description="Numeric GOID (if URN not provided)",
    ),
):
    con = app.state.con

    if goid_h128 is None:
        if urn is None:
            raise HTTPException(
                status_code=400, detail="Either urn or goid_h128 must be specified"
            )
        row = con.execute(
            "SELECT function_goid_h128 FROM docs.v_function_summary WHERE urn = ? LIMIT 1;",
            [urn],
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Function not found for urn")
        goid_h128 = int(row[0])

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
        [goid_h128],
    ).fetch_df()

    records = df.to_dict("records")

    for rec in records:
        for key in ("test_goid_h128", "function_goid_h128"):
            if rec.get(key) is not None:
                try:
                    rec[key] = int(rec[key])
                except Exception:
                    pass
        for key in ("markers",):
            rec[key] = _decode_json_field(rec.get(key))

    return [TestToFunctionEdge(**rec) for rec in records]
