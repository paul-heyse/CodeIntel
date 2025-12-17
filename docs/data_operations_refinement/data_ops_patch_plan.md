# Data Operations Patch Plan

**Goal:** Deliver a best-in-class, production-safe data operations backbone across **DuckDB**, **Ibis**, and **SQLGlot**, spanning **build** (materialization + view management) and **serving** (semantic queries + export + MCP).

This is written as a *PR checklist* with:

- exact function signatures
- acceptance criteria per PR
- a test matrix mapping each risk to concrete tests

It is designed to fit the current abstractions without large refactors:

- `StorageGateway` (composition root)
- `DuckDBSession` (connection lifecycle + bootstrap)
- `IbisGateway` (expression compilation + write fast lanes)
- `Warehouse` (table/view read/write façade + profiling)
- serving connection pools (`storage.gateway.pool` / `serving.db.pool`)

---

## Guiding principles

1. **Constant-memory export paths**  
   No `fetchall()`, no `list(...)` for NDJSON/export flows, no in-memory Parquet/Arrow buffers for large exports.

2. **Single SQL boundary**  
   - User input: *never* interpolated into SQL strings.
   - Dynamic SQL: built via Ibis + SQLGlot or via parameterized DuckDB `execute(sql, params)`.

3. **Deterministic bootstrap**  
   Extensions, secrets, and init SQL must be configured once, consistently, and with explicit “INSTALL allowed?” policy.

4. **Schema contracts are canonical**  
   `TableSchema` remains the source of truth; DDL and validation must be generated programmatically (no hand-maintained type maps).

5. **Upgrade safety**  
   SQLGlot/Ibis bumps are “compiler upgrades”: gated by golden SQL + execution checks.

---

## Minimal PR sequence

> The PRs below are ordered to reduce operational risk first (exports + write paths), then standardize compilation + typing, then add observability + upgrade gates.

### PR1 — True streaming NDJSON exports (HTTP + kernel)
### PR2 — Arrow/Parquet exports without row buffering (spool-to-disk)
### PR3 — MCP exports without buffering (ResourceStore streaming writer)
### PR4 — Remove pandas fallback: UPSERT-from-expression + bulk DataFrame fast lane
### PR5 — Centralize DuckDB bootstrap: LOAD vs INSTALL policy + init SQL + health checks
### PR6 — Schema → Ibis → SQLGlot DDL bridge (remove manual type mapping)
### PR7 — SQLGlot AST hook + canonical fingerprint utilities
### PR8 — Contract-driven filter typing + large IN-list staging + (optional) parameter binding
### PR9 — Serving telemetry + fingerprints in responses + (optional) query cache
### PR10 — View semantic diff artifacts + compiler upgrade gates

---

## PR checklist details

Each PR section includes:

- **Files**
- **Function signatures**
- **Acceptance criteria**
- **Tests to add**

---

# PR1 — True streaming NDJSON exports (HTTP + kernel)

## Why
Current export paths buffer results in memory in two places:

- `SemanticQueryKernel.export_rows()` calls `fetchall()`
- HTTP `/export` routes call `list(kernel.export_rows(...))`

This can OOM for large exports and delays “first byte”.

## Files

- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/http/routes/v1/export.py`
- `src/codeintel/serving/http/streaming.py`
- `src/codeintel/serving/settings.py`

## Function signatures

### 1) Serving settings: chunk size knobs

```python
# src/codeintel/serving/settings.py

@dataclass(frozen=True)
class ServingSettings:
    ...
    export_chunk_size: int = 10_000  # batches for Arrow/NDJSON conversion
    export_spool_dir: Path | None = None  # where to place temp export files (PR2)
```

### 2) Kernel: stream rows using Arrow batches (no fetchall)

```python
# src/codeintel/serving/semantic/kernel.py

from collections.abc import Iterator

class SemanticQueryKernel:
    ...

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        """Yield rows for streaming export (constant memory).

        Implementation must:
        - keep the DB connection open for the duration of streaming
        - avoid fetchall() / list() materialization
        - guarantee cleanup on generator close (client disconnect)
        """
        ...
```

**Implementation note (behavioral requirement):**
- Build Ibis expression as today.
- Convert to Arrow record batches: `expr.to_pyarrow_batches(chunk_size=self.settings.export_chunk_size)`
- Iterate batches and `yield` row dicts per row (or per batch-to-pylist).
- Wrap generator body in `try/finally` so disconnect / early close cleans up.

### 3) HTTP export route: remove `list(...)` buffering for NDJSON

```python
# src/codeintel/serving/http/routes/v1/export.py

@router.post("/semantic/{view_id}")
async def export_view(...):
    ...
    if payload.format == "ndjson":
        # return StreamingResponse directly from kernel iterator
        rows = kernel.export_rows(payload)
        return ndjson_response(rows, filename=f"{view_id}.ndjson")
```

### 4) NDJSON streamer: accept iterables/generators unchanged

Already fine, but keep as “no pre-materialization”.

## Acceptance criteria

- NDJSON export begins streaming immediately (first chunk returned without computing all rows).
- No call sites remain that do `list(kernel.export_rows(...))` in HTTP NDJSON path.
- `SemanticQueryKernel.export_rows()` contains **no** `fetchall()` and does not accumulate all rows.
- A client disconnect closes the generator and releases the DuckDB connection (no pool starvation).

## Tests to add

- `tests/serving/test_export_streaming_ndjson.py`

Suggested tests:

1) **No materialization on route call**
   - Stub `kernel.export_rows` with a generator that raises if iterated during route handler.
   - Ensure calling the route returns a `StreamingResponse` without raising.

2) **Generator cleanup on close**
   - Stub a context-managed connection that flips a flag on close.
   - Create iterator from `export_rows`, advance once, then close generator; assert cleanup flag set.

---

# PR2 — Arrow/Parquet exports without row buffering (spool-to-disk)

## Why
Current code:

- converts rows → `pa.Table.from_pylist(rows)` (requires full row list)
- writes to an in-memory `BytesIO` buffer (stores full output in memory)

We want constant memory, even if it’s not “time-to-first-byte” streaming.

## Files

- `src/codeintel/serving/http/routes/v1/export.py`
- `src/codeintel/serving/semantic/kernel.py` (optional helper)
- `src/codeintel/serving/settings.py`

## Function signatures

### 1) Kernel: provide Arrow batch reader for exports (optional, but recommended)

```python
# src/codeintel/serving/semantic/kernel.py

class SemanticQueryKernel:
    ...

    def export_arrow_batches(self, request: SemanticExportRequest) -> "pa.RecordBatchReader":
        """Return a pyarrow RecordBatchReader for the export query.

        - Requires pyarrow at runtime
        - Keeps DB connection open while batches are consumed
        """
        ...
```

If you want to avoid adding pyarrow types into the kernel module’s import surface, annotate as `object` and import pyarrow lazily inside the function.

### 2) Export route: spool to temp file, then stream file handle

```python
# src/codeintel/serving/http/routes/v1/export.py

async def _parquet_response(...) -> StreamingResponse:
    """Write parquet to a temp file using ParquetWriter and return StreamingResponse."""
    ...

async def _arrow_response(...) -> StreamingResponse:
    """Write Arrow IPC file to a temp file using pa.ipc.new_file and return StreamingResponse."""
    ...
```

**Implementation notes:**
- Use `settings.export_spool_dir` or `tempfile.TemporaryDirectory()` for file location.
- Write in `run_in_threadpool(...)` to avoid blocking the event loop.
- Ensure temp file removal via `BackgroundTask` or generator finalizer.

## Acceptance criteria

- Parquet/Arrow export does not build `rows: list[...]` in memory.
- Parquet/Arrow export does not create `pa.Table.from_pylist(rows)` from Python dicts.
- Output is written to disk incrementally using Arrow record batches.
- Temp files are removed after response completes or client disconnects.

## Tests to add

- `tests/serving/test_export_spooling.py`

Suggested tests:
- Ensure `_parquet_response` and `_arrow_response` never call `kernel.export_rows()` or `list(...)`.
- Ensure temp file is created and then removed (use temp spool dir fixture).
- Ensure response headers include correct content types + filenames.

---

# PR3 — MCP exports without buffering (ResourceStore streaming writer)

## Why
Current MCP export tool still buffers the full export in memory:

```python
rows = await limiter.run(lambda: list(kernel.export_rows(request)))
store.put_with_metadata(rows, ...)
```

We want MCP export to write to disk as rows stream, while preserving metadata.

## Files

- `src/codeintel/serving/mcp/app.py`
- `src/codeintel/serving/mcp/resource_store.py`
- `src/codeintel/serving/mcp/resources.py` (if needed)
- `src/codeintel/serving/mcp/response_models.py` (populate query_hash/schema_hash later)

## Function signatures

### 1) ResourceStore: streaming NDJSON writer with metadata

```python
# src/codeintel/serving/mcp/resource_store.py

from collections.abc import Iterable, Iterator

class ResourceStore:
    ...

    def put_ndjson_stream(
        self,
        rows: Iterable[dict[str, object]],
        *,
        view_id: str,
        columns: tuple[str, ...],
        column_types: dict[str, str] | None = None,
        compiled_sql: str | None = None,
        snapshot: dict[str, str] | None = None,
    ) -> tuple[str, StoredArtifact, StoredMetadata]:
        """Write NDJSON incrementally and persist a .meta.json sidecar.

        Must not materialize `rows` as a list.
        Must return accurate row_count and size_bytes.
        """
        ...
```

(Optionally, keep `put_with_metadata(rows: list[...])` as a convenience wrapper that calls the streaming writer with `iter(rows)`.)

### 2) MCP tool: pass iterator straight through

```python
# src/codeintel/serving/mcp/app.py

rows_iter = kernel.export_rows(request)
token, artifact, meta = store.put_ndjson_stream(
    rows_iter,
    view_id=view_id,
    columns=columns,
    column_types=column_types,
    compiled_sql=compiled_sql,
    snapshot=snapshot_dict,
)
```

**Important:** column discovery needs a non-buffering approach:
- Either take `columns` from schema inventory (preferred)
- Or fetch a tiny preview first (e.g., first batch) then chain it back into the iterator.

Recommended approach: derive `columns` + `column_types` from `SchemaInventory` for the view’s `table_key`.

## Acceptance criteria

- MCP export tool does not allocate `rows: list[...]` for NDJSON.
- ResourceStore writes line-by-line and produces correct row_count in metadata.
- Export can handle “large rows” without OOM.
- Disconnected/aborted export does not leave partially-written temp files without metadata; partial export either:
  - is deleted, or
  - is marked as incomplete in metadata with safe behavior.

## Tests to add

- `tests/serving/test_mcp_export_streaming_store.py`

Suggested tests:
- Use a generator that yields many rows; ensure memory does not balloon by verifying no list conversion (unit-style: generator raises if iterated early).
- Ensure meta sidecar exists and row_count matches.

---

# PR4 — Remove pandas fallback: UPSERT-from-expression + bulk DataFrame fast lane

## Why
`IbisGateway._write_ibis_expression(..., on_conflict=...)` currently falls back to:

- `expr.to_pandas()` → huge memory risk
- Python tuple inserts for DataFrame writes → slow

We want “Arrow-first” / DuckDB-native writes.

## Files

- `src/codeintel/storage/ibis_adapter.py`
- `src/codeintel/storage/duckdb_policy_backend.py`

## Function signatures

### 1) Policy backend: add UPSERT-from-SELECT

```python
# src/codeintel/storage/duckdb_policy_backend.py

from collections.abc import Sequence

class DuckDBPolicyBackend:
    ...

    def upsert_select(
        self,
        *,
        schema: str,
        table: str,
        columns: list[str],
        select_sql: str,
        conflict_columns: Sequence[str],
        update_columns: Sequence[str] | None = None,
    ) -> int:
        """INSERT ... SELECT ... ON CONFLICT DO UPDATE.

        - Must be safe against SQL injection (select_sql is compiler output)
        - Must preserve current upsert semantics used by values-based upsert
        """
        ...
```

### 2) IbisGateway: implement upsert-from-expression without pandas

```python
# src/codeintel/storage/ibis_adapter.py

class IbisGateway:
    ...

    def _write_ibis_expression(
        self,
        write_ctx: _WriteContext,
        expr: "it.Table",
        columns: list[str],
        *,
        on_conflict: OnConflict | None = None,
    ) -> WriteResult:
        """Write an Ibis expression via INSERT SELECT / UPSERT SELECT.

        Must not call expr.to_pandas().
        """
        ...
```

### 3) IbisGateway: bulk DataFrame writes via register + INSERT SELECT

```python
# src/codeintel/storage/ibis_adapter.py

class IbisGateway:
    ...

    def _write_dataframe(
        self,
        write_ctx: _WriteContext,
        df: "pd.DataFrame",
        columns: list[str],
        *,
        on_conflict: OnConflict | None = None,
    ) -> WriteResult:
        """Write a Pandas DataFrame.

        Strategy:
        - small df: keep current tuple path
        - large df: con.register(temp_name, df) + INSERT SELECT (and optional UPSERT)
        Must unregister temp_name.
        """
        ...
```

**Heuristic knob (suggested):**
- `CODEINTEL_DUCKDB_BULK_WRITE_THRESHOLD_ROWS` (default e.g. 10_000)

## Acceptance criteria

- UPSERT-from-expression never materializes the expression to pandas.
- Bulk DataFrame inserts do not loop Python tuples for large data.
- Temp objects created for staging are cleaned up (unregister/drop) even on exceptions.

## Tests to add

- `tests/storage/test_upsert_from_expression_no_pandas.py`
- `tests/storage/test_bulk_dataframe_write_fast_lane.py`

Suggested tests:
- Monkeypatch `it.Table.to_pandas` to raise; verify upsert still succeeds.
- Verify that `con.unregister(temp_name)` (or equivalent) is called even if insert fails.

---

# PR5 — Centralize DuckDB bootstrap: LOAD vs INSTALL policy + init SQL + health checks

## Why
Extension loading is currently:

- environment-driven in `storage/gateway/connection.py` (always INSTALL+LOAD)
- ad-hoc in `storage/serving/search_index.py` (tries LOAD then INSTALL+LOAD)

Serving should **not** run `INSTALL` implicitly. Build can, but only explicitly.

## Files

- `src/codeintel/storage/gateway/connection.py`
- `src/codeintel/storage/backend/duckdb_session.py`
- `src/codeintel/storage/serving/search_index.py`
- (optional) `src/codeintel/storage/gateway/config.py` (if we extend config)

## Function signatures

### 1) Add explicit bootstrap policy (env + config)

```python
# src/codeintel/storage/gateway/connection.py

from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True, slots=True)
class DuckDBBootstrapPolicy:
    extensions: tuple[str, ...] = ()
    allow_install: bool = False
    init_sql: tuple[str, ...] = ()  # optional additional init statements

def load_bootstrap_policy_from_env() -> DuckDBBootstrapPolicy:
    """Parse CODEINTEL_DUCKDB_EXTENSIONS + CODEINTEL_DUCKDB_ALLOW_INSTALL + init SQL."""
    ...
```

### 2) Apply policy during connect

```python
# src/codeintel/storage/gateway/connection.py

def connect(
    config: StorageConfig,
    *,
    duckdb_config: DuckDBConnectConfig | None = None,
    bootstrap: DuckDBBootstrapPolicy | None = None,
) -> DuckDBConnection:
    ...
```

Behavior:
- If `config.read_only` is True → force `allow_install=False` even if env says otherwise.
- Use `LOAD ext` always; only `INSTALL ext` if allow_install.

### 3) DuckDBSession: single entry point for bootstrap

```python
# src/codeintel/storage/backend/duckdb_session.py

@dataclass(frozen=True, slots=True)
class DuckDBSession:
    ...
    bootstrap: DuckDBBootstrapPolicy | None = None

    def open(self) -> DuckDBConnection: ...
    def open_reader(self) -> DuckDBConnection: ...
```

### 4) Search index: remove implicit INSTALL fallback by default

```python
# src/codeintel/storage/serving/search_index.py

def ensure_fts_index(con: DuckDBConnection, *, table_key: str = "docs.search_documents") -> str:
    """Ensure FTS exists.

    Must not run INSTALL in serving contexts. If extension missing, raise a clear error
    telling operators how to enable it (bootstrap allow_install or pre-install ext).
    """
    ...
```

## Acceptance criteria

- Serving read-only paths never call `INSTALL` unless explicitly configured *and* not read-only.
- Extension enablement is consistent across build + serving.
- Clear error when an extension is required but not available in prod.
- Optional: a startup health check validates required extensions and emits actionable messages.

## Tests to add

- `tests/storage/test_duckdb_bootstrap_policy.py`

Suggested tests:
- With `read_only=True`, ensure attempted install is blocked even if env enables it.
- Verify parsing of env extension list and init SQL.

---

# PR6 — Schema → Ibis → SQLGlot DDL bridge (remove manual type mapping)

## Why
`DuckDBPolicyBackend` currently hand-maintains `_column_type_to_sqlglot(...)`.

We want to generate ColumnDefs via Ibis Schema round-trip (`Schema.to_sqlglot_column_defs(...)`) so that type parsing and dialect output stays correct as SQLGlot evolves.

## Files

- `src/codeintel/storage/schema/ibis_roundtrip.py` (new)
- `src/codeintel/storage/duckdb_policy_backend.py`

## Function signatures

```python
# src/codeintel/storage/schema/ibis_roundtrip.py

import ibis
from sqlglot import exp

from codeintel.core.schemas.primitives import TableSchema

DUCKDB_DIALECT = "duckdb"

def table_schema_to_ibis(schema: TableSchema) -> ibis.Schema:
    """Convert CodeIntel TableSchema to an Ibis Schema."""
    ...

def ibis_schema_to_sqglot_column_defs(schema: ibis.Schema) -> list[exp.ColumnDef]:
    """Convert Ibis Schema to SQLGlot ColumnDefs."""
    ...

def table_schema_to_sqglot_column_defs(schema: TableSchema) -> list[exp.ColumnDef]:
    """Convenience wrapper: TableSchema -> Ibis Schema -> ColumnDefs."""
    ...
```

Then update backend:

```python
# src/codeintel/storage/duckdb_policy_backend.py

class DuckDBPolicyBackend:
    ...

    def create_table_from_schema(self, schema: TableSchema, *, drop_existing: bool = False) -> None:
        """Generate CREATE TABLE from TableSchema using Ibis->SQLGlot column defs."""
        ...
```

## Acceptance criteria

- `_column_type_to_sqlglot` and manual mapping are removed (or unused).
- DDL emitted for existing contracts is stable and parseable by SQLGlot.
- Existing schema apply flows still work.

## Tests to add

- `tests/storage/test_schema_ddl_roundtrip.py`

Suggested tests:
- For a representative subset of schemas, generate SQL and parse it with SQLGlot; assert no errors.
- Ensure NOT NULL behavior matches `Column.nullable`.

---

# PR7 — SQLGlot AST hook + canonical fingerprint utilities

## Why
We need:

- query fingerprints for caching + provenance
- semantic diffs that don’t depend on string formatting
- consistent canonicalization for compiled SQL

Ibis supports `con.compiler.to_sqlglot(expr)`.

## Files

- `src/codeintel/storage/ibis_adapter.py`
- `src/codeintel/storage/sqlglot_utils.py` (new, recommended)
- `src/codeintel/serving/semantic/kernel.py` (optional for query hash)
- `src/codeintel/serving/mcp/response_models.py` (populate fields later)

## Function signatures

### 1) SQLGlot utilities

```python
# src/codeintel/storage/sqlglot_utils.py

from __future__ import annotations

import hashlib
from sqlglot import exp
from sqlglot.optimizer import optimize
from sqlglot.optimizer.normalize import normalize

DUCKDB_DIALECT = "duckdb"

def canonicalize_sqlglot(
    expression: exp.Expression,
    *,
    dialect: str = DUCKDB_DIALECT,
    qualify: bool = True,
) -> exp.Expression:
    """Return a canonical SQLGlot AST suitable for stable hashing."""
    ...

def fingerprint_sqlglot(
    expression: exp.Expression,
    *,
    dialect: str = DUCKDB_DIALECT,
    prefix: str = "q_",
) -> str:
    """Stable hash for a canonicalized SQLGlot AST."""
    ...
```

### 2) IbisGateway: AST access

```python
# src/codeintel/storage/ibis_adapter.py

from sqlglot import exp

class IbisGateway:
    ...

    def to_sqlglot(self, expr: "it.Table") -> exp.Expression:
        """Compile an Ibis expression to a SQLGlot AST (DuckDB dialect)."""
        ...

    def query_fingerprint(self, expr: "it.Table") -> str:
        """Return stable fingerprint for an Ibis expression."""
        ...
```

## Acceptance criteria

- We can compile any serving semantic query expression to SQLGlot AST.
- Fingerprints are stable across formatting/whitespace differences.
- Canonicalization is isolated to a single utility module.

## Tests to add

- `tests/storage/test_sqlglot_fingerprint_stability.py`

Suggested tests:
- Two semantically identical expressions compile to different SQL strings but same fingerprint after canonicalization.
- Fingerprint prefix formatting is stable.

---

# PR8 — Contract-driven filter typing + large IN-list staging + optional param binding

## Why
Serving currently validates identifiers but not operator/type compatibility, and `IN` lists can produce huge SQL strings.

We want:
- fast fail for invalid operator/type combos
- scalable IN-list handling (staging or parameter arrays)
- (optional) param binding strategy for stable templates

## Files

- `src/codeintel/serving/semantic/query_builder.py`
- `src/codeintel/serving/semantic/models.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/semantic/in_list.py` (new)
- `src/codeintel/serving/semantic/validation.py` (new, recommended)

## Function signatures

### 1) Operator/type validation

```python
# src/codeintel/serving/semantic/validation.py

from __future__ import annotations
from dataclasses import dataclass

from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.inventory import SchemaInventory

class FilterValidationError(ValueError):
    pass

def validate_filters_against_schema(
    *,
    filters: list[FilterSpec],
    table_key: str,
    inventory: SchemaInventory,
) -> None:
    """Validate operator compatibility (string ops, numeric comparisons, IN types)."""
    ...
```

### 2) IN-list staging (execution-scoped)

```python
# src/codeintel/serving/semantic/in_list.py

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import ibis
import ibis.expr.types as it

@contextmanager
def stage_in_list_values(
    ibis_con: "ibis.backends.duckdb.Backend",
    *,
    name: str,
    values: list[object],
    dtype: str,
) -> Iterator[it.Table]:
    """Create a temp table of values and drop it on exit."""
    ...
```

### 3) Query builder: route IN filters through staged tables when needed

We avoid returning an expression that outlives the staging context. Instead, kernel wraps *build + execute* inside the staging context.

```python
# src/codeintel/serving/semantic/query_builder.py

def build_query(
    *,
    ibis_con: DuckDBBackend,
    plan: SemanticQueryPlan,
    staged_in_lists: dict[str, "it.Table"] | None = None,
) -> "it.Table":
    """Build expression using staged tables for large IN lists when provided."""
    ...
```

### 4) Kernel: stage + execute within generator / context

```python
# src/codeintel/serving/semantic/kernel.py

class SemanticQueryKernel:
    ...

    def _execute_semantic_plan(...):
        """Validate filters, stage IN lists if necessary, execute safely."""
        ...
```

## Acceptance criteria

- Invalid operator/type combinations fail before hitting DuckDB.
- Large IN filters do not create massive SQL strings or hit placeholder limits.
- Staged temp tables are always cleaned up (even on exceptions / disconnects).
- Existing behavior for small IN lists remains unchanged.

## Tests to add

- `tests/serving/test_filter_validation.py`
- `tests/serving/test_in_list_staging_cleanup.py`

Suggested tests:
- `contains` on INTEGER raises `FilterValidationError`.
- IN list above threshold triggers staging, and temp table is dropped on exit (even if query fails).

---

# PR9 — Serving telemetry + fingerprints in responses + optional query cache

## Why
We want consistent observability + stable provenance across HTTP and MCP.

## Files

- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/http/metrics.py` (optional integration)
- `src/codeintel/serving/mcp/response_models.py`
- `src/codeintel/serving/mcp/app.py` (populate query_hash/schema_hash)
- `src/codeintel/serving/semantic/cache.py` (new, optional)

## Function signatures

### 1) Telemetry dataclass + timed execute helper

```python
# src/codeintel/serving/semantic/telemetry.py (new, recommended)

from dataclasses import dataclass
from time import perf_counter
from typing import Any

@dataclass(frozen=True)
class QueryTelemetry:
    sql: str
    duration_ms: float
    query_hash: str | None = None
    schema_hash: str | None = None
    rows_returned: int | None = None

def timed_execute(con: Any, *, sql: str, params: list[object] | None = None) -> tuple[Any, QueryTelemetry]:
    ...
```

### 2) Kernel populates query_hash + schema_hash

```python
# src/codeintel/serving/semantic/kernel.py

class SemanticQueryKernel:
    ...

    def _schema_hash_for_view(self, *, view_id: str, inventory: SchemaInventory) -> str | None:
        """Hash column names + types from SchemaInventory."""
        ...
```

### 3) Optional cache interface

```python
# src/codeintel/serving/semantic/cache.py

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass

@dataclass(frozen=True)
class CacheKey:
    query_hash: str
    schema_hash: str

class QueryCache:
    def get(self, key: CacheKey) -> list[dict[str, object]] | None: ...
    def put(self, key: CacheKey, rows: list[dict[str, object]]) -> None: ...
```

(If caching is enabled, keep it conservative: only cache non-export query() results up to a small limit.)

## Acceptance criteria

- Telemetry includes duration and stable fingerprints.
- MCP export metadata includes schema_hash/query_hash (when available).
- Optional cache (if implemented) is correctness-first and can be disabled by default.

## Tests to add

- `tests/serving/test_query_fingerprints_in_responses.py`
- `tests/serving/test_query_telemetry.py`
- `tests/serving/test_query_cache_correctness.py` (only if cache included)

---

# PR10 — View semantic diff artifacts + compiler upgrade gates

## Why
View SQL drift is hard to review. SQLGlot/Ibis upgrades can silently change semantics.

We want:
- build artifact: semantic diff of old vs new view SQL
- tests that gate compiler upgrades via golden SQL snapshots + execution checks

## Files

- `src/codeintel/storage/views/diff.py` (new)
- `src/codeintel/storage/views/dependencies.py` (extend)
- `tests/compiler/test_golden_sql.py` (new)
- `tests/compiler/test_execution_validation.py` (new)

## Function signatures

```python
# src/codeintel/storage/views/diff.py

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ViewDiffArtifact:
    view_key: str
    old_sql: str
    new_sql: str
    semantic_diff: str  # human-readable diff summary
    old_fingerprint: str
    new_fingerprint: str

def diff_view_sql(*, old_sql: str, new_sql: str) -> ViewDiffArtifact:
    ...
```

Golden SQL fixtures:

```python
# tests/compiler/test_golden_sql.py

def test_golden_sql_snapshots_are_stable(fresh_gateway: StorageGateway) -> None:
    """Compile representative expressions/views and compare against stored snapshots."""
```

## Acceptance criteria

- Build produces a semantic diff artifact for view changes.
- A “compiler upgrade” PR that bumps SQLGlot/Ibis must update golden snapshots intentionally.
- Execution validation test suite runs critical queries on DuckDB and asserts same row-level results.

## Tests to add

- `tests/compiler/test_golden_sql.py`
- `tests/compiler/test_execution_validation.py`

---

## Test matrix: risk → tests

This maps the risks in the enhancement plan to specific tests you can implement and enforce in CI.

| Risk | What can go wrong | Test(s) that catch it | Where |
|------|-------------------|------------------------|-------|
| Serving export buffering regressions | OOM, latency, time-to-first-byte | NDJSON route unit test ensuring no `list()`; kernel streaming test ensuring no `fetchall()` | `tests/serving/test_export_streaming_ndjson.py` |
| Client disconnect/backpressure | leaked connections/temp tables, pool starvation | generator-close cleanup test; staging cleanup on early close | `tests/serving/test_export_streaming_ndjson.py`, `tests/serving/test_in_list_staging_cleanup.py` |
| `LOAD` vs `INSTALL` misconfiguration | missing extensions in prod or accidental installs | bootstrap policy tests (read-only forbids install; clear error messaging) | `tests/storage/test_duckdb_bootstrap_policy.py` |
| Upsert fast-lane semantic drift | wrong rows updated/inserted | compare values-based upsert vs upsert-select for same inputs; conflict behavior tests | `tests/storage/test_upsert_from_expression_no_pandas.py` |
| Cache correctness | wrong results served | cache key includes schema_hash + query_hash; test that schema change invalidates cache | `tests/serving/test_query_cache_correctness.py` |
| SQLGlot/Ibis compiler drift | silent query semantics changes | golden SQL snapshot diffs + execution validation | `tests/compiler/test_golden_sql.py`, `tests/compiler/test_execution_validation.py` |
| Complex types blast radius | schema/model incompatibility | (deferred) ensure ColumnType stays primitive-only; contract validation test | `tests/storage/test_schema_ddl_roundtrip.py` (guard) |

---

## Rollout notes

- **Keep fallbacks initially, but measurable**: for example, bulk-write fallback to tuple inserts stays for small DataFrames and for environments without pyarrow.
- **Add feature flags where operationally useful**:
  - export spooling directory
  - bulk write threshold
  - cache enablement
  - “allow install” policy (default off for serving)
- **Prefer inventory-driven metadata**: schema types/columns should come from `SchemaInventory` rather than discovering from first row.

---

## Definition of done

You can declare the “data ops backbone” best-in-class when:

1. Export endpoints (HTTP + MCP) handle million-row results without OOM.
2. No pandas materialization occurs on write paths except explicitly chosen by caller.
3. Extensions/secrets/init SQL are deterministic and controlled via policy.
4. Schema DDL is generated through a contract-driven round trip (Ibis → SQLGlot).
5. Compiler upgrades are gated by golden snapshots and execution tests.
