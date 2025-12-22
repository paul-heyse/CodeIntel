

Below is a **drop-in, diff-style patch set** that layers in the “statement redaction hook” for DuckDB spans:

* **Default behavior:** DB spans **do not emit raw SQL** into `db.statement`.
* Instead, they emit: **operation name + stable hash prefix** (e.g., `SELECT [sha256:ab12…]`), while also attaching the full digest as `codeintel.db.statement.sha256`.
* It’s wired at your **single connection-creation point** (`DuckDBSession`), so it becomes automatically consistent across **CLI / HTTP / MCP** as long as they all go through the same session bootstrap.

Why this works: OpenTelemetry’s DB-API instrumentation populates `db.statement` from the first argument to `cursor.execute` / `executemany`, so we intercept that at the tracing proxy layer and replace it with a safe fingerprint. ([OpenTelemetry Python Contrib][1])

---

## Behavior and knobs

### Defaults (safe for prod)

* `db.statement`: `"<OP> [sha256:<prefix>]"` (no raw SQL)
* `codeintel.db.statement.sha256`: full SHA-256 hex digest of a normalized statement shape

### Environment variables

* `CODEINTEL_OTEL_DUCKDB_TRACING`

  * If set to `false/0/off`: disables this instrumentation.
* `CODEINTEL_OTEL_DB_STATEMENT_MODE`

  * `hash` (default) → `SELECT [sha256:abcd1234…]`
  * `operation` → `SELECT`
  * `none` → omit `db.statement` (empty display), but keep the hash attribute
  * `full` → **unsafe** (emits full SQL)
* `CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN`

  * default `16` (controls the prefix length shown in `db.statement`)

---

## Patch set

### 1) Add a small centralized observability package

```diff
--- /dev/null
+++ b/src/codeintel/observability/__init__.py
@@ -0,0 +1,20 @@
+"""Cross-cutting observability utilities.
+
+This package is intentionally lightweight and safe to import from both CLI and
+serving entrypoints.
+
+The primary goal is to keep observability concerns *centralized* and
+transport-agnostic (CLI/HTTP/MCP).
+"""
+
+from __future__ import annotations
+
+from codeintel.observability.duckdb_tracing import ensure_duckdb_tracing
+from codeintel.observability.sql_redaction import SQLStatementMode, RedactedSQL, redact_sql
+
+__all__ = [
+    "SQLStatementMode",
+    "RedactedSQL",
+    "ensure_duckdb_tracing",
+    "redact_sql",
+]
```

---

### 2) Add the SQL redaction + stable hashing utility

```diff
--- /dev/null
+++ b/src/codeintel/observability/sql_redaction.py
@@ -0,0 +1,154 @@
+"""SQL statement redaction utilities.
+
+This module exists for one reason: *avoid leaking raw SQL text* (which may
+contain user-controlled strings, PII, or secrets) into tracing backends while
+still preserving performance insight.
+
+The primary use-case is OpenTelemetry database spans (e.g. DB-API
+instrumentation), where the default behavior is to attach the executed SQL to a
+span attribute (commonly `db.statement`).
+"""
+
+from __future__ import annotations
+
+import hashlib
+import re
+from dataclasses import dataclass
+from typing import Literal, overload
+
+
+SQLStatementMode = Literal["full", "hash", "operation", "none"]
+
+
+_LEADING_WS_RE = re.compile(r"^\s+")
+_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
+_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
+
+# Conservative literal redaction. The goal is stable hashing (grouping), not a
+# perfect SQL lexer.
+_SQL_SINGLE_QUOTED_STRING_RE = re.compile(r"'(?:''|[^'])*'")
+_SQL_HEX_LITERAL_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
+_SQL_UUID_RE = re.compile(
+    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
+)
+_SQL_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
+_WS_RE = re.compile(r"\s+")
+
+
+@dataclass(frozen=True, slots=True)
+class RedactedSQL:
+    """Result of applying a redaction policy to a SQL statement."""
+
+    mode: SQLStatementMode
+    operation: str
+    """Best-effort SQL operation (e.g., SELECT/INSERT/CREATE)."""
+
+    statement_hash: str | None
+    """SHA-256 hex digest of a normalized form of the statement (if computed)."""
+
+    display: str
+    """Safe string suitable for attaching to telemetry (e.g., `db.statement`)."""
+
+
+def _to_text(statement: str | bytes) -> str:
+    if isinstance(statement, bytes):
+        return statement.decode("utf-8", "replace")
+    return statement
+
+
+def _strip_comments(sql: str) -> str:
+    # Remove /* ... */ first, then -- ...
+    sql = _SQL_BLOCK_COMMENT_RE.sub(" ", sql)
+    sql = _SQL_LINE_COMMENT_RE.sub(" ", sql)
+    return sql
+
+
+def _extract_operation(sql: str) -> str:
+    sql = _strip_comments(sql)
+    sql = _LEADING_WS_RE.sub("", sql)
+    head = sql.split(maxsplit=1)
+    return head[0] if head else ""
+
+
+def _normalize_for_hash(sql: str) -> str:
+    """Normalize SQL for stable hashing.
+
+    This intentionally does *not* attempt full parsing. It aims to:
+      1) remove comments
+      2) replace common literal forms with `?`
+      3) collapse whitespace
+    """
+
+    sql = _strip_comments(sql)
+    sql = _SQL_SINGLE_QUOTED_STRING_RE.sub("?", sql)
+    sql = _SQL_HEX_LITERAL_RE.sub("?", sql)
+    sql = _SQL_UUID_RE.sub("?", sql)
+    sql = _SQL_NUMBER_RE.sub("?", sql)
+    sql = _WS_RE.sub(" ", sql).strip()
+    return sql
+
+
+def _sha256_hex(text: str) -> str:
+    return hashlib.sha256(text.encode("utf-8"), usedforsecurity=False).hexdigest()
+
+
+@overload
+def redact_sql(statement: str, *, mode: SQLStatementMode = "hash", hash_len: int = 16) -> RedactedSQL:
+    ...
+
+
+@overload
+def redact_sql(statement: bytes, *, mode: SQLStatementMode = "hash", hash_len: int = 16) -> RedactedSQL:
+    ...
+
+
+def redact_sql(
+    statement: str | bytes,
+    *,
+    mode: SQLStatementMode = "hash",
+    hash_len: int = 16,
+) -> RedactedSQL:
+    """Redact a SQL statement for safe telemetry.
+
+    Parameters
+    ----------
+    statement
+        Raw SQL statement (bytes or str).
+    mode
+        Redaction mode.
+        - "full": return original SQL as display (unsafe for production).
+        - "hash": return "<OP> [sha256:<prefix>]".
+        - "operation": return only operation keyword (e.g., "SELECT").
+        - "none": return empty string.
+    hash_len
+        Prefix length of the SHA-256 hex digest to include in display for
+        "hash" mode.
+
+    Returns
+    -------
+    RedactedSQL
+        Redacted representation.
+    """
+
+    text = _to_text(statement)
+    operation = _extract_operation(text)
+
+    if mode == "full":
+        return RedactedSQL(mode=mode, operation=operation, statement_hash=None, display=text)
+
+    normalized = _normalize_for_hash(text)
+    digest = _sha256_hex(normalized) if normalized else None
+
+    if mode == "none":
+        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display="")
+
+    if mode == "operation":
+        return RedactedSQL(mode=mode, operation=operation, statement_hash=digest, display=operation)
+
+    # Default: "hash"
+    prefix = (digest or "")[: max(0, hash_len)]
+    display = f"{operation} [sha256:{prefix}]" if operation and prefix else operation
+    return RedactedSQL(mode="hash", operation=operation, statement_hash=digest, display=display)
+
+
+__all__ = ["RedactedSQL", "SQLStatementMode", "redact_sql"]
```

---

### 3) Add the DuckDB tracing wrapper that *redacts* `db.statement`

This is the “redaction hook” layer. It wraps `duckdb.connect` once, and then instruments:

* `con.execute(...)` (important for DuckDB usage patterns)
* `con.cursor().execute(...)` (for completeness)

```diff
--- /dev/null
+++ b/src/codeintel/observability/duckdb_tracing.py
@@ -0,0 +1,207 @@
+"""DuckDB OpenTelemetry (DB-API) tracing helpers.
+
+This module is designed to be called from the *single* DuckDB connection
+creation point (``DuckDBSession``) so that:
+
+    HTTP/MCP span → semantic operation span → DuckDB span (with query timing)
+
+The incremental feature in this patch is *SQL statement redaction*:
+we ensure the DB span does not include full SQL text in the `db.statement`
+attribute by default. Instead we attach a stable hash fingerprint.
+
+Implementation notes
+-------------------
+OpenTelemetry's DB-API instrumentation sets ``db.statement`` from the first
+argument passed to ``cursor.execute``/``executemany``.
+
+To avoid emitting raw SQL, we provide a small custom connection/cursor proxy
+that uses a custom ``CursorTracer`` which rewrites span attributes before they
+are recorded.
+"""
+
+from __future__ import annotations
+
+from functools import lru_cache
+from typing import TYPE_CHECKING
+
+from codeintel.core.env import get_bool, get_int, get_str
+from codeintel.observability.sql_redaction import SQLStatementMode, redact_sql
+
+if TYPE_CHECKING:
+    from collections.abc import Callable
+
+
+_ENV_DUCKDB_TRACING_ENABLED = "CODEINTEL_OTEL_DUCKDB_TRACING"
+_ENV_DB_STATEMENT_MODE = "CODEINTEL_OTEL_DB_STATEMENT_MODE"
+_ENV_DB_STATEMENT_HASH_LEN = "CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN"
+
+
+def _get_statement_mode() -> SQLStatementMode:
+    raw = get_str(_ENV_DB_STATEMENT_MODE)
+    if raw is None:
+        return "hash"
+    value = raw.strip().lower()
+    if value in {"full", "hash", "operation", "none"}:
+        return value  # type: ignore[return-value]
+    # Fail closed: do not leak SQL.
+    return "hash"
+
+
+def _get_hash_len() -> int:
+    return int(get_int(_ENV_DB_STATEMENT_HASH_LEN) or 16)
+
+
+@lru_cache(maxsize=1)
+def ensure_duckdb_tracing() -> None:
+    """Enable DuckDB DB-API tracing (idempotent).
+
+    This wraps ``duckdb.connect`` so any connection created afterwards will
+    emit DB spans.
+
+    Redaction behavior is controlled via:
+      - ``CODEINTEL_OTEL_DB_STATEMENT_MODE``: full|hash|operation|none
+      - ``CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN``: integer prefix length
+    """
+
+    enabled = get_bool(_ENV_DUCKDB_TRACING_ENABLED)
+    if enabled is False:
+        return
+
+    try:
+        import duckdb
+
+        import wrapt
+        from opentelemetry import trace as trace_api
+        from opentelemetry.instrumentation.dbapi import (
+            CursorTracer,
+            DatabaseApiIntegration,
+            wrap_connect,
+        )
+        from opentelemetry.semconv.trace import SpanAttributes
+    except Exception:
+        # Observability must never prevent CodeIntel from working.
+        return
+
+    mode = _get_statement_mode()
+    hash_len = _get_hash_len()
+
+    class _RedactingCursorTracer(CursorTracer):
+        """Cursor tracer that replaces `db.statement` with a safe fingerprint."""
+
+        def _populate_span(self, span: trace_api.Span, cursor, *args):
+            if not span.is_recording():
+                return
+
+            # Base behavior sets db.system + db.name + db.statement; we do
+            # the same, but ensure db.statement never contains raw SQL.
+            raw_statement = super().get_statement(cursor, args)
+            redacted = redact_sql(raw_statement, mode=mode, hash_len=hash_len)
+
+            span.set_attribute(SpanAttributes.DB_SYSTEM, self._db_api_integration.database_system)
+            span.set_attribute(SpanAttributes.DB_NAME, self._db_api_integration.database)
+
+            # Keep low-cardinality operation name as an explicit attribute.
+            # (Span name is already the operation in DB-API instrumentation.)
+            if redacted.operation:
+                span.set_attribute("db.operation", redacted.operation)
+                span.set_attribute("db.operation.name", redacted.operation)
+
+            # Store a stable fingerprint for grouping.
+            if redacted.statement_hash:
+                span.set_attribute("codeintel.db.statement.sha256", redacted.statement_hash)
+
+            # And store a safe display value in the conventional attribute.
+            if redacted.display:
+                span.set_attribute(SpanAttributes.DB_STATEMENT, redacted.display)
+
+            # Preserve any connection-level attributes collected by the DB-API integration.
+            for k, v in self._db_api_integration.span_attributes.items():
+                span.set_attribute(k, v)
+
+    class _RedactingTracedCursorProxy(wrapt.ObjectProxy):
+        def __init__(self, cursor, db_api_integration: DatabaseApiIntegration):
+            super().__init__(cursor)
+            self._self_cursor_tracer = _RedactingCursorTracer(db_api_integration)
+
+        def execute(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.execute, *args, **kwargs
+            )
+            # DuckDB returns the cursor/connection for chaining.
+            return self if result is self.__wrapped__ else result
+
+        def executemany(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.executemany, *args, **kwargs
+            )
+            return self if result is self.__wrapped__ else result
+
+        def callproc(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.callproc, *args, **kwargs
+            )
+            return self if result is self.__wrapped__ else result
+
+        def __enter__(self):
+            self.__wrapped__.__enter__()
+            return self
+
+        def __exit__(self, *args, **kwargs):
+            return self.__wrapped__.__exit__(*args, **kwargs)
+
+    class _RedactingTracedConnectionProxy(wrapt.ObjectProxy):
+        def __init__(self, connection, db_api_integration: DatabaseApiIntegration):
+            super().__init__(connection)
+            self._self_db_api_integration = db_api_integration
+            # DuckDB users typically call `con.execute(...)` directly (without
+            # an explicit cursor). Instrument the connection object itself.
+            self._self_cursor_tracer = _RedactingCursorTracer(db_api_integration)
+
+        def execute(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.execute, *args, **kwargs
+            )
+            return self if result is self.__wrapped__ else result
+
+        def executemany(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.executemany, *args, **kwargs
+            )
+            return self if result is self.__wrapped__ else result
+
+        def callproc(self, *args, **kwargs):
+            result = self._self_cursor_tracer.traced_execution(
+                self.__wrapped__, self.__wrapped__.callproc, *args, **kwargs
+            )
+            return self if result is self.__wrapped__ else result
+
+        def cursor(self, *args, **kwargs):
+            cursor = self.__wrapped__.cursor(*args, **kwargs)
+            return _RedactingTracedCursorProxy(cursor, self._self_db_api_integration)
+
+        def __enter__(self):
+            self.__wrapped__.__enter__()
+            return self
+
+        def __exit__(self, *args, **kwargs):
+            return self.__wrapped__.__exit__(*args, **kwargs)
+
+    class _RedactingDatabaseApiIntegration(DatabaseApiIntegration):
+        def wrapped_connection(self, connect_method: Callable[..., object], args, kwargs):
+            connection = connect_method(*args, **kwargs)
+            self.get_connection_attributes(connection)
+            return _RedactingTracedConnectionProxy(connection, self)
+
+    # Wrap the duckdb.connect function itself (best single connection-creation point).
+    wrap_connect(
+        __name__,
+        duckdb,
+        "connect",
+        database_system="duckdb",
+        capture_parameters=False,
+        enable_commenter=False,
+        db_api_integration_factory=_RedactingDatabaseApiIntegration,
+    )
+
+
+__all__ = ["ensure_duckdb_tracing"]
```

---

### 4) Wire it into the single connection-creation point

```diff
--- a/src/codeintel/storage/backend/duckdb_session.py
+++ b/src/codeintel/storage/backend/duckdb_session.py
@@ -22,6 +22,7 @@
 
 import duckdb
 
+from codeintel.observability.duckdb_tracing import ensure_duckdb_tracing
 from codeintel.storage.gateway.extensions import (
     load_extensions_from_env,
     load_required_extensions,
@@ -292,6 +293,13 @@
 def _open_primary_connection(
     config: StorageConfig,
     *,
     duckdb_config: DuckDBConnectConfig | None = None,
 ) -> DuckDBConnection:
+    # Best-effort: enable DB-API tracing for DuckDB at the single connection
+    # creation point.
+    #
+    # This is intentionally safe/no-op if OpenTelemetry is not installed or not
+    # configured.
+    ensure_duckdb_tracing()
+
     cfg = duckdb_config
     if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
         con = duckdb.connect(str(Path(":memory:")))
```

---

### 5) Add focused unit tests for the redaction behavior

```diff
--- /dev/null
+++ b/tests/observability/test_sql_redaction.py
@@ -0,0 +1,48 @@
+from __future__ import annotations
+
+from codeintel.observability.sql_redaction import redact_sql
+
+
+def test_redact_sql_default_hash_mode() -> None:
+    redacted = redact_sql("SELECT * FROM core.symbols WHERE id = 123")
+    assert redacted.mode == "hash"
+    assert redacted.operation == "SELECT"
+    assert redacted.statement_hash is not None
+    assert len(redacted.statement_hash) == 64
+    assert redacted.display.startswith("SELECT [sha256:")
+
+
+def test_redact_sql_hash_is_stable_across_literals() -> None:
+    a = redact_sql("SELECT * FROM t WHERE id = 1 AND name = 'alice'")
+    b = redact_sql("SELECT * FROM t WHERE id = 999 AND name = 'bob'")
+    assert a.statement_hash == b.statement_hash
+
+
+def test_redact_sql_strips_comments_for_hashing() -> None:
+    a = redact_sql("/*comment*/ SELECT * FROM t -- tail\n WHERE id = 1")
+    b = redact_sql("SELECT * FROM t WHERE id = 2")
+    assert a.statement_hash == b.statement_hash
+
+
+def test_redact_sql_operation_mode() -> None:
+    redacted = redact_sql("INSERT INTO t VALUES (1)", mode="operation")
+    assert redacted.display == "INSERT"
+    assert redacted.statement_hash is not None
+
+
+def test_redact_sql_none_mode() -> None:
+    redacted = redact_sql("DELETE FROM t WHERE x = 1", mode="none")
+    assert redacted.display == ""
+    assert redacted.statement_hash is not None
+
+
+def test_redact_sql_full_mode_is_unsafe_but_supported() -> None:
+    sql = "UPDATE t SET x = 'secret' WHERE id = 7"
+    redacted = redact_sql(sql, mode="full")
+    assert redacted.display == sql
+    assert redacted.statement_hash is None
+
+
+def test_redact_sql_accepts_bytes() -> None:
+    redacted = redact_sql(b"SELECT 1")
+    assert redacted.operation == "SELECT"
```

---

## What you’ll see in traces after this patch

For a DB call like:

```python
con.execute("SELECT * FROM core.symbols WHERE name = 'alice' AND id = 123")
```

Your span will have:

* **Span name:** `SELECT` (as before)
* **Attributes:**

  * `db.statement = "SELECT [sha256:8f0c4e6d...]"`  (safe)
  * `codeintel.db.statement.sha256 = "<64 hex chars>"`
  * `db.operation = "SELECT"` / `db.operation.name = "SELECT"`

No raw SQL will be emitted unless you explicitly opt into `CODEINTEL_OTEL_DB_STATEMENT_MODE=full`.

---

## One small follow-on (optional) improvement

If you want this to be *even more “best-in-class”* for ops dashboards, you can later add a second low-cardinality attribute (aligned with newer semantic conventions) like `db.query.summary` (e.g., `SELECT core.symbols`) **without** full query text. That’s consistent with the direction of the database semantic conventions. ([opentelemetry.io][2])

[1]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/_modules/opentelemetry/instrumentation/dbapi.html "opentelemetry.instrumentation.dbapi — OpenTelemetry Python Contrib  documentation"
[2]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/?utm_source=chatgpt.com "Semantic conventions for database client spans"


# low-cardianality attribyte db.query.summary #

Below is a **diff-style incremental patch set** that adds the “ops-dashboard friendly” low-cardinality attribute **`db.query.summary`** (and uses it as the DB span name), **without emitting full SQL text**. This follows the stable OpenTelemetry database semantic conventions guidance around `db.query.summary` and span naming. ([OpenTelemetry][1])

---

## Patch 1 — Add a SQLGlot-powered `summarize_sql_duckdb()` helper

**File:** `src/codeintel/storage/sqlglot_tools.py`

```diff
--- /mnt/data/codeintel_orig/src/codeintel/storage/sqlglot_tools.py	2025-12-20 06:34:15.000000000 +0000
+++ /mnt/data/codeintel_repo/src/codeintel/storage/sqlglot_tools.py	2025-12-22 05:38:43.035131121 +0000
@@ -13,6 +13,7 @@
 from __future__ import annotations
 
 import hashlib
+import re
 from collections.abc import Mapping
 from typing import TYPE_CHECKING
 
@@ -40,6 +41,7 @@
     "fingerprint_sql_duckdb",
     "parse_one_duckdb",
     "render_sql_duckdb",
+    "summarize_sql_duckdb",
 ]
 
 SchemaMapping = Mapping[str, Mapping[str, str]]
@@ -156,6 +158,187 @@
     return hashlib.sha256(canon.encode("utf-8")).hexdigest()
 
 
+_MAX_QUERY_SUMMARY_CHARS = 255
+_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
+
+
+def summarize_sql_duckdb(sql: str) -> str | None:
+    """Generate a low-cardinality summary for a DuckDB SQL string.
+
+    This is intended for observability attributes like ``db.query.summary``.
+    The summary is designed to be short, stable, and free of literal values.
+
+    Notes
+    -----
+    - Uses SQLGlot parsing when possible.
+    - Falls back to a conservative token-based summary when parsing fails.
+    - Truncates to 255 characters (without cutting inside a token), aligning with
+      the OpenTelemetry database semantic conventions.
+
+    Parameters
+    ----------
+    sql
+        DuckDB SQL string.
+
+    Returns
+    -------
+    str | None
+        A low-cardinality query summary (e.g., ``SELECT core.symbols``), or None
+        when the input is empty/unusable.
+    """
+    stripped = sql.strip()
+    if not stripped:
+        return None
+
+    try:
+        root = parse_one_duckdb(stripped)
+    except ParseError:
+        return _fallback_query_summary(stripped)
+
+    parts = _query_summary_parts_from_root(root, raw_sql=stripped)
+    return _truncate_query_summary_parts(parts)
+
+
+def _fallback_query_summary(sql: str) -> str | None:
+    tokens = _FALLBACK_TOKEN_RE.findall(sql)
+    if not tokens:
+        return None
+    # Keep only the first one or two identifier-like tokens to avoid capturing
+    # literal values (including quoted strings) and to keep cardinality low.
+    parts = [tokens[0]]
+    if len(tokens) > 1:
+        parts.append(tokens[1])
+    return _truncate_query_summary_parts(parts)
+
+
+def _query_summary_parts_from_root(root: exp.Expression, *, raw_sql: str) -> list[str]:
+    # Prefer AST-derived operation names so WITH ... SELECT ... is summarized as SELECT.
+    if isinstance(root, exp.Insert):
+        return _query_summary_parts_for_insert(root, raw_sql=raw_sql)
+
+    operation = _operation_name_for_root(root)
+    parts: list[str] = [operation] if operation else []
+    parts.extend(_query_summary_targets_for_expression(root, raw_sql=raw_sql))
+    return parts
+
+
+def _query_summary_parts_for_insert(root: exp.Insert, *, raw_sql: str) -> list[str]:
+    parts: list[str] = ["INSERT"]
+
+    target = _format_table_for_summary(getattr(root, "this", None))
+    if target:
+        parts.append(target)
+
+    nested = getattr(root, "expression", None)
+    if isinstance(nested, exp.Expression):
+        nested_op = _operation_name_for_root(nested) or "SELECT"
+        parts.append(nested_op)
+        nested_targets = _query_summary_targets_for_expression(
+            nested,
+            raw_sql=raw_sql,
+            exclude={target.lower()} if target else None,
+        )
+        parts.extend(nested_targets)
+
+    return parts
+
+
+def _operation_name_for_root(root: exp.Expression) -> str | None:
+    if isinstance(root, exp.Select):
+        return "SELECT"
+    if isinstance(root, exp.Update):
+        return "UPDATE"
+    if isinstance(root, exp.Delete):
+        return "DELETE"
+    if isinstance(root, exp.Insert):
+        return "INSERT"
+    if isinstance(root, exp.Create):
+        return "CREATE"
+    if isinstance(root, exp.Drop):
+        return "DROP"
+    key = getattr(root, "key", None)
+    if isinstance(key, str) and key:
+        return key.replace("_", " ").upper()
+    return None
+
+
+def _query_summary_targets_for_expression(
+    root: exp.Expression,
+    *,
+    raw_sql: str,
+    exclude: set[str] | None = None,
+) -> list[str]:
+    exclude = exclude or set()
+    sql_lower = raw_sql.lower()
+
+    # Prefer CTE-safe physical table extraction.
+    tables = extract_table_refs(root)
+    formatted: list[tuple[int, str]] = []
+    for table in tables:
+        key = _format_table_for_summary(table)
+        if not key:
+            continue
+        key_lower = key.lower()
+        if key_lower in exclude:
+            continue
+        # Attempt to preserve a human-friendly order by sorting by first appearance.
+        pos = _best_effort_table_position(sql_lower, key_lower)
+        formatted.append((pos, key))
+
+    formatted.sort(key=lambda item: item[0])
+
+    out: list[str] = []
+    seen: set[str] = set()
+    for _, key in formatted:
+        k = key.lower()
+        if k in seen:
+            continue
+        out.append(key)
+        seen.add(k)
+    return out
+
+
+def _best_effort_table_position(sql_lower: str, table_key_lower: str) -> int:
+    # Look for schema.table first; if absent, look for unqualified table.
+    pos = sql_lower.find(table_key_lower)
+    if pos != -1:
+        return pos
+    if "." in table_key_lower:
+        _, table = table_key_lower.split(".", 1)
+        pos2 = sql_lower.find(table)
+        if pos2 != -1:
+            return pos2
+    # If not found, put it later but keep deterministic.
+    return 10**9
+
+
+def _format_table_for_summary(node: object) -> str | None:
+    if not isinstance(node, exp.Table):
+        return None
+    schema = node.db
+    name = node.name
+    if not name:
+        return None
+    if schema:
+        return f"{schema}.{name}"
+    return name
+
+
+def _truncate_query_summary_parts(parts: list[str]) -> str:
+    kept: list[str] = []
+    length = 0
+    for part in parts:
+        part = part.strip()
+        if not part:
+            continue
+        add_len = len(part) + (1 if kept else 0)
+        if length + add_len > _MAX_QUERY_SUMMARY_CHARS:
+            break
+        kept.append(part)
+        length += add_len
+    return " ".join(kept)
+
+
 def extract_table_refs(root: exp.Expression) -> tuple[exp.Table, ...]:
     """Extract physical table references from a parsed AST.
```

---

## Patch 2 — Unit tests for query summaries

**File:** `tests/storage/test_sqlglot_tools.py`

```diff
--- /mnt/data/codeintel_orig/tests/storage/test_sqlglot_tools.py	2025-12-20 06:18:18.000000000 +0000
+++ /mnt/data/codeintel_repo/tests/storage/test_sqlglot_tools.py	2025-12-22 05:39:22.593875785 +0000
@@ -11,6 +11,7 @@
     extract_table_keys_duckdb,
     fingerprint_sql_duckdb,
     parse_one_duckdb,
+    summarize_sql_duckdb,
 )
 from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
 
@@ -77,3 +78,39 @@
         lineage["repo_commit"],
         frozenset({"core.modules.repo", "core.modules.commit"}),
     )
+
+
+def test_summarize_sql_duckdb_emits_low_cardinality_select() -> None:
+    """summarize_sql_duckdb summarizes SELECT queries with table targets."""
+    summary = summarize_sql_duckdb("SELECT * FROM core.symbols WHERE name = 'foo'")
+    expect_equal(summary, "SELECT core.symbols")
+
+
+def test_summarize_sql_duckdb_ignores_cte_names() -> None:
+    """summarize_sql_duckdb uses physical tables, not CTE aliases."""
+    sql = """
+    WITH t AS (
+        SELECT * FROM core.modules
+    )
+    SELECT *
+    FROM t
+    JOIN analytics.function_metrics fm ON 1 = 1
+    """
+    summary = summarize_sql_duckdb(sql)
+    # Order is based on best-effort appearance; ensure the CTE name isn't present.
+    expect_true(summary is not None, message="summary produced")
+    assert summary is not None
+    expect_true(" t" not in summary.lower(), message="cte alias not included")
+    expect_true("core.modules" in summary, message="physical table included")
+    expect_true("analytics.function_metrics" in summary, message="physical table included")
+
+
+def test_summarize_sql_duckdb_handles_insert_select() -> None:
+    """summarize_sql_duckdb includes INSERT target + SELECT source."""
+    sql = """
+    INSERT INTO analytics.rollups
+    SELECT *
+    FROM core.symbols
+    """
+    summary = summarize_sql_duckdb(sql)
+    expect_equal(summary, "INSERT analytics.rollups SELECT core.symbols")
```

---

## Patch 3 — Add DuckDB span wrapper that emits `db.query.summary` and no SQL text

This is the “incremental patch” that actually **adds `db.query.summary` onto the DuckDB spans**, and **uses it as the span name** (what most dashboards group by). That’s directly in line with the database semconv direction. ([OpenTelemetry][1])

**File (new):** `src/codeintel/storage/backend/duckdb_tracing.py`

```diff
--- /dev/null	2025-12-22 05:30:13.867664352 +0000
+++ /mnt/data/codeintel_repo/src/codeintel/storage/backend/duckdb_tracing.py	2025-12-22 05:48:38.888307823 +0000
@@ -0,0 +1,231 @@
+"""OpenTelemetry tracing wrappers for DuckDB connections.
+
+This module provides a single composition point for database tracing:
+``DuckDBSession`` creates a DuckDB connection, and then (optionally) wraps it
+with a lightweight proxy that emits OpenTelemetry database spans.
+
+Design goals
+------------
+- **Centralized**: instrumentation happens at connection creation.
+- **Low-noise by default**: only create DB spans when there is already an active
+  parent span (typical for HTTP/MCP request spans, CLI command spans, etc.).
+- **Privacy by default**: do not emit full SQL text. Instead, emit:
+  - ``db.query.summary`` (low-cardinality grouping key)
+  - a one-way SQL hash (debugging / correlation)
+
+The emitted attributes align with the stable OpenTelemetry database semantic
+conventions, which recommend using ``db.query.summary`` as the span name when
+available.
+"""
+
+from __future__ import annotations
+
+import hashlib
+import os
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, Any, Callable
+
+from codeintel.storage.sqlglot_tools import summarize_sql_duckdb
+
+try:
+    from opentelemetry import trace as otel_trace
+    from opentelemetry.trace import SpanKind
+    from opentelemetry.trace.status import Status, StatusCode
+except ImportError:  # pragma: no cover
+    otel_trace = None  # type: ignore[assignment]
+    SpanKind = None  # type: ignore[assignment]
+    Status = None  # type: ignore[assignment]
+    StatusCode = None  # type: ignore[assignment]
+
+if TYPE_CHECKING:
+    from collections.abc import Sequence
+
+    from codeintel.storage.gateway.protocol import DuckDBConnection
+
+__all__ = ["instrument_duckdb_connection"]
+
+
+_OTEL_ENABLED_ENV = "CODEINTEL_OTEL_DUCKDB_ENABLED"
+_OTEL_REQUIRE_PARENT_ENV = "CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT"
+_OTEL_CAPTURE_HASH_ENV = "CODEINTEL_OTEL_DUCKDB_CAPTURE_SQL_HASH"
+
+_DB_SYSTEM_NAME = "duckdb"
+
+
+def instrument_duckdb_connection(con: DuckDBConnection) -> DuckDBConnection:
+    """Return an OpenTelemetry-instrumented DuckDB connection.
+
+    If OpenTelemetry is not installed, or if instrumentation is disabled via
+    environment variables, returns the original connection.
+    """
+    if otel_trace is None:
+        return con
+
+    if not _env_flag(_OTEL_ENABLED_ENV, default=True):
+        return con
+
+    # Avoid double-wrapping in case a connection passes through multiple layers.
+    if isinstance(con, _TracedDuckDBConnection):
+        return con
+
+    return _TracedDuckDBConnection(con)  # type: ignore[return-value]
+
+
+def _env_flag(name: str, *, default: bool) -> bool:
+    raw = os.environ.get(name)
+    if raw is None:
+        return default
+    return raw.strip().lower() not in {"0", "false", "no", "off"}
+
+
+def _should_start_db_span() -> bool:
+    if otel_trace is None:
+        return False
+
+    if not _env_flag(_OTEL_ENABLED_ENV, default=True):
+        return False
+
+    if _env_flag(_OTEL_REQUIRE_PARENT_ENV, default=True):
+        span = otel_trace.get_current_span()
+        if span is None:
+            return False
+        ctx = span.get_span_context()
+        return bool(ctx and getattr(ctx, "is_valid", False))
+
+    return True
+
+
+def _sql_hash(sql: str) -> str:
+    return hashlib.sha256(sql.encode("utf-8")).hexdigest()
+
+
+def _operation_name_from_summary(summary: str | None) -> str | None:
+    if not summary:
+        return None
+    first = summary.split(" ", 1)[0].strip()
+    return first or None
+
+
+def _execute_with_db_span(fn: Callable[[], Any], *, sql: str) -> Any:
+    if not _should_start_db_span():
+        return fn()
+
+    assert otel_trace is not None
+
+    summary = summarize_sql_duckdb(sql)
+    operation = _operation_name_from_summary(summary)
+
+    attributes: dict[str, Any] = {
+        "db.system.name": _DB_SYSTEM_NAME,
+    }
+
+    if operation:
+        attributes["db.operation.name"] = operation
+
+    if summary:
+        attributes["db.query.summary"] = summary
+
+    if _env_flag(_OTEL_CAPTURE_HASH_ENV, default=True):
+        attributes["codeintel.db.query.hash"] = _sql_hash(sql)
+
+    # Per semconv guidance, prefer db.query.summary as the span name.
+    span_name = summary or operation or "db.query"
+
+    tracer = otel_trace.get_tracer(__name__)
+    with tracer.start_as_current_span(
+        span_name,
+        kind=SpanKind.CLIENT if SpanKind is not None else None,
+        attributes=attributes,
+    ) as span:
+        try:
+            return fn()
+        except Exception as exc:
+            span.record_exception(exc)
+            # Mark status ERROR (OpenTelemetry Python pattern). :contentReference[oaicite:2]{index=2}
+            if Status is not None and StatusCode is not None:
+                span.set_status(Status(StatusCode.ERROR, str(exc)))
+            raise
+
+
+@dataclass(slots=True)
+class _TracedDuckDBCursor:
+    """Cursor proxy that instruments ``execute``/``executemany``."""
+
+    _cursor: Any
+
+    def execute(self, sql: str, params: Sequence[object] | None = None) -> _TracedDuckDBCursor:
+        def _call() -> Any:
+            if params is None:
+                return self._cursor.execute(sql)
+            return self._cursor.execute(sql, params)
+
+        _execute_with_db_span(_call, sql=sql)
+        return self
+
+    def executemany(
+        self,
+        sql: str,
+        seq_of_params: Sequence[Sequence[object]],
+    ) -> _TracedDuckDBCursor:
+        def _call() -> Any:
+            return self._cursor.executemany(sql, seq_of_params)
+
+        _execute_with_db_span(_call, sql=sql)
+        return self
+
+    def __getattr__(self, name: str) -> Any:  # pragma: no cover
+        return getattr(self._cursor, name)
+
+
+@dataclass(slots=True)
+class _TracedDuckDBConnection:
+    """Connection proxy that instruments DuckDB SQL execution."""
+
+    _con: Any
+
+    def execute(
+        self,
+        sql: str,
+        params: Sequence[object] | None = None,
+    ) -> _TracedDuckDBConnection:
+        def _call() -> Any:
+            if params is None:
+                return self._con.execute(sql)
+            return self._con.execute(sql, params)
+
+        _execute_with_db_span(_call, sql=sql)
+
+        # DuckDB's .execute returns the connection; preserve chaining while
+        # keeping the proxy active.
+        return self
+
+    def cursor(self, *args: Any, **kwargs: Any) -> _TracedDuckDBCursor:
+        cur = self._con.cursor(*args, **kwargs)
+        return _TracedDuckDBCursor(cur)
+
+    def close(self) -> None:
+        return self._con.close()
+
+    def __enter__(self) -> _TracedDuckDBConnection:  # pragma: no cover
+        self._con.__enter__()
+        return self
+
+    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:  # pragma: no cover
+        return self._con.__exit__(exc_type, exc, tb)
+
+    def __getattr__(self, name: str) -> Any:  # pragma: no cover
+        return getattr(self._con, name)
```

What this achieves:

* **Span name** becomes the summary string (e.g., `SELECT core.symbols`) which is exactly what most APM dashboards want. ([OpenTelemetry][1])
* Span carries:

  * `db.query.summary`: low-card grouping key (no literal values)
  * `db.operation.name`: `SELECT` / `INSERT` / …
  * `codeintel.db.query.hash`: sha256(sql) (optional, for correlation)
* **No `db.statement` / `db.query.text`** is emitted (so no full SQL text leaves the process).

---

## Patch 4 — Wire the wrapper at the single connection-creation point

**File:** `src/codeintel/storage/backend/duckdb_session.py`

```diff
--- /mnt/data/codeintel_orig/src/codeintel/storage/backend/duckdb_session.py	2025-12-20 06:03:35.000000000 +0000
+++ /mnt/data/codeintel_repo/src/codeintel/storage/backend/duckdb_session.py	2025-12-22 05:46:28.096262991 +0000
@@ -293,19 +293,30 @@
 def _open_primary_connection(
     config: StorageConfig,
     *,
     duckdb_config: DuckDBConnectConfig | None = None,
 ) -> DuckDBConnection:
     cfg = duckdb_config
-    if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
-        con = duckdb.connect(str(Path(":memory:")))
+
+    # If a file-backed DB doesn't exist yet, create it so we can open it normally.
+    if (
+        not config.read_only
+        and config.db_path != Path(":memory:")
+        and not config.db_path.exists()
+    ):
+        con_boot = duckdb.connect(str(Path(":memory:")))
         db_path_str = str(config.db_path).replace("'", "''")
-        con.execute(f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')")
-        con.execute("USE main_db")
-        con.close()
-        if cfg is None:
-            return duckdb.connect(str(config.db_path), read_only=False)
-        return duckdb.connect(str(config.db_path), read_only=False, config=cfg)
+        con_boot.execute(
+            f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')"
+        )
+        con_boot.execute("USE main_db")
+        con_boot.close()
 
     if cfg is None:
-        return duckdb.connect(str(config.db_path), read_only=config.read_only)
-    return duckdb.connect(str(config.db_path), read_only=config.read_only, config=cfg)
+        con = duckdb.connect(str(config.db_path), read_only=config.read_only)
+    else:
+        con = duckdb.connect(str(config.db_path), read_only=config.read_only, config=cfg)
+
+    # Optional OpenTelemetry instrumentation (no-op if OTEL isn't installed / enabled).
+    from codeintel.storage.backend.duckdb_tracing import instrument_duckdb_connection
+
+    return instrument_duckdb_connection(con)
```

---

## Operational knobs (so this stays “best-in-class”)

These are intentionally **boring and low-risk**:

* `CODEINTEL_OTEL_DUCKDB_ENABLED=0` → disable wrapping entirely
* `CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT=0` → always emit DB spans (including bootstrap / migrations)
* `CODEINTEL_OTEL_DUCKDB_CAPTURE_SQL_HASH=0` → drop the hash attribute (keep only low-card summary)

---

## Expected result in traces

When your upstream request span exists (HTTP/MCP/CLI), DB spans will show up as:

* **Span name:** `SELECT core.symbols` (or similar)
* **Attributes:**

  * `db.system.name = duckdb`
  * `db.operation.name = SELECT`
  * `db.query.summary = SELECT core.symbols`
  * `codeintel.db.query.hash = <sha256>`

This is the “dashboard-friendly” grouping key the database semantic conventions are converging on. ([OpenTelemetry][1])

---

If you want the *very next* refinement after this (still low-risk): I can add a tiny “summary normalization” hook so that things like `core.symbols AS s` and `core.symbols s` always summarize identically, and optionally cap the number of target tables included (while still respecting the 255-char limit).

[1]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/ "Semantic conventions for database client spans | OpenTelemetry"

# summary normalization hook #

Below is the “next refinement” you asked for: a **small, low-risk query-summary normalization hook** that makes `core.symbols AS s` and `core.symbols s` summarize identically, **optionally caps the number of target tables**, and **enforces the OpenTelemetry-recommended 255-character truncation** without cutting tokens mid-way. The output is written as a **technical narrative + representative code snippets** you can drop into your existing DuckDB span wrapper.

This aligns with the OpenTelemetry database semantic conventions, where:

* `db.query.summary` is intended to be **low-cardinality** and **safe**, and
* instrumentations that parse queries **SHOULD truncate** the summary to **255 characters** (without truncating within a token). ([OpenTelemetry][1])
  Also, database spans **should use `{db.query.summary}` as the span name** when available. ([OpenTelemetry][1])

---

## What we’re fixing and why

### The problem

If your summary is currently built via a light regex (e.g., grabbing `FROM <token>`), then these two syntaxes often produce **different target tokens**:

* `FROM core.symbols AS s`
* `FROM core.symbols s`

…because the naive extraction might include `AS s` in one form or include `s` in the other.

### The goal

Produce a stable summary like:

* `SELECT core.symbols`

…no matter which alias syntax is used.

### The constraints

1. **Low-cardinality & safe** (no dynamic literals, no user strings/paths). ([OpenTelemetry][1])
2. **Token-safe truncation to 255 chars** (do not slice inside a table name). ([OpenTelemetry][1])
3. **Optional cap** on number of targets (tables) included, to keep dashboards clean.

---

## Recommended design

### Summary generation model

We’ll generate a summary following the “operation + targets” spirit of the OTel spec:

* `SELECT <target1> <target2> ...` ([OpenTelemetry][1])

We’ll do it by:

1. Parsing SQL into an AST (SQLGlot).
2. Extracting table references (`exp.Table`) in a stable order.
3. **Normalizing** table identifiers (strip alias, normalize quoting/format, optionally ignore CTE references).
4. Applying:

   * **target cap** (e.g., keep first 3 distinct targets, then append `...` token), and
   * **255-char token-safe truncation**.

SQLGlot has exactly the primitives you want here: parse SQL → traverse AST and find `Table` nodes. 

---

## Representative implementation

### 1) New module: `codeintel/observability/db_query_summary.py`

```python
# codeintel/observability/db_query_summary.py

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None


@dataclass(frozen=True)
class QuerySummaryConfig:
    # Keep it small by default; make configurable via env if you want.
    max_targets: Optional[int] = 3

    # OTel semantic conventions recommend token-safe truncation at 255 chars.
    # (Do not truncate inside a token/target.)
    max_len: int = 255

    # Dialect used for parsing (DuckDB in your case)
    dialect: str = "duckdb"

    # If table-ish target contains suspicious high-cardinality characters
    # (paths, URLs, etc.), hash it instead of emitting raw.
    hash_suspicious_targets: bool = True


_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9_$.]+$")


def summarize_db_query(sql: str, cfg: QuerySummaryConfig = QuerySummaryConfig()) -> str:
    """
    Generate a stable, low-cardinality db.query.summary for OTel.

    Key behaviors:
      - "core.symbols AS s" and "core.symbols s" normalize to "core.symbols"
      - optional max_targets cap (with "..." token)
      - token-safe truncation to cfg.max_len (default 255)
    """
    sql = (sql or "").strip()
    if not sql:
        return "DB"

    # Fast-path fallback if SQLGlot isn't available for some reason.
    if sqlglot is None:
        return _fallback_summary(sql, cfg)

    try:
        tree = sqlglot.parse_one(sql, read=cfg.dialect)  # SQLGlot parse_one entrypoint
    except Exception:
        return _fallback_summary(sql, cfg)

    operation = _operation_name(tree)
    targets = _extract_table_targets(tree)
    targets = _normalize_targets(targets, cfg=cfg)

    # cap targets (optional)
    tokens: List[str] = [operation]
    tokens += _cap_targets(tokens=targets, max_targets=cfg.max_targets)

    # token-safe 255-char truncation (do not cut inside a token)
    summary = _join_tokens_token_safe(tokens, max_len=cfg.max_len)

    # If we dropped tokens due to length or cap, optionally add "..." if it fits
    # and isn't already present.
    if summary != " ".join(tokens) and not summary.endswith("..."):
        maybe = _join_tokens_token_safe(tokens + ["..."], max_len=cfg.max_len)
        if len(maybe) > len(summary):
            summary = maybe

    return summary


def _operation_name(tree) -> str:
    """
    Best-effort operation extraction from the parsed tree.
    Keeps this stable and readable.
    """
    # Handle WITH wrapper: in SQLGlot, the top node may be a With;
    # the underlying statement is usually in tree.this
    stmt = getattr(tree, "this", None) if tree.__class__.__name__.lower() == "with" else tree
    if stmt is None:
        stmt = tree

    # Use class name as operation label (SELECT/INSERT/UPDATE/DELETE/CREATE/etc).
    op = stmt.__class__.__name__.upper()
    return op


def _extract_table_targets(tree) -> List["exp.Table"]:
    """
    Extract exp.Table nodes in traversal order.
    """
    if exp is None:
        return []
    return list(tree.find_all(exp.Table))


def _normalize_targets(tables: Iterable["exp.Table"], cfg: QuerySummaryConfig) -> List[str]:
    """
    Convert Table AST nodes to canonical identifiers while:
      - stripping aliases (by ignoring alias node entirely)
      - standardizing qualifiers (catalog/schema/table) into a dotted form
      - hashing suspicious/high-cardinality targets (optional)
      - de-duplicating while preserving first-seen order
    """
    seen = set()
    out: List[str] = []

    for t in tables:
        ident = _table_identifier(t)
        if not ident:
            continue

        ident = _sanitize_target(ident, cfg=cfg)

        if ident not in seen:
            seen.add(ident)
            out.append(ident)

    return out


def _table_identifier(t: "exp.Table") -> str:
    """
    Attempt to build a canonical dotted identifier: [catalog].[schema].[table]
    without including any alias.

    SQLGlot generally stores the alias separately, so simply ignoring alias
    removes "AS s" vs "s" differences.
    """
    def _name(node) -> Optional[str]:
        if node is None:
            return None
        # SQLGlot nodes often provide .name for identifiers
        n = getattr(node, "name", None)
        if isinstance(n, str) and n:
            return n
        # Fallback: stringify
        s = str(node).strip()
        return s.strip('"`[]') if s else None

    # Common SQLGlot Table structure:
    #   t.this -> table identifier
    #   t.args.get("db") -> schema/database
    #   t.args.get("catalog") -> catalog (if present)
    table = _name(getattr(t, "this", None))
    db = _name(getattr(t, "args", {}).get("db"))
    catalog = _name(getattr(t, "args", {}).get("catalog"))

    parts = [p for p in (catalog, db, table) if p]
    return ".".join(parts) if parts else ""


def _sanitize_target(target: str, cfg: QuerySummaryConfig) -> str:
    """
    Keep targets low-cardinality and safe.
    If it looks like a path/URL or otherwise “suspicious”, hash it.
    """
    target = target.strip()
    if not target:
        return target

    if not cfg.hash_suspicious_targets:
        return target

    # If it contains typical path/url/punctuation characters, treat as suspicious.
    suspicious = any(ch in target for ch in ["/", "\\", ":", "?", "#", "%", "@"])
    too_long = len(target) > 80
    safe_match = bool(_SAFE_TARGET_RE.match(target))

    if suspicious or too_long or not safe_match:
        return f"h:{_short_hash(target)}"

    return target


def _short_hash(s: str) -> str:
    # short, stable, low-overhead
    return hashlib.blake2s(s.encode("utf-8"), digest_size=6).hexdigest()


def _cap_targets(tokens: List[str], max_targets: Optional[int]) -> List[str]:
    """
    Cap number of targets, appending '...' as a separate token when truncated.
    """
    if max_targets is None or max_targets <= 0:
        return list(tokens)

    if len(tokens) <= max_targets:
        return list(tokens)

    kept = list(tokens[:max_targets])
    kept.append("...")
    return kept


def _join_tokens_token_safe(tokens: List[str], max_len: int) -> str:
    """
    Join tokens with spaces without exceeding max_len.
    Never truncates inside a token.
    """
    if max_len <= 0:
        return ""

    out: List[str] = []
    total = 0

    for tok in tokens:
        tok = (tok or "").strip()
        if not tok:
            continue

        add_len = len(tok) if not out else (1 + len(tok))
        if total + add_len > max_len:
            break

        out.append(tok)
        total += add_len

    return " ".join(out) if out else "DB"


def _fallback_summary(sql: str, cfg: QuerySummaryConfig) -> str:
    """
    Extremely conservative fallback: extract first keyword and first FROM target token.
    Still normalizes 'AS alias' and 'alias' by only taking the first token after FROM.
    """
    # operation
    m = re.match(r"^\s*([A-Za-z]+)", sql)
    op = (m.group(1) if m else "DB").upper()

    # first FROM <token>
    fm = re.search(r"\bFROM\s+([^\s,;()]+)", sql, flags=re.IGNORECASE)
    if not fm:
        return op[: cfg.max_len]

    raw = fm.group(1).strip()
    raw = _sanitize_target(raw, cfg=cfg)
    summary = f"{op} {raw}"
    return summary[: cfg.max_len]
```

**Why this works for alias normalization:** the summary is derived from `exp.Table` nodes, and aliases are stored separately in the AST—so the canonical identifier ignores them. That makes `AS s` vs implicit alias syntax produce the same target. SQLGlot’s ability to parse and traverse `Table` nodes is the core enabler here. 

---

### 2) Integration point: inside your DuckDB span wrapper

Wherever you currently create the DuckDB query span (your single connection/session execute wrapper), add:

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from codeintel.observability.db_query_summary import summarize_db_query, QuerySummaryConfig

_cfg = QuerySummaryConfig(
    dialect="duckdb",
    max_targets=3,   # tweak as desired
    max_len=255,
)

def _instrumented_execute(conn, sql: str, params=None):
    tracer = trace.get_tracer(__name__)

    # Use a generic name first; update after we compute summary
    with tracer.start_as_current_span("duckdb.query", kind=SpanKind.CLIENT) as span:
        summary = summarize_db_query(sql, cfg=_cfg)

        # OTel: db.query.summary is intended as low-cardinality grouping key,
        # and span name SHOULD be {db.query.summary} if available.
        span.set_attribute("db.query.summary", summary)
        span.update_name(summary)  # key dashboard win: spans group by summary :contentReference[oaicite:7]{index=7}

        # If you still emit db.query.text, ensure it is sanitized/opt-in
        # per OTel guidance. :contentReference[oaicite:8]{index=8}

        return conn.execute(sql, params) if params is not None else conn.execute(sql)
```

This directly implements the guidance that span name should be `{db.query.summary}` when present. ([OpenTelemetry][1])

---

## What about the 255-character rule?

The OTel semconv explicitly calls out that if you parse `db.query.summary`, you **should truncate to 255 characters** and ensure truncation **does not occur within an operation name or target**. ([OpenTelemetry][1])

That’s why the join function is “token-safe”: it simply **stops before exceeding** the limit, rather than slicing strings.

---

## Representative tests (pytest)

```python
# tests/test_db_query_summary.py

from codeintel.observability.db_query_summary import summarize_db_query, QuerySummaryConfig

def test_alias_normalization_as_vs_implicit():
    cfg = QuerySummaryConfig(max_targets=3, dialect="duckdb")
    a = "SELECT * FROM core.symbols AS s WHERE s.id = 1"
    b = "SELECT * FROM core.symbols s WHERE s.id = 1"
    assert summarize_db_query(a, cfg=cfg) == summarize_db_query(b, cfg=cfg)

def test_target_cap_appends_ellipsis():
    cfg = QuerySummaryConfig(max_targets=2, dialect="duckdb")
    sql = "SELECT * FROM t1 JOIN t2 ON t1.id=t2.id JOIN t3 ON t2.id=t3.id"
    summary = summarize_db_query(sql, cfg=cfg)
    assert summary.startswith("SELECT ")
    assert summary.count(" ") <= 1 + 2  # op + 2 targets (+ maybe ...)
    assert summary.endswith("...")

def test_token_safe_255_char_truncation():
    cfg = QuerySummaryConfig(max_targets=None, max_len=40, dialect="duckdb")
    # Many targets => will truncate by dropping whole tokens, not slicing
    sql = "SELECT * FROM a JOIN b ON a.x=b.x JOIN c ON b.x=c.x JOIN d ON c.x=d.x"
    summary = summarize_db_query(sql, cfg=cfg)
    assert len(summary) <= 40
    # No partial token chopping: tokens are space-separated; last token is whole.
    assert all(tok for tok in summary.split(" "))

def test_suspicious_targets_are_hashed():
    cfg = QuerySummaryConfig(max_targets=3, dialect="duckdb", hash_suspicious_targets=True)
    sql = "SELECT * FROM 's3://bucket/some/very/long/path/file.parquet'"
    summary = summarize_db_query(sql, cfg=cfg)
    assert "h:" in summary  # hashed token instead of raw path
```

---

## Notes on “best-in-class” observability + safety

* The database semconv explicitly warns that **non-parameterized query text** should **not** be collected by default unless sanitized/redacted. ([OpenTelemetry][2])
  Your earlier redaction/hashing patch was exactly the right direction. This refinement keeps `db.query.summary` useful while staying low-risk.
* The semconv also says `db.query.summary` **should not contain dynamic or sensitive data** and is meant as a **grouping key**. ([OpenTelemetry][1])
  That’s why the “hash suspicious targets” guard is included—DuckDB can reference files/URLs in `FROM`, which would otherwise explode cardinality.

---

## Practical defaults I’d use in CodeIntel

* `max_targets = 3` (keeps dashboards tidy; still gives context)
* `hash_suspicious_targets = True` (prevents accidental leakage/cardinality blowups)
* `max_len = 255` (matches OTel guidance) ([OpenTelemetry][1])

---

If you want the **next** refinement after this: we can make the summary **multi-operation aware** (e.g., `INSERT shipping_details SELECT orders` in the exact semconv example style) while still applying alias normalization + caps. The semantic conventions explicitly give that format guidance. ([OpenTelemetry][1])

[1]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/ "Semantic conventions for database client spans | OpenTelemetry"
[2]: https://opentelemetry.io/docs/specs/semconv/database/sql/ "Semantic conventions for SQL databases client operations | OpenTelemetry"


# multi-operation aware db.query.summary generator #

Below is a **semconv-aligned**, **multi-operation-aware** `db.query.summary` generator that preserves your earlier refinements (**alias normalization**, **caps**, **255-char truncation**) and produces the exact style you called out:
`INSERT shipping_details SELECT orders`.

This matches OpenTelemetry’s database span guidance: preserve operations + targets **in-order**, format as `{operation1} {target1} {operation2} {target2}…`, and **truncate to 255 chars without cutting inside an operation/target**. ([OpenTelemetry][1])

Also, for “best-in-class” correctness: when a query involves **multiple operations or multiple targets**, the semantic conventions explicitly warn against populating `db.operation.name` / `db.collection.name` from parsing (ambiguity), and recommend leaning on `db.query.summary` for naming/grouping. ([GitHub][2])

---

## What changes in this refinement

### 1) Multi-operation aware summary (INSERT…SELECT, CREATE…AS SELECT, etc.)

Instead of summarizing “just the first keyword”, we build an **operation/target sequence** from the parsed query:

* **INSERT … SELECT … FROM orders** → `INSERT shipping_details SELECT orders` ([OpenTelemetry][1])
* **Anonymous/derived table** → `SELECT SELECT orders customers` ([OpenTelemetry][1])

### 2) Alias normalization remains intact

We **ignore aliases entirely** (so `core.symbols s` and `core.symbols AS s` summarize identically).

### 3) Caps + 255-char truncation at token boundaries

* Cap targets (e.g., 5 tables per operation) to keep cardinality low.
* Truncate to **255 chars** without splitting a token, as required. ([OpenTelemetry][1])

### 4) Uses SQLGlot AST traversal (your existing dependency)

SQLGlot supports `parse_one()` and `find_all(exp.Table)` for table discovery and AST traversal. 

(If you want the reference handy: )

---

## Representative implementation

### `observability/db_query_summary.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class QuerySummaryConfig:
    dialect: str = "duckdb"

    # Semconv guidance: truncate to 255 chars when parsing to set db.query.summary
    # and do not truncate inside operation or target tokens.
    max_len: int = 255

    # Keep it low-cardinality and readable.
    max_targets_per_operation: int = 5

    # If we drop tokens due to caps/truncation, optionally add a constant marker.
    add_ellipsis_if_truncated: bool = True

    # Treat CTE references as "anonymous SELECT" targets (low-cardinality),
    # and expand them to their underlying base tables where possible.
    expand_ctes: bool = True


@dataclass(frozen=True)
class QuerySummary:
    summary: str
    operations: Tuple[str, ...]   # in encountered order
    targets: Tuple[str, ...]      # in encountered order (after normalization/caps)


def build_db_query_summary(sql: str, cfg: QuerySummaryConfig = QuerySummaryConfig()) -> Optional[QuerySummary]:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return None

    try:
        tree = sqlglot.parse_one(sql, read=cfg.dialect)
    except Exception:
        # Fail open: summary is optional.
        return None

    cte_map = _collect_ctes(tree) if cfg.expand_ctes else {}

    tokens: List[str] = []
    ops: List[str] = []
    targets: List[str] = []

    _append_op_target_sequence(
        node=tree,
        out_tokens=tokens,
        out_ops=ops,
        out_targets=targets,
        cfg=cfg,
        cte_map=cte_map,
        cte_stack=set(),
    )

    if not tokens:
        return None

    # Final semconv truncation at token boundaries
    summary = _truncate_tokens(tokens, max_len=cfg.max_len, add_ellipsis=cfg.add_ellipsis_if_truncated)
    if not summary:
        return None

    return QuerySummary(summary=summary, operations=tuple(ops), targets=tuple(targets))


# ---------------------------
# AST extraction
# ---------------------------

def _append_op_target_sequence(
    *,
    node: exp.Expression,
    out_tokens: List[str],
    out_ops: List[str],
    out_targets: List[str],
    cfg: QuerySummaryConfig,
    cte_map: Dict[str, exp.Expression],
    cte_stack: Set[str],
) -> None:
    """
    Builds the semconv sequence:
        {operation1} {target1} {operation2} {target2} {target3} ...

    Key idea: recurse into nested query-producing expressions so we can emit
    INSERT ... SELECT ..., CREATE TABLE ... SELECT ..., etc.
    """
    op = _operation_name(node)
    if op is None:
        # If we can't classify, do a best-effort: try to find a SELECT inside.
        inner = _first_query_child(node)
        if inner is not None:
            _append_op_target_sequence(
                node=inner,
                out_tokens=out_tokens,
                out_ops=out_ops,
                out_targets=out_targets,
                cfg=cfg,
                cte_map=cte_map,
                cte_stack=cte_stack,
            )
        return

    out_tokens.append(op)
    out_ops.append(op)

    # Operation "primary target" (INSERT target table, UPDATE table, CREATE object, etc.)
    primary_target = _primary_target(node)
    if primary_target:
        out_tokens.append(primary_target)
        out_targets.append(primary_target)

    # SELECT-like: add FROM/JOIN targets (tables, derived selects, CTE expansions)
    if _is_select_like(node):
        select_targets = _select_targets(node, cfg=cfg, cte_map=cte_map, cte_stack=cte_stack)
        for t in select_targets:
            out_tokens.append(t)
            out_targets.append(t)

    # Multi-operation: recurse into nested query child (INSERT .. <query>, CREATE .. AS <query>, etc.)
    nested_query = _nested_query_child(node)
    if nested_query is not None:
        _append_op_target_sequence(
            node=nested_query,
            out_tokens=out_tokens,
            out_ops=out_ops,
            out_targets=out_targets,
            cfg=cfg,
            cte_map=cte_map,
            cte_stack=cte_stack,
        )


def _operation_name(node: exp.Expression) -> Optional[str]:
    """
    Returns the operation token. Keep it low-cardinality and semconv-friendly.
    """
    key = getattr(node, "key", "") or ""

    if isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return "SELECT"

    if key == "insert" or isinstance(node, exp.Insert):
        return "INSERT"
    if key == "update" or isinstance(node, exp.Update):
        return "UPDATE"
    if key == "delete" or isinstance(node, exp.Delete):
        return "DELETE"
    if key == "merge" or isinstance(node, exp.Merge):
        return "MERGE"

    if key == "create" or isinstance(node, exp.Create):
        # Prefer a 2-word operation (still low cardinality) for readability.
        kind = node.args.get("kind")
        kind_str = str(kind).strip().upper() if kind else "TABLE"
        return f"CREATE {kind_str}".strip()

    if key == "drop" or isinstance(node, exp.Drop):
        kind = node.args.get("kind")
        kind_str = str(kind).strip().upper() if kind else "TABLE"
        return f"DROP {kind_str}".strip()

    if key == "command" or isinstance(node, exp.Command):
        # Examples: SET, PRAGMA. Try to use the command name if present.
        name = node.args.get("this")
        if isinstance(name, exp.Expression):
            name = name.name if hasattr(name, "name") else name.sql()
        return str(name).strip().upper() if name else "COMMAND"

    # Add more if you want (CALL/EXECUTE/etc). Otherwise return None.
    return None


def _primary_target(node: exp.Expression) -> Optional[str]:
    """
    Primary target of the operation (e.g. INSERT target table, CREATE target table/view).
    For SELECT, we do NOT set a primary target here (targets are handled by _select_targets).
    """
    if _is_select_like(node):
        return None

    # Usually the primary object is under "this"
    candidate = node.args.get("this")
    table = candidate if isinstance(candidate, exp.Table) else (candidate.find(exp.Table) if isinstance(candidate, exp.Expression) else None)
    if isinstance(table, exp.Table):
        return _format_table_identifier(table)

    return None


def _nested_query_child(node: exp.Expression) -> Optional[exp.Expression]:
    """
    Finds a nested query child that represents a secondary operation:
      INSERT INTO t SELECT ...
      CREATE TABLE t AS SELECT ...
    """
    # Common arg key in sqlglot for INSERT/CREATE is "expression"
    child = node.args.get("expression")
    if isinstance(child, exp.Expression) and _is_select_like(child):
        return child

    # Some nodes wrap the query differently; fallback:
    inner = _first_query_child(node)
    return inner


def _first_query_child(node: exp.Expression) -> Optional[exp.Expression]:
    """
    Best-effort: walk immediate children and return first SELECT-like child.
    """
    for v in node.args.values():
        if isinstance(v, exp.Expression) and _is_select_like(v):
            return v
        if isinstance(v, list):
            for item in v:
                if isinstance(item, exp.Expression) and _is_select_like(item):
                    return item
    return None


def _is_select_like(node: exp.Expression) -> bool:
    return isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)) or getattr(node, "key", "") in {
        "select", "union", "intersect", "except"
    }


def _select_targets(
    node: exp.Expression,
    *,
    cfg: QuerySummaryConfig,
    cte_map: Dict[str, exp.Expression],
    cte_stack: Set[str],
) -> List[str]:
    """
    Returns target tokens for a SELECT-like operation.
    Includes:
      - base tables (no aliases)
      - derived tables as the literal token "SELECT" (anonymous table), then their base tables
      - CTE references treated similarly (optional), to keep it low-cardinality
    """
    # Identify CTE names from the current node's WITH clause if present.
    cte_names = set(cte_map.keys())

    # Extract FROM/JOIN sources (best-effort, version-tolerant).
    sources = _query_sources_in_order(node)

    out: List[str] = []
    seen: Set[str] = set()

    def emit(token: str) -> None:
        if token and token not in seen:
            out.append(token)
            seen.add(token)

    for src in sources:
        # Case 1: plain table
        if isinstance(src, exp.Table):
            name = _format_table_identifier(src)
            # CTE reference: treat as anonymous select + expand
            if cfg.expand_ctes and src.name in cte_names and src.name not in cte_stack:
                emit("SELECT")
                cte_stack.add(src.name)
                emit_many(_select_targets(cte_map[src.name], cfg=cfg, cte_map=cte_map, cte_stack=cte_stack))
                cte_stack.remove(src.name)
            else:
                emit(name)
            continue

        # Case 2: derived table / subquery (JOIN (SELECT...) t, FROM (SELECT...) t)
        inner_query = _unwrap_to_query(src)
        if inner_query is not None and _is_select_like(inner_query):
            emit("SELECT")
            emit_many(_select_targets(inner_query, cfg=cfg, cte_map=cte_map, cte_stack=cte_stack))
            continue

    # Cap targets to keep summary low-cardinality/readable.
    if len(out) > cfg.max_targets_per_operation:
        out = out[: cfg.max_targets_per_operation]

    return out


def emit_many(tokens: Sequence[str]) -> None:
    # Helper for inner closure above (Python scoping convenience).
    raise RuntimeError("This should be shadowed by the closure in _select_targets")


def _query_sources_in_order(node: exp.Expression) -> List[exp.Expression]:
    """
    Extract the "top-level" FROM + JOIN sources in a predictable order.
    This is intentionally conservative; it doesn’t try to include tables used only in subexpressions.
    """
    out: List[exp.Expression] = []

    from_ = node.args.get("from")
    if isinstance(from_, exp.Expression):
        out.extend(_from_expressions(from_))
    elif isinstance(from_, list):
        for x in from_:
            if isinstance(x, exp.Expression):
                out.append(x)

    joins = node.args.get("joins") or []
    if isinstance(joins, list):
        for j in joins:
            if not isinstance(j, exp.Expression):
                continue
            j_this = getattr(j, "this", None) or j.args.get("this")
            if isinstance(j_this, exp.Expression):
                out.append(j_this)

    return out


def _from_expressions(from_: exp.Expression) -> List[exp.Expression]:
    if hasattr(from_, "expressions") and isinstance(from_.expressions, list):
        return [x for x in from_.expressions if isinstance(x, exp.Expression)]

    exprs = from_.args.get("expressions")
    if isinstance(exprs, list):
        return [x for x in exprs if isinstance(x, exp.Expression)]

    this = getattr(from_, "this", None) or from_.args.get("this")
    if isinstance(this, exp.Expression):
        return [this]

    # Sometimes sqlglot may put a Table directly
    if isinstance(from_, exp.Table):
        return [from_]

    return []


def _unwrap_to_query(node: exp.Expression) -> Optional[exp.Expression]:
    """
    If node is a subquery/derived-table wrapper, return its inner query expression.
    """
    # Common wrapper nodes have "this" pointing at the query
    this = getattr(node, "this", None) or node.args.get("this")
    if isinstance(this, exp.Expression) and _is_select_like(this):
        return this

    # Otherwise, see if there's a query-like child
    return _first_query_child(node)


def _collect_ctes(tree: exp.Expression) -> Dict[str, exp.Expression]:
    """
    Build a mapping: cte_name -> cte_query_expression

    sqlglot itself uses a pattern like:
      with_ = expression.args.get("with") or exp.With()
      cte_names = {cte.alias_or_name for cte in with_.expressions}
    (example from sqlglot optimizer docs).
    """
    with_ = tree.args.get("with")
    if not isinstance(with_, exp.Expression):
        return {}

    ctes = getattr(with_, "expressions", None) or with_.args.get("expressions") or []
    if not isinstance(ctes, list):
        return {}

    out: Dict[str, exp.Expression] = {}
    for cte in ctes:
        if not isinstance(cte, exp.Expression):
            continue
        name = getattr(cte, "alias_or_name", None)
        if not name:
            # fallback: look for a TableAlias and read its identifier text
            ta = cte.find(exp.TableAlias)
            name = ta.text("this") if ta is not None else None

        inner = getattr(cte, "this", None) or cte.args.get("this")
        if isinstance(name, str) and isinstance(inner, exp.Expression):
            out[name] = inner
    return out


def _format_table_identifier(table: exp.Table) -> str:
    """
    Alias-normalized table token: use catalog.db.table if present, ignore aliases.
    """
    parts: List[str] = []

    for key in ("catalog", "db", "this"):
        v = table.args.get(key)
        if v is None:
            continue
        parts.append(_format_identifier_piece(v))

    # Fallback: name-only
    if not parts and hasattr(table, "name"):
        return str(table.name)

    return ".".join([p for p in parts if p])


def _format_identifier_piece(v: object) -> str:
    """
    Conservative quoting normalization:
      - If sqlglot Identifier says quoted=True, emit with double quotes.
      - Otherwise emit as-is (best effort).
    """
    if isinstance(v, exp.Identifier):
        raw = v.this
        if v.args.get("quoted"):
            return f"\"{str(raw).replace('\"', '\"\"')}\""
        return str(raw)

    if isinstance(v, exp.Expression):
        # Attempt to recover identifier text
        if hasattr(v, "this"):
            return str(getattr(v, "this"))
        return v.sql()

    return str(v)


# ---------------------------
# Truncation
# ---------------------------

def _truncate_tokens(tokens: Sequence[str], *, max_len: int, add_ellipsis: bool) -> str:
    """
    Join tokens with spaces, truncating to max_len without splitting tokens
    (semconv requirement for db.query.summary parsing).
    """
    if not tokens:
        return ""

    out: List[str] = []
    length = 0
    truncated = False

    for tok in tokens:
        tok = " ".join(tok.split())  # normalize internal spaces if any
        if not tok:
            continue

        # +1 for a joining space if out already has content
        projected = length + (1 if out else 0) + len(tok)
        if projected > max_len:
            truncated = True
            break

        out.append(tok)
        length = projected

    if truncated and add_ellipsis:
        # Add a constant ellipsis token only if it fits.
        ell = "…"
        projected = length + (1 if out else 0) + len(ell)
        if projected <= max_len:
            out.append(ell)

    return " ".join(out)
```

**Notes on the above:**

* The “multi-operation” behavior is driven by `_nested_query_child()` which recurses into the query-producing child so `INSERT … SELECT …` becomes `INSERT <target> SELECT <sources>`. This is exactly the semconv example. ([OpenTelemetry][1])
* Alias normalization is achieved by `_format_table_identifier()` ignoring aliases and emitting only catalog/db/this.
* The 255-char truncation rule is applied in `_truncate_tokens()` at token boundaries. ([OpenTelemetry][1])
* SQLGlot parsing + AST traversal is the intended mechanism (`parse_one`, `find_all`, etc.). 

---

## Integration snippet (DuckDB span creation point)

Where you create the DuckDB span (your single centralized “execute” wrapper), do:

```python
qs = build_db_query_summary(sql, cfg=QuerySummaryConfig(dialect="duckdb"))

if qs:
    span.set_attribute("db.query.summary", qs.summary)

    # Best-in-class: only set db.operation.name / db.collection.name
    # when unambiguous (single operation / single collection).
    # Otherwise rely on db.query.summary as the grouping/span-name key.
    if len(set(qs.operations)) == 1:
        span.set_attribute("db.operation.name", qs.operations[0])

    # Only treat as a single collection if there’s exactly one “real” table target
    # and no anonymous SELECT targets.
    material_targets = [t for t in qs.targets if t != "SELECT" and t != "…"]
    if len(set(material_targets)) == 1:
        span.set_attribute("db.collection.name", material_targets[0])
```

This aligns with the semantic conventions’ intent that `db.query.summary` is the grouping key for complex queries, and avoids ambiguous `db.operation.name` / `db.collection.name` for multi-operation/multi-target queries. ([GitHub][2])

---

## Minimal tests you should add (pytest-style)

```python
def test_insert_select_semconv_example():
    sql = """
    INSERT INTO shipping_details (order_id, address)
    SELECT order_id, address
    FROM orders
    WHERE order_id = ?
    """
    qs = build_db_query_summary(sql)
    assert qs and qs.summary == "INSERT shipping_details SELECT orders"


def test_multi_table_select():
    sql = "SELECT * FROM songs, artists WHERE songs.artist_id = artists.id"
    qs = build_db_query_summary(sql)
    assert qs and qs.summary == "SELECT songs artists"


def test_derived_table_is_anonymous_select():
    sql = """
    SELECT order_date
    FROM (
      SELECT *
      FROM orders o
      JOIN customers c ON o.customer_id = c.customer_id
    )
    """
    qs = build_db_query_summary(sql)
    assert qs and qs.summary.startswith("SELECT SELECT")
    assert "orders" in qs.summary
    assert "customers" in qs.summary


def test_alias_normalization_same_summary():
    q1 = "SELECT * FROM core.symbols s JOIN core.refs r ON s.id = r.id"
    q2 = "SELECT * FROM core.symbols AS s JOIN core.refs AS r ON s.id = r.id"
    assert build_db_query_summary(q1).summary == build_db_query_summary(q2).summary
```

Those tests map directly onto the semconv examples (including `INSERT … SELECT …`) and your alias-normalization requirement. ([OpenTelemetry][1])

---

## Optional “extra polish” you can add right after this

1. **Make `db.query.summary` the span name for DB spans** (when you have it), since it’s explicitly intended as a span name / metrics attribute in practice. ([GitHub][2])
2. Add a tiny “dialect selection” hook: use `db.system.name` to choose SQLGlot dialect (`duckdb`, `postgres`, etc.) so the same code path works across engines.

If you want, I can also provide a **drop-in “SpanAttributeBuilder” helper** that takes `(sql, db_system_name, db_namespace, …)` and returns a dict of semconv-perfect attributes (including the “only set db.operation.name when unambiguous” rule), so every call site stays 3–5 lines long.

[1]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/ "Semantic conventions for database client spans | OpenTelemetry"
[2]: https://github.com/open-telemetry/semantic-conventions/releases "Releases · open-telemetry/semantic-conventions · GitHub"




# drop-in “SpanAttributeBuilder” helper #

Below is a **repo-ready, “extra polish” refinement** that does three things:

1. **Makes DB span names equal to `db.query.summary`** (when you have it), exactly as the database span naming spec recommends. ([OpenTelemetry][1])
2. Adds a **dialect selection hook**: uses `db.system.name` to pick the right SQLGlot dialect so one codepath works across engines. ([OpenTelemetry][2])
3. Introduces a **drop‑in `SpanAttributeBuilder`** so each callsite stays ~3–5 lines, while the “semconv correctness” is centralized.

Because I can’t directly open your ZIP in this environment, the file paths are **representative**—but the design is intended to drop cleanly into the kind of centralized DB/session layer you’ve been building.

---

## Why this change is “best‑in‑class” for observability

### Span name: use `db.query.summary` (not `SELECT`, not raw SQL)

The current database span naming guidance is explicit:

* Span name **SHOULD be `{db.query.summary}`** if available. ([OpenTelemetry][1])
* `db.query.summary` is intended to be a **low-cardinality grouping key** and is “intended to provide a … grouping key that can be used as a span name or a metric attribute in common cases.” ([OpenTelemetry][1])
* When `db.query.summary` is generated from parsing, it **SHOULD be truncated to 255 characters**, and truncation should not cut through an operation/target token. ([OpenTelemetry][1])

This is exactly what we implement below.

### Don’t “invent” `db.operation.name` from SQL text

The same spec warns that operation name **should not be extracted from** `db.query.text` in systems where multi-operation statements are possible (SQL absolutely is). ([OpenTelemetry][1])

So the builder below:

* **never derives `db.operation.name` from SQL**
* only sets it if your higher-level API already knows it (e.g., an ORM call or explicit operation hook)

---

## Implementation overview

You add one small module (or package):

* `observability/db_span_attributes.py`

  * `SqlDialectResolver` (db.system.name → sqlglot dialect)
  * `QuerySummaryGenerator` (SQL → `db.query.summary`)
  * `SpanAttributeBuilder` (returns `{span_name, attrs}`)

Then, in your **single DB entry point** (e.g., `DuckDBSession.execute()` / pool wrapper), you do:

```python
spec = DB_SPANS.build(sql, db_system_name="duckdb", db_namespace=self.db_namespace)
with tracer.start_as_current_span(spec.name, kind=SpanKind.CLIENT, attributes=spec.attributes):
    return self._conn.execute(sql)
```

That’s it at callsites.

---

## 1) New module: `SpanAttributeBuilder` + dialect hook + summary generation

### `codeintel/observability/db_span_attributes.py` (representative)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


_DB_QUERY_SUMMARY_MAX_LEN = 255  # per semconv guidance :contentReference[oaicite:6]{index=6}
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpanSpec:
    """Return value used by callsites to start spans in 3–5 lines."""
    name: str
    attributes: dict[str, Any]


class SqlDialectResolver:
    """
    Tiny “dialect selection hook”:
    Use db.system.name (semconv) to choose SQLGlot dialect.
    """

    # db.system.name well-known values for SQL include mysql/postgresql/sqlite/etc. :contentReference[oaicite:7]{index=7}
    _MAP: Mapping[str, str] = {
        "duckdb": "duckdb",
        "postgresql": "postgres",
        "cockroachdb": "postgres",
        "mysql": "mysql",
        "mariadb": "mysql",
        "sqlite": "sqlite",
        # Optional extras you may care about:
        "microsoft.sql_server": "tsql",
        "oracle.db": "oracle",
        "trino": "trino",
        # If you see "other_sql", you can omit dialect to let sqlglot guess.
        # "other_sql": <no entry>
    }

    def resolve(self, db_system_name: str | None) -> str | None:
        if not db_system_name:
            return None
        return self._MAP.get(db_system_name)


class QuerySummaryGenerator:
    """
    Generates db.query.summary following the semconv format:
      {operation1} {target1} {operation2} {target2} ...
    and truncates safely to 255 chars without splitting tokens. :contentReference[oaicite:8]{index=8}
    """

    def __init__(
        self,
        *,
        dialect_resolver: SqlDialectResolver | None = None,
        max_len: int = _DB_QUERY_SUMMARY_MAX_LEN,
        max_targets_per_op: int = 6,
    ) -> None:
        self._dialects = dialect_resolver or SqlDialectResolver()
        self._max_len = max_len
        self._max_targets_per_op = max_targets_per_op

    def generate(self, sql: str, *, db_system_name: str | None) -> str | None:
        sql = (sql or "").strip()
        if not sql:
            return None

        dialect = self._dialects.resolve(db_system_name)

        try:
            # parse can return multiple statements (semicolon-separated)
            stmts = sqlglot.parse(sql, read=dialect) if dialect else sqlglot.parse(sql)
        except ParseError:
            return None

        tokens: list[str] = []
        for stmt in stmts:
            tokens.extend(self._tokens_for_statement(stmt, dialect=dialect))

        return self._join_and_truncate(tokens)

    def _tokens_for_statement(self, node: exp.Expression, *, dialect: str | None) -> list[str]:
        # INSERT ... SELECT ... => "INSERT <table> SELECT <table>" example style :contentReference[oaicite:9]{index=9}
        if isinstance(node, exp.Insert):
            toks = ["INSERT"]
            target = self._render_table_ref(getattr(node, "this", None), dialect=dialect)
            if target:
                toks.append(target)

            # node.expression may be SELECT (INSERT INTO t SELECT ...)
            sub = getattr(node, "expression", None)
            if isinstance(sub, exp.Expression):
                toks.extend(self._tokens_for_statement(sub, dialect=dialect))
            return toks

        if isinstance(node, exp.Select):
            toks = ["SELECT"]
            toks.extend(self._select_targets(node, dialect=dialect))
            return toks

        if isinstance(node, exp.Update):
            toks = ["UPDATE"]
            target = self._render_table_ref(getattr(node, "this", None), dialect=dialect)
            if target:
                toks.append(target)
            return toks

        if isinstance(node, exp.Delete):
            toks = ["DELETE"]
            target = self._render_table_ref(getattr(node, "this", None), dialect=dialect)
            if target:
                toks.append(target)
            return toks

        if isinstance(node, exp.Create):
            toks = ["CREATE"]
            target = self._render_table_ref(getattr(node, "this", None), dialect=dialect)
            if target:
                toks.append(target)
            return toks

        # Fallback: use SQLGlot's node key as an operation-like token
        return [str(getattr(node, "key", "STATEMENT")).upper()]

    def _select_targets(self, sel: exp.Select, *, dialect: str | None) -> list[str]:
        """
        Extract targets in a way that matches semconv intent:
        - tables in FROM/JOIN (in appearance order)
        - if a FROM source is a subquery, include nested "SELECT ..." tokens (see semconv nested example) :contentReference[oaicite:10]{index=10}
        """
        out: list[str] = []

        def consume_source(src: exp.Expression) -> None:
            # FROM (SELECT ... ) => include nested op+targets as tokens
            if isinstance(src, exp.Subquery):
                inner = getattr(src, "this", None)
                if isinstance(inner, exp.Expression):
                    out.extend(self._tokens_for_statement(inner, dialect=dialect))
                return

            # Direct table reference
            if isinstance(src, exp.Table):
                ref = self._render_table_ref(src, dialect=dialect)
                if ref:
                    out.append(ref)
                return

            # Some dialects/queries wrap sources; try to unwrap `.this`
            inner = getattr(src, "this", None)
            if isinstance(inner, exp.Expression) and inner is not src:
                consume_source(inner)

        from_clause = sel.args.get("from")
        if from_clause is not None:
            for src in from_clause.expressions:
                consume_source(src)

        for j in sel.args.get("joins") or []:
            join_src = getattr(j, "this", None)
            if isinstance(join_src, exp.Expression):
                consume_source(join_src)

        # De-dup while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for t in out:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return deduped[: self._max_targets_per_op]

    def _render_table_ref(self, node: Any, *, dialect: str | None) -> str | None:
        """
        Render schema-qualified names when present; avoid aliases.
        We intentionally keep it simple & stable for low-cardinality summaries.
        """
        if not isinstance(node, exp.Table):
            return None

        parts: list[str] = []
        for key in ("catalog", "db", "this"):
            v = node.args.get(key)
            if v is None:
                continue
            if isinstance(v, exp.Expression):
                parts.append(v.sql(dialect=dialect) if dialect else v.sql())
            else:
                parts.append(str(v))

        ref = ".".join(parts).strip()
        return ref or None

    def _join_and_truncate(self, tokens: list[str]) -> str | None:
        """
        Truncate to 255 chars without cutting through tokens (operation/target). :contentReference[oaicite:11]{index=11}
        """
        cleaned: list[str] = []
        for t in tokens:
            t = _SPACE_RE.sub(" ", (t or "").strip())
            if t:
                cleaned.append(t)

        if not cleaned:
            return None

        acc = ""
        for t in cleaned:
            candidate = t if not acc else f"{acc} {t}"
            if len(candidate) > self._max_len:
                break
            acc = candidate

        return acc or None


class SpanAttributeBuilder:
    """
    Semconv-perfect(ish) DB span attribute builder:
    - Always sets db.system.name
    - Sets db.namespace if provided
    - Generates db.query.summary (low-cardinality) and uses it as span name
    - Does NOT derive db.operation.name from SQL text (pass it explicitly if known)
    """

    def __init__(
        self,
        *,
        summary_generator: QuerySummaryGenerator | None = None,
        emit_legacy_db_statement_keys: bool = False,
    ) -> None:
        self._summary = summary_generator or QuerySummaryGenerator()
        self._emit_legacy = emit_legacy_db_statement_keys

    def build(
        self,
        sql: str | None,
        *,
        db_system_name: str,
        db_namespace: str | None = None,
        # optional “hooks” for higher-level APIs:
        db_operation_name: str | None = None,
        target_override_for_span_name: str | None = None,
    ) -> SpanSpec:
        attrs: dict[str, Any] = {
            "db.system.name": db_system_name,  # required by semconv :contentReference[oaicite:12]{index=12}
        }
        if db_namespace:
            attrs["db.namespace"] = db_namespace

        if db_operation_name:
            # Only set when you *already know* it's unambiguous (don’t parse SQL for this). :contentReference[oaicite:13]{index=13}
            attrs["db.operation.name"] = _SPACE_RE.sub(" ", db_operation_name.strip())

        summary = None
        if sql:
            summary = self._summary.generate(sql, db_system_name=db_system_name)
            if summary:
                attrs["db.query.summary"] = summary  # recommended for grouping :contentReference[oaicite:14]{index=14}

        # Span naming per spec:
        # 1) db.query.summary if available :contentReference[oaicite:15]{index=15}
        # 2) else {db.operation.name} {target}
        # 3) else target
        # 4) else db.system.name :contentReference[oaicite:16]{index=16}
        span_name = self._choose_span_name(
            db_system_name=db_system_name,
            db_namespace=db_namespace,
            summary=summary,
            operation=attrs.get("db.operation.name"),
            target_override=target_override_for_span_name,
        )

        # Optional: if you need compatibility with older emitters/backends.
        # (Many contrib instrumentations still emit db.statement/db.system/db.name.)
        if self._emit_legacy:
            attrs.setdefault("db.system", db_system_name)
            if db_namespace:
                attrs.setdefault("db.name", db_namespace)
            # Intentionally NOT emitting db.statement by default; that’s the high-cardinality/sensitive one. :contentReference[oaicite:17]{index=17}

        return SpanSpec(name=span_name, attributes=attrs)

    def _choose_span_name(
        self,
        *,
        db_system_name: str,
        db_namespace: str | None,
        summary: str | None,
        operation: str | None,
        target_override: str | None,
    ) -> str:
        if summary:
            return summary

        target = target_override or db_namespace
        if operation and target:
            return f"{operation} {target}"
        if target:
            return target
        return db_system_name
```

**Notes on semconv alignment:**

* This builder follows the current “DB span name” rules: span name should be `db.query.summary` if present; otherwise fall back to operation+target, then target, then system name. ([OpenTelemetry][1])
* It generates summaries in the **operation/target** format and **safe truncation** pattern. ([OpenTelemetry][1])
* It does **not** derive `db.operation.name` from SQL text (you pass it only when you truly have an unambiguous operation via a higher-level API). ([OpenTelemetry][1])

---

## 2) Integrate in your single DB entry point

Wherever you currently do “execute SQL” centrally (examples: `DuckDBSession`, `DuckDBPool`, `QueryRunner`, etc.), add something like:

### `codeintel/db/duckdb_session.py` (representative)

```python
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from codeintel.observability.db_span_attributes import SpanAttributeBuilder

tracer = trace.get_tracer(__name__)
DB_SPANS = SpanAttributeBuilder(
    emit_legacy_db_statement_keys=False,  # keep stable-only unless you need dup keys
)

class DuckDBSession:
    def __init__(self, conn, *, db_namespace: str = "codeintel", db_system_name: str = "duckdb"):
        self._conn = conn
        self._db_namespace = db_namespace
        self._db_system_name = db_system_name

    def execute(self, sql: str, params=None):
        spec = DB_SPANS.build(
            sql,
            db_system_name=self._db_system_name,
            db_namespace=self._db_namespace,
        )

        # For in-process DuckDB you *may* choose INTERNAL, but CLIENT is still common for “DB call” spans. :contentReference[oaicite:21]{index=21}
        with tracer.start_as_current_span(
            spec.name,
            kind=SpanKind.CLIENT,
            attributes=spec.attributes,
        ) as span:
            try:
                if params is None:
                    return self._conn.execute(sql)
                return self._conn.execute(sql, params)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
```

This is the exact “3–5 lines at callsites” outcome you were aiming for:

* `spec = DB_SPANS.build(...)`
* `start_as_current_span(spec.name, attributes=spec.attributes)`
* execute

---

## 3) If you currently rely on `opentelemetry-instrumentation-dbapi`

You *can* keep it, but there’s an important caveat:

* The DBAPI contrib instrumentation currently names spans as the first SQL token (e.g., `SELECT`) and populates older keys like `db.statement`, `db.system`, `db.name` in its default `_populate_span`. ([OpenTelemetry Python Contrib][3])
* It does not provide a clean “span name callback” for DB spans, so “make span name = db.query.summary” is not a trivial config toggle.

If you want to remain “best-in-class” *and* keep DBAPI contrib, the lowest-friction approach is typically:

* keep DBAPI instrumentation for other drivers,
* but for DuckDB (where you already have a centralized wrapper), prefer the **manual span wrapper** above so you can follow the current DB naming conventions exactly. ([OpenTelemetry][1])

(If you still want a “keep DBAPI but rename spans” plan: the next clean technique is a tiny custom `SpanProcessor` that updates span name at end if `db.query.summary` exists—but that won’t satisfy the “attributes at span creation time” guidance as well as the builder approach. ([OpenTelemetry][1]))

---

## 4) Minimal unit tests you should add (high value)

### `tests/test_db_query_summary.py` (representative)

```python
from codeintel.observability.db_span_attributes import QuerySummaryGenerator

def test_select_single_table():
    g = QuerySummaryGenerator()
    s = g.generate("SELECT * FROM core.symbols WHERE id = 1", db_system_name="duckdb")
    assert s == "SELECT core.symbols"

def test_insert_select_multi_operation():
    g = QuerySummaryGenerator()
    sql = """
    INSERT INTO shipping_details (order_id, address)
    SELECT order_id, address FROM orders WHERE order_id = 1
    """
    s = g.generate(sql, db_system_name="duckdb")
    assert s.startswith("INSERT shipping_details SELECT orders")

def test_truncation_does_not_split_tokens():
    g = QuerySummaryGenerator(max_targets_per_op=100)
    long = "SELECT * FROM " + ", ".join(f"t{i}" for i in range(200))
    s = g.generate(long, db_system_name="duckdb")
    assert len(s) <= 255
    assert not s.endswith("t")  # crude check that we didn’t cut mid-token
```

These tests protect the exact “polish” features: **format + multi-operation + safe truncation**. ([OpenTelemetry][1])

---

## 5) Practical knobs you may want to expose via config

Even if you don’t change behavior now, it’s worth making these values “central config”:

* `DB_QUERY_SUMMARY_MAX_TARGETS` (default 6)
* `DB_QUERY_SUMMARY_MAX_LEN` (255 per spec) ([OpenTelemetry][1])
* `emit_legacy_db_statement_keys` (off by default; on only if you need compatibility with older attribute keys)

---

## Internal references used from your provided library docs

* SQLGlot AST traversal / parsing patterns that enable summary extraction: 
* (Optional) Ibis ↔ SQLGlot integration surface (useful if you later generate SQL via Ibis and still want consistent summaries): 

---

If you want the *next* refinement after this, I can show a small extension to the builder that optionally emits `db.query.text` **only when parameterized** (or only after SQLGlot literal-redaction), aligned with the sanitization guidance—without increasing cardinality by default. ([OpenTelemetry][1])

[1]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/ "Semantic conventions for database client spans | OpenTelemetry"
[2]: https://opentelemetry.io/docs/specs/semconv/database/sql/ "Semantic conventions for SQL databases client operations | OpenTelemetry"
[3]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/_modules/opentelemetry/instrumentation/dbapi.html "opentelemetry.instrumentation.dbapi — OpenTelemetry Python Contrib  documentation"


# extend spandattributebuilder for parametized query or literal redacted via SQLGlot #

Below is the “next refinement” in the exact spirit you described: **extend your existing SpanAttributeBuilder so `db.query.text` is emitted only when (a) the query is parameterized, or (b) the SQL has been literal‑redacted via SQLGlot**, and keep this **opt‑in by default** so you don’t increase cardinality unless you explicitly choose to.

This aligns directly with the OpenTelemetry database semantic conventions:

* **Non‑parameterized `db.query.text` SHOULD NOT be collected by default** unless you sanitize it by redacting literals. ([OpenTelemetry][1])
* **Sanitization SHOULD replace all literals** (string, numeric, date/time, boolean, interval, binary, hex…) with a placeholder, and the placeholder **SHOULD be `?`** (unless `?` has meaning in that DB). ([OpenTelemetry][2])
* Parameter values themselves are **opt‑in** (they’re not part of “safe by default”). ([OpenTelemetry][1])

---

## Design goals

1. **Best-in-class privacy posture:** never emit raw literals in `db.query.text` unless you explicitly opt in.
2. **No default cardinality increase:** your current “summary-first” approach stays the default; query text is *off* unless configured.
3. **One centralized policy point:** the builder decides; call sites stay tiny.
4. **Dialect-aware + resilient:** choose SQLGlot dialect from `db.system.name`, but fall back safely if parsing fails.

---

## Policy model (recommended)

Add a small enum for query text emission:

* `never` (default): do not emit `db.query.text`
* `parameterized`: emit only when `params is not None` **and** the SQL appears to use placeholders
* `redacted`: emit only **after** SQLGlot redaction succeeds
* `parameterized_or_redacted`: parameterized first; else try redaction
* `always` (debug only): emit raw (generally not recommended)

Even if OTel says parameterized text “should” be collected by default, your explicit requirement is “no cardinality increase by default”, so this makes that a conscious switch. ([OpenTelemetry][1])

---

## Representative implementation

### 1) New module: SQL sanitization using SQLGlot

Create something like `codeintel/observability/sql_text.py` (rename to match your tree):

```python
# codeintel/observability/sql_text.py

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

# --- small helpers ---

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# DuckDB placeholder forms: ?, $1, $param. :contentReference[oaicite:4]{index=4}
_DUCKDB_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(\$[A-Za-z_][A-Za-z0-9_]*)")

# Generic “good enough” placeholder detection for other SQL dialects.
_GENERIC_PLACEHOLDER_RE = re.compile(r"(\?)|(\$[0-9]+)|(:[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class QueryTextSanitizerConfig:
    max_len: int = 4096
    strip_comments: bool = True
    collapse_in_lists: bool = True  # optional cardinality control


def looks_parameterized(sql: str, *, db_system_name: str) -> bool:
    """
    Heuristic: does the SQL contain placeholders typical for the backend?
    For DuckDB specifically: ?, $1, $param. :contentReference[oaicite:5]{index=5}
    """
    if db_system_name.lower() == "duckdb":
        return bool(_DUCKDB_PLACEHOLDER_RE.search(sql))
    return bool(_GENERIC_PLACEHOLDER_RE.search(sql))


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _maybe_strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return sql


def _collapse_in_clause(sql: str) -> str:
    """
    Optional “extra safety”: if you have IN (?, ?, ?, ?) after redaction,
    you can collapse it to IN (?) to prevent extreme length/cost/cardinality.
    This is explicitly called out as a MAY in db span sanitization guidance. :contentReference[oaicite:6]{index=6}
    """
    # Collapse repeated placeholders inside IN ( ... )
    sql = re.sub(r"\bIN\s*\(\s*(\?\s*,\s*){2,}\?\s*\)", "IN (?)", sql, flags=re.IGNORECASE)
    return sql


def redact_sql_literals_with_sqlglot(
    sql: str,
    *,
    dialect: Optional[str],
    cfg: QueryTextSanitizerConfig,
) -> Optional[str]:
    """
    Returns sanitized SQL where literal values are replaced by '?' placeholders.

    OTel guidance: sanitization SHOULD replace all literals with a placeholder
    (commonly '?'). :contentReference[oaicite:7]{index=7}
    """
    if cfg.strip_comments:
        sql = _maybe_strip_comments(sql)

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        # SQLGlot not available; caller can decide to fall back or skip.
        return None

    # Build a placeholder node for literals.
    # SQLGlot has a Placeholder expression type (used for parameter placeholders). :contentReference[oaicite:8]{index=8}
    Placeholder = getattr(exp, "Placeholder", None)

    def placeholder_node():
        if Placeholder is not None:
            return Placeholder()
        # Fallback: still redact, though it may serialize as a string literal.
        Literal = getattr(exp, "Literal", None)
        if Literal is not None:
            return Literal.string("?")
        return None

    def transform(node: exp.Expression) -> exp.Expression:
        # Replace common literal-like nodes. OTel includes many literal categories. :contentReference[oaicite:9]{index=9}
        if isinstance(node, exp.Literal):
            ph = placeholder_node()
            return ph or node

        Boolean = getattr(exp, "Boolean", None)
        if Boolean is not None and isinstance(node, Boolean):
            ph = placeholder_node()
            return ph or node

        Interval = getattr(exp, "Interval", None)
        if Interval is not None and isinstance(node, Interval):
            ph = placeholder_node()
            return ph or node

        # Some dialects represent binary/hex literals distinctly; best-effort:
        HexString = getattr(exp, "HexString", None)
        if HexString is not None and isinstance(node, HexString):
            ph = placeholder_node()
            return ph or node

        return node

    try:
        # SQLGlot supports AST transforms via .transform(...). :contentReference[oaicite:10]{index=10}
        parsed = sqlglot.parse_one(sql, read=dialect) if dialect else sqlglot.parse_one(sql)
        sanitized_expr = parsed.transform(transform)
        sanitized_sql = sanitized_expr.sql(dialect=dialect) if dialect else sanitized_expr.sql()
    except Exception:
        return None

    if cfg.collapse_in_lists:
        sanitized_sql = _collapse_in_clause(sanitized_sql)

    return _truncate(sanitized_sql, cfg.max_len)
```

Key points:

* This uses SQLGlot’s AST transformation capability (`transform`) rather than regex‑only rewriting. 
* It replaces literals with a placeholder, which is exactly what the OTel sanitization guidance recommends. ([OpenTelemetry][2])
* For DuckDB, placeholder syntax includes `?`, `$1`, `$param`, and the Python DB-API supports passing values separately when using those placeholders. ([DuckDB][3])

---

### 2) Extend your SpanAttributeBuilder

Below is a representative “drop-in” style change. Adapt names/paths to your existing builder module.

```python
# codeintel/observability/span_attributes.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .sql_text import (
    QueryTextSanitizerConfig,
    looks_parameterized,
    redact_sql_literals_with_sqlglot,
)

ParamsT = Optional[object]  # could be Sequence[Any] | Mapping[str, Any] | None


class DbQueryTextPolicy(str, Enum):
    NEVER = "never"
    PARAMETERIZED = "parameterized"
    REDACTED = "redacted"
    PARAMETERIZED_OR_REDACTED = "parameterized_or_redacted"
    ALWAYS = "always"  # debug only; not recommended


@dataclass(frozen=True)
class DbQueryTextConfig:
    policy: DbQueryTextPolicy = DbQueryTextPolicy.NEVER  # default = no extra cardinality
    sanitizer: QueryTextSanitizerConfig = QueryTextSanitizerConfig()


@dataclass(frozen=True)
class SpanAttributeBuilder:
    # ... your existing fields (service name, env, etc.) ...
    db_query_text: DbQueryTextConfig = DbQueryTextConfig()

    def _sqlglot_dialect_for_db_system(self, db_system_name: str) -> Optional[str]:
        # you likely already added this “dialect selection” hook in the prior refinement
        # (duckdb, postgres, mysql, …). Keep it centralized.
        key = (db_system_name or "").lower()
        if key in ("duckdb",):
            return "duckdb"
        if key in ("postgresql", "postgres"):
            return "postgres"
        if key in ("mysql",):
            return "mysql"
        if key in ("sqlite", "sqlite3"):
            return "sqlite"
        return None

    def _maybe_db_query_text(
        self,
        *,
        sql: str,
        params: ParamsT,
        db_system_name: str,
    ) -> Optional[str]:
        pol = self.db_query_text.policy

        if pol == DbQueryTextPolicy.NEVER:
            return None

        if pol == DbQueryTextPolicy.ALWAYS:
            # Strongly consider keeping this for local dev only.
            return sql

        # “parameterized” path:
        if pol in (DbQueryTextPolicy.PARAMETERIZED, DbQueryTextPolicy.PARAMETERIZED_OR_REDACTED):
            if params is not None and looks_parameterized(sql, db_system_name=db_system_name):
                # OTel: parameterized query text is generally okay to collect;
                # parameter *values* are opt-in. :contentReference[oaicite:14]{index=14}
                return sql

        # “redacted” path:
        if pol in (DbQueryTextPolicy.REDACTED, DbQueryTextPolicy.PARAMETERIZED_OR_REDACTED):
            dialect = self._sqlglot_dialect_for_db_system(db_system_name)
            redacted = redact_sql_literals_with_sqlglot(
                sql,
                dialect=dialect,
                cfg=self.db_query_text.sanitizer,
            )
            return redacted

        return None

    def build_db_span_attributes(
        self,
        *,
        sql: str,
        params: ParamsT,
        db_system_name: str,
        db_namespace: Optional[str],
        # plus your existing inputs: operation, summary, etc.
        db_operation_name: Optional[str],
        db_query_summary: Optional[str],
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {}

        # Your existing attributes:
        # attrs["db.system.name"] = db_system_name
        # attrs["db.namespace"] = db_namespace
        # attrs["db.operation.name"] = db_operation_name (only if unambiguous)
        # attrs["db.query.summary"] = db_query_summary (low-cardinality)
        #
        # (Not repeating here since you already added them in prior patches.)

        # Add query text ONLY if policy allows:
        qt = self._maybe_db_query_text(sql=sql, params=params, db_system_name=db_system_name)
        if qt:
            # OTel: db.query.text should be sanitized for non-parameterized SQL. :contentReference[oaicite:15]{index=15}
            # Note: this is span attribute only; avoid using it as a metrics dimension by default.
            attrs["db.query.text"] = qt

        return attrs
```

Why this is “best-in-class”:

* It matches OTel’s intended model: **`db.query.summary` for low-cardinality identity**, **`db.query.text` only when safe/sanitized**. ([OpenTelemetry][1])
* It keeps your call sites tiny and your behavior consistent across CLI/HTTP/MCP (since the policy is centralized in the builder).

---

### 3) Call-site example (DuckDB wrapper)

Wherever you create the DB span today (your “single connection-creation point” wrapper), pass `params` through:

```python
# inside your instrumented DuckDB execute wrapper

attrs = span_attr_builder.build_db_span_attributes(
    sql=sql,
    params=params,               # IMPORTANT: enables the “parameterized” safe-path
    db_system_name="duckdb",
    db_namespace=db_path_or_name,
    db_operation_name=op_name,
    db_query_summary=summary,
)

with tracer.start_as_current_span(summary or op_name or "db.query") as span:
    for k, v in attrs.items():
        span.set_attribute(k, v)
    return conn.execute(sql, params) if params is not None else conn.execute(sql)
```

DuckDB’s Python DB-API supports passing values as a second argument when your SQL contains placeholders like `?` or `$1`. ([DuckDB][4])

---

## Suggested default config

In your service defaults:

* `DbQueryTextPolicy.NEVER` (no change in cardinality, safest)
* Provide an environment toggle for troubleshooting:

  * `CODEINTEL_OTEL_DB_QUERY_TEXT=parameterized_or_redacted`
  * Optionally also control `max_len` / `collapse_in_lists`

This way, production remains “summary-only” unless you opt in, but you can flip it on to debug latency hotspots.

---

## Mini test cases (representative)

```python
def test_parameterized_keeps_text_when_enabled(builder):
    builder = builder.with_db_query_text_policy("parameterized")
    sql = "select * from t where id = ?"
    attrs = builder.build_db_span_attributes(
        sql=sql, params=[123], db_system_name="duckdb",
        db_namespace="main", db_operation_name="SELECT", db_query_summary="SELECT t"
    )
    assert attrs["db.query.text"] == sql

def test_non_parameterized_redacts_literals(builder):
    builder = builder.with_db_query_text_policy("redacted")
    sql = "select * from users where email = 'a@b.com' and age > 42"
    attrs = builder.build_db_span_attributes(
        sql=sql, params=None, db_system_name="duckdb",
        db_namespace="main", db_operation_name="SELECT", db_query_summary="SELECT users"
    )
    # shape depends on dialect/generator, but literals should be placeholders:
    assert "a@b.com" not in attrs["db.query.text"]
    assert "42" not in attrs["db.query.text"]
```

---

## Practical notes & pitfalls

* **Redaction isn’t “zero leakage”:** even redacted query text still exposes table/column names and query structure. If that’s sensitive, keep using **`db.query.summary` + hashed SQL** and only enable text in controlled environments.
* **Parsing failures happen:** SQLGlot may fail on some edge syntax; the implementation above simply returns `None` (meaning: “don’t emit query text”) rather than emitting unsafe raw SQL.
* **Do not put `db.query.text` on metrics by default:** span attributes are one thing; metric dimensions can explode cardinality and cost.

---

If you want the *next* incremental refinement after this one (still low-risk), I can show how to add an **opt-in `db.query.parameter.<key>` emitter** for *named* parameters only (and only for a strict allowlist of keys), which stays aligned with the spec’s “values are opt-in” posture. ([OpenTelemetry][1])

[1]: https://opentelemetry.io/docs/specs/semconv/database/sql/?utm_source=chatgpt.com "Semantic conventions for SQL databases client operations"
[2]: https://opentelemetry.io/docs/specs/semconv/database/database-spans/?utm_source=chatgpt.com "Semantic conventions for database client spans"
[3]: https://duckdb.org/docs/stable/sql/query_syntax/prepared_statements.html?utm_source=chatgpt.com "Prepared Statements"
[4]: https://duckdb.org/docs/stable/clients/python/dbapi.html?utm_source=chatgpt.com "Python DB API"


# **opt‑in** emitter for **`db.query.parameter.<key>`** span attributes #

Below is the next “low-risk, best‑in‑class” refinement: an **opt‑in** emitter for **`db.query.parameter.<key>`** span attributes that:

* only emits **named parameters** (Python `Mapping[str, Any]`, e.g. dict)
* only emits keys from a **strict allowlist**
* (optionally) only emits keys that actually appear as placeholders in the SQL text
* stays aligned with the semantic‑conventions posture that **parameter values are opt‑in** (not collected by default). ([OpenTelemetry][1])

I’ll give you:

1. the technical narrative (design + guardrails)
2. a representative “patch set” (diff-style) you can transplant
3. tests you can drop in
4. an optional variant if you’re relying on DBAPI auto-instrumentation

---

## Why this is useful (and how to keep it safe)

### What semantic conventions are trying to achieve

OpenTelemetry’s DB guidance distinguishes between:

* **low-cardinality query identity** (your `db.query.summary`, operation name, db.system, db.namespace)
* **query text** (often sensitive; many teams redact or gate it)
* **parameter values** (explicitly **opt-in**, because they can contain PII/high-cardinality values). ([OpenTelemetry][1])

The conventions define **`db.query.parameter.<key>`** for parameter values, but the intent is: **you only collect these when you *really* mean to**, and you keep it controlled. ([OpenTelemetry][1])

### The “low-risk” posture we implement

To keep this “best-in-class” without surprising privacy/cost outcomes, the implementation below enforces:

1. **Opt-in**: off by default (no allowlist → no emission). ([OpenTelemetry][1])
2. **Named only**: emit only when params are a `Mapping[str, Any]` (dict-like).
3. **Strict allowlist**: emit only for keys you explicitly approve.
4. **Skip batches**: do **not** emit parameter values for batch executions (e.g., `executemany`), aligning with the spec guidance to avoid batch capture. ([OpenTelemetry][1])
5. **Scalar-only by default**: only emit scalar primitives (bool/int/float/str), and cap string length.

For DuckDB specifically: named parameters use `$name` placeholders and a dict passed to `execute()`.

---

## Patch set (representative diffs)

> Notes:
>
> * Replace the file paths with your actual layout (I’m using `codeintel/observability/...` as a reasonable “centralized observability” home).
> * If you already have a “SpanAttributeBuilder” from prior patches, the changes are simply: **add `params`**, call `emit_db_query_parameters()`, and merge the returned dict.

### 1) New module: parameter emitter

```diff
diff --git a/codeintel/observability/db_query_parameters.py b/codeintel/observability/db_query_parameters.py
new file mode 100644
index 0000000..abcd123
--- /dev/null
+++ b/codeintel/observability/db_query_parameters.py
@@ -0,0 +1,230 @@
+from __future__ import annotations
+
+from dataclasses import dataclass
+from typing import Any, Mapping, Optional
+import hashlib
+import re
+
+
+# DuckDB supports named parameters via `$name` and passing a dict to execute(). 
+_DUCKDB_NAMED_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
+
+
+ScalarAttr = str | bool | int | float
+
+
+@dataclass(frozen=True)
+class DBQueryParameterConfig:
+    """
+    Controls emission of db.query.parameter.<key> attributes.
+
+    Safety defaults:
+      - disabled unless allowlist is non-empty AND enabled=True
+      - scalar values only (no lists/tuples/dicts)
+      - string truncation
+    """
+
+    enabled: bool = False
+    allowed_keys: frozenset[str] = frozenset()
+
+    # If True, only emit keys that appear as placeholders in the SQL text.
+    require_key_in_sql: bool = True
+
+    # Value controls
+    max_string_len: int = 80
+    hash_string_values_for_keys: frozenset[str] = frozenset()
+
+    # Batch safety: never emit for executemany / batch-style calls
+    disable_on_batch: bool = True
+
+    def is_effectively_enabled(self) -> bool:
+        return self.enabled and bool(self.allowed_keys)
+
+
+def _truncate(s: str, max_len: int) -> str:
+    if max_len <= 0:
+        return ""
+    if len(s) <= max_len:
+        return s
+    # Use a single ellipsis char to keep things tidy
+    return s[: max(0, max_len - 1)] + "…"
+
+
+def _hash_str(s: str) -> str:
+    # Stable, deterministic, non-reversible (but still high-cardinality if inputs are unique)
+    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
+    return digest[:16]
+
+
+def _coerce_scalar(value: Any, *, max_string_len: int) -> Optional[ScalarAttr]:
+    """
+    Convert a Python value into an OTel attribute-friendly scalar.
+    Returns None if the value should not be emitted.
+    """
+    if value is None:
+        return None
+    if isinstance(value, bool):
+        return value
+    if isinstance(value, int) and not isinstance(value, bool):
+        return value
+    if isinstance(value, float):
+        return value
+    if isinstance(value, str):
+        return _truncate(value, max_string_len)
+
+    # Conservative default: stringify non-scalars is *not* low-risk.
+    # If you want to allow UUID/datetime, consider adding explicit cases here.
+    return None
+
+
+def _extract_named_param_keys(sql: str, *, db_system_name: str) -> set[str]:
+    """
+    Extract named parameter placeholders from SQL.
+
+    For DuckDB, named placeholders look like $name. 
+    """
+    db = (db_system_name or "").lower()
+    if db == "duckdb":
+        return set(_DUCKDB_NAMED_PARAM_RE.findall(sql or ""))
+
+    # Generic fallback: if you later support other engines with named placeholders,
+    # add dialect-specific extractors here (e.g., :name, %(name)s, @name).
+    return set()
+
+
+def emit_db_query_parameters(
+    *,
+    sql: str,
+    params: Any | None,
+    db_system_name: str,
+    config: DBQueryParameterConfig,
+    is_batch: bool = False,
+) -> dict[str, ScalarAttr]:
+    """
+    Returns a dict of attributes:
+      { "db.query.parameter.<key>": <scalar> }
+
+    Constraints:
+      - named parameters only: params must be Mapping[str, Any]
+      - allowlist keys only
+      - optionally require placeholder presence in SQL
+      - skip batches
+    """
+    if not config.is_effectively_enabled():
+        return {}
+
+    if config.disable_on_batch and is_batch:
+        return {}
+
+    if not isinstance(params, Mapping):
+        # Named-only: skip positional params (list/tuple)
+        return {}
+
+    # Ensure keys are strings
+    if not all(isinstance(k, str) for k in params.keys()):
+        return {}
+
+    keys_in_sql: set[str] = set()
+    if config.require_key_in_sql:
+        keys_in_sql = _extract_named_param_keys(sql, db_system_name=db_system_name)
+        if not keys_in_sql:
+            # If we can't find placeholders, do nothing (avoids accidental emission)
+            return {}
+
+    attrs: dict[str, ScalarAttr] = {}
+    for key in config.allowed_keys:
+        if key not in params:
+            continue
+        if config.require_key_in_sql and key not in keys_in_sql:
+            continue
+
+        raw = _coerce_scalar(params[key], max_string_len=config.max_string_len)
+        if raw is None:
+            continue
+
+        if isinstance(raw, str) and key in config.hash_string_values_for_keys:
+            raw = _hash_str(raw)
+
+        # Attribute name per semantic conventions: db.query.parameter.<key> :contentReference[oaicite:8]{index=8}
+        attrs[f"db.query.parameter.{key}"] = raw
+
+    return attrs
```

Key points embedded in code:

* “named only” (`Mapping` check)
* strict allowlist
* (DuckDB) placeholder extraction for `$name`
* skip batch operations (matches the “don’t capture in batch” guidance) ([OpenTelemetry][1])

---

### 2) Wire into your existing DB span attribute builder

This assumes you already have something like a centralized builder that produces semconv-ish DB attributes (`db.system.name`, `db.namespace`, `db.query.summary`, etc.) from earlier patches.

```diff
diff --git a/codeintel/observability/span_attribute_builder.py b/codeintel/observability/span_attribute_builder.py
index 1111111..2222222 100644
--- a/codeintel/observability/span_attribute_builder.py
+++ b/codeintel/observability/span_attribute_builder.py
@@ -1,12 +1,18 @@
 from __future__ import annotations
 
 from typing import Any
+
+from .db_query_parameters import (
+    DBQueryParameterConfig,
+    emit_db_query_parameters,
+)
 
 class SpanAttributeBuilder:
-    def __init__(self, *, sql_summarizer, redactor, dialect_selector):
+    def __init__(self, *, sql_summarizer, redactor, dialect_selector, db_query_parameter_config: DBQueryParameterConfig):
         self._sql_summarizer = sql_summarizer
         self._redactor = redactor
         self._dialect_selector = dialect_selector
+        self._db_query_parameter_config = db_query_parameter_config
 
-    def build_db_span_attributes(self, *, sql: str, db_system_name: str, db_namespace: str) -> dict[str, Any]:
+    def build_db_span_attributes(self, *, sql: str, params: Any | None, db_system_name: str, db_namespace: str, is_batch: bool = False) -> dict[str, Any]:
         attrs: dict[str, Any] = {}
 
         # existing: db.system.name, db.namespace, db.operation.name, db.query.summary, etc.
         attrs["db.system.name"] = db_system_name
         attrs["db.namespace"] = db_namespace
 
         summary = self._sql_summarizer.summarize(sql, dialect=self._dialect_selector(db_system_name))
         if summary:
             attrs["db.query.summary"] = summary
 
+        # NEW: allowlisted named parameter emission (opt-in) :contentReference[oaicite:11]{index=11}
+        attrs.update(
+            emit_db_query_parameters(
+                sql=sql,
+                params=params,
+                db_system_name=db_system_name,
+                config=self._db_query_parameter_config,
+                is_batch=is_batch,
+            )
+        )
+
         return attrs
```

---

### 3) Pass `params` from your central DuckDB execution path

If your “single choke point” looks like `DuckDBSession.execute(sql, params=None)` (or similar), update it so the builder sees params.

```diff
diff --git a/codeintel/storage/duckdb_session.py b/codeintel/storage/duckdb_session.py
index 3333333..4444444 100644
--- a/codeintel/storage/duckdb_session.py
+++ b/codeintel/storage/duckdb_session.py
@@ -1,10 +1,12 @@
 from __future__ import annotations
 
 from typing import Any, Mapping
 from opentelemetry import trace
 
 class DuckDBSession:
     def __init__(self, conn, attr_builder):
         self._conn = conn
         self._attr_builder = attr_builder
         self._tracer = trace.get_tracer(__name__)
 
-    def execute(self, sql: str, params: Any | None = None):
+    def execute(self, sql: str, params: Any | None = None):
         # If you already set span name to db.query.summary, keep that behavior.
         attrs = self._attr_builder.build_db_span_attributes(
             sql=sql,
+            params=params,
             db_system_name="duckdb",
             db_namespace="codeintel",
+            is_batch=False,
         )
 
         span_name = attrs.get("db.query.summary") or "duckdb.query"
         with self._tracer.start_as_current_span(span_name) as span:
             for k, v in attrs.items():
                 span.set_attribute(k, v)
 
             if params is None:
                 return self._conn.execute(sql)
             return self._conn.execute(sql, params)
 
+    def executemany(self, sql: str, params_seq: list[Any]):
+        attrs = self._attr_builder.build_db_span_attributes(
+            sql=sql,
+            params=None,       # batch: don't emit
+            db_system_name="duckdb",
+            db_namespace="codeintel",
+            is_batch=True,     # ensures emit_db_query_parameters skips :contentReference[oaicite:12]{index=12}
+        )
+        span_name = attrs.get("db.query.summary") or "duckdb.executemany"
+        with self._tracer.start_as_current_span(span_name) as span:
+            for k, v in attrs.items():
+                span.set_attribute(k, v)
+            return self._conn.executemany(sql, params_seq)
```

DuckDB’s named parameter convention is `$name` plus a dict passed as the second argument to `execute()`.

---

### 4) Add config wiring (environment → `DBQueryParameterConfig`)

This is intentionally boring and explicit. The core rule: **no keys → no emission**.

```diff
diff --git a/codeintel/config/observability.py b/codeintel/config/observability.py
new file mode 100644
index 0000000..5555555
--- /dev/null
+++ b/codeintel/config/observability.py
@@ -0,0 +1,70 @@
+from __future__ import annotations
+
+import os
+from .types import truthy  # or inline your own
+from codeintel.observability.db_query_parameters import DBQueryParameterConfig
+
+
+def load_db_query_parameter_config(env: dict[str, str] | None = None) -> DBQueryParameterConfig:
+    env = env or dict(os.environ)
+
+    enabled = truthy(env.get("CODEINTEL_OTEL_CAPTURE_DB_QUERY_PARAMETERS", "false"))
+
+    raw_keys = env.get("CODEINTEL_OTEL_DB_QUERY_PARAMETER_KEYS", "")
+    keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
+
+    # Optional: hash string values for specific keys (e.g. repo names)
+    raw_hash_keys = env.get("CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS", "")
+    hash_keys = frozenset(k.strip() for k in raw_hash_keys.split(",") if k.strip())
+
+    require_key_in_sql = truthy(env.get("CODEINTEL_OTEL_DB_QUERY_PARAMETER_REQUIRE_IN_SQL", "true"))
+    max_string_len = int(env.get("CODEINTEL_OTEL_DB_QUERY_PARAMETER_MAX_STRLEN", "80"))
+
+    # Safe default: if enabled but no keys, still do nothing.
+    return DBQueryParameterConfig(
+        enabled=enabled,
+        allowed_keys=keys,
+        require_key_in_sql=require_key_in_sql,
+        max_string_len=max_string_len,
+        hash_string_values_for_keys=hash_keys,
+        disable_on_batch=True,
+    )
```

Example runtime config:

* `CODEINTEL_OTEL_CAPTURE_DB_QUERY_PARAMETERS=true`
* `CODEINTEL_OTEL_DB_QUERY_PARAMETER_KEYS=limit,offset,lang`
* (optional) `CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS=repo`

This matches the “values are opt‑in” stance. ([OpenTelemetry][1])

---

## Tests (drop-in)

These are small but extremely effective at preventing accidental “oops we emitted everything” regressions.

```diff
diff --git a/tests/observability/test_db_query_parameters.py b/tests/observability/test_db_query_parameters.py
new file mode 100644
index 0000000..9999999
--- /dev/null
+++ b/tests/observability/test_db_query_parameters.py
@@ -0,0 +1,120 @@
+from __future__ import annotations
+
+from codeintel.observability.db_query_parameters import (
+    DBQueryParameterConfig,
+    emit_db_query_parameters,
+)
+
+
+def test_disabled_by_default():
+    cfg = DBQueryParameterConfig(enabled=False, allowed_keys=frozenset({"limit"}))
+    attrs = emit_db_query_parameters(
+        sql="SELECT * FROM t WHERE x = $limit",
+        params={"limit": 10},
+        db_system_name="duckdb",
+        config=cfg,
+    )
+    assert attrs == {}
+
+
+def test_named_only_mapping_required():
+    cfg = DBQueryParameterConfig(enabled=True, allowed_keys=frozenset({"limit"}))
+    attrs = emit_db_query_parameters(
+        sql="SELECT * FROM t WHERE x = $limit",
+        params=[10],
+        db_system_name="duckdb",
+        config=cfg,
+    )
+    assert attrs == {}
+
+
+def test_allowlist_and_in_sql_gate():
+    cfg = DBQueryParameterConfig(
+        enabled=True,
+        allowed_keys=frozenset({"limit", "offset"}),
+        require_key_in_sql=True,
+    )
+    attrs = emit_db_query_parameters(
+        sql="SELECT * FROM t LIMIT $limit",
+        params={"limit": 25, "offset": 100},
+        db_system_name="duckdb",
+        config=cfg,
+    )
+    assert attrs == {"db.query.parameter.limit": 25}
+
+
+def test_truncates_strings():
+    cfg = DBQueryParameterConfig(
+        enabled=True,
+        allowed_keys=frozenset({"q"}),
+        max_string_len=5,
+        require_key_in_sql=True,
+    )
+    attrs = emit_db_query_parameters(
+        sql="SELECT * FROM t WHERE q = $q",
+        params={"q": "abcdefgh"},
+        db_system_name="duckdb",
+        config=cfg,
+    )
+    assert attrs["db.query.parameter.q"].startswith("abcd")
+
+
+def test_skips_batch():
+    cfg = DBQueryParameterConfig(enabled=True, allowed_keys=frozenset({"limit"}))
+    attrs = emit_db_query_parameters(
+        sql="INSERT INTO t VALUES ($limit)",
+        params={"limit": 1},
+        db_system_name="duckdb",
+        config=cfg,
+        is_batch=True,
+    )
+    assert attrs == {}
```

---

## Optional variant if you’re using `opentelemetry.instrumentation.dbapi`

If you instrument DB spans via the DBAPI wrapper, note:

* The contrib DBAPI instrumentation historically captured parameters behind a `capture_parameters` flag and used `db.statement.parameters` (older naming). ([OpenTelemetry Python Contrib][2])
* If you turn on its parameter capture naïvely, you’ll likely get **all parameters**, which conflicts with the “strict allowlist” design.

### Lowest-risk approach

Keep DBAPI instrumentation for timing/spans, but do **not** enable its capture-all parameters.

Instead:

* emit allowlisted `db.query.parameter.<key>` from **your own central query wrapper spans** (the approach above), or
* if you *must* attach parameters to the DBAPI-created span, you’ll need a deeper customization (custom cursor proxy / tracer) because the instrumentation sets attributes inside its internal `_populate_span()` path. ([OpenTelemetry Python Contrib][3])

Given your “centralized architecture” goal, the approach above (emit in your central DB span creation point) is usually the cleanest and easiest to reason about.

---

## Practical guidance for choosing allowlist keys

To keep dashboards useful and cheap:

Good allowlist keys (usually low-cardinality):

* `limit`, `offset`
* `lang` / `language`
* `repo_kind` / `index_variant` (categorical)
* `feature_flag` (categorical)

Avoid allowlisting:

* `user_id`, `email`, `path`, `repo_name` (often high-cardinality / sensitive)

If you *really* want visibility for a sensitive string key, prefer hashing (`CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS=repo`) so you can correlate repeated occurrences without exporting the raw string.

---

## Source notes (DuckDB + parameterization)

DuckDB supports both positional (`?`) and named (`$name`) parameters, with named parameters supplied via a dict passed to `execute()`.

---

If you want the *next* incremental refinement after this: we can extend the emitter to also support **positional parameters** using the spec’s 0‑based index keys (still allowlisted), but I kept this patch strictly to your request: **named only + strict allowlist + opt‑in**. ([OpenTelemetry][1])

[1]: https://opentelemetry.io/docs/specs/semconv/database/sql-server/ "Semantic conventions for Microsoft SQL Server client operations | OpenTelemetry"
[2]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/dbapi/dbapi.html "OpenTelemetry Database API Instrumentation — OpenTelemetry Python Contrib  documentation"
[3]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/_modules/opentelemetry/instrumentation/dbapi.html "opentelemetry.instrumentation.dbapi — OpenTelemetry Python Contrib  documentation"
