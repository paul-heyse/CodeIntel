Below is a concrete “diff-style” integration plan that focuses **only on libraries you already depend on** (as seen in your `pyproject.toml`) and on **underused capabilities** that directly improve: clarity, extensibility, hardness, maintainability, and “best-in-class” production polish.

I’m keeping this very practical: **exact files / entry points in your tree**, plus **example diffs** you can apply incrementally.

---

## Patch set 1 — Use your centralized SQLGlot toolkit everywhere (serving SQL fingerprints)

### What this buys you

Right now you have *two* independent SQL fingerprint/canonicalization paths:

* Serving: `src/codeintel/serving/semantic/fingerprints.py` uses a light `parse_one(...).sql(...)`.
* Storage: `src/codeintel/storage/sqlglot_tools.py` already implements a **stronger canonicalization pipeline** (normalize → qualify → optimize → render), and also centralizes fingerprinting.

Unifying these removes semantic drift and makes fingerprints/debugging more stable across the product.

### Entry points

* `src/codeintel/serving/semantic/fingerprints.py` (the duplication)
* `src/codeintel/storage/sqlglot_tools.py` (the canonical source of truth)

### Diff

```diff
diff --git a/src/codeintel/serving/semantic/fingerprints.py b/src/codeintel/serving/semantic/fingerprints.py
index 1111111..2222222 100644
--- a/src/codeintel/serving/semantic/fingerprints.py
+++ b/src/codeintel/serving/semantic/fingerprints.py
@@ -16,11 +16,10 @@ import hashlib
 import json
 from dataclasses import dataclass
 from typing import TYPE_CHECKING
 
-from sqlglot import parse_one
-from sqlglot.errors import ParseError
+from codeintel.storage.sqlglot_tools import ParseError, fingerprint_sql_duckdb
 
 if TYPE_CHECKING:
     from collections.abc import Iterable, Mapping, Sequence
 
@@ -52,23 +51,18 @@ def sqlglot_canonical_sha256(sql: str) -> str:
     Returns
     -------
     str
         SHA256 hex digest of the canonical SQL form.
     """
-    canonical = sql
     try:
-        canonical = parse_one(sql, read="duckdb").sql(dialect="duckdb")
-    except (ParseError, ValueError):
-        canonical = sql
-    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
+        # Use the storage-layer canonicalization pipeline (normalize → qualify → optimize → render).
+        return fingerprint_sql_duckdb(sql)
+    except (ParseError, ValueError):
+        # Fallback: hash raw SQL if parsing/canonicalization fails.
+        return hashlib.sha256(sql.encode("utf-8")).hexdigest()
```

### Rollout note

This will change the *exact* `sql_fingerprint` you emit (because canonicalization is stronger). If that fingerprint is treated as a stable external contract, you can:

* keep the old path as `legacy_sqlglot_canonical_sha256`, or
* version the field (e.g. `sql_fingerprint_v2`) in MCP responses.

---

## Patch set 2 — Add SQLGlot “semantic diff” for upgrade gates + debugging

You already have “upgrade gate” tests (great). The next jump in “best-in-class maintainability” is: when something changes due to upstream (sqlglot/ibis) upgrades, you want **a meaningful semantic diff**, not just “string A != string B”.

### Entry points

* Add helper in: `src/codeintel/storage/sqlglot_tools.py`
* Optional usage in: `tests/storage/test_sql_compiler_upgrade_gates.py`

### Diff (new helper)

```diff
diff --git a/src/codeintel/storage/sqlglot_tools.py b/src/codeintel/storage/sqlglot_tools.py
index 3333333..4444444 100644
--- a/src/codeintel/storage/sqlglot_tools.py
+++ b/src/codeintel/storage/sqlglot_tools.py
@@ -14,7 +14,7 @@ import hashlib
 from collections.abc import Mapping
 from typing import TYPE_CHECKING
 
-from sqlglot import exp, parse_one
+from sqlglot import diff as semantic_diff, exp, parse_one
 from sqlglot.errors import ParseError, SqlglotError
 from sqlglot.lineage import lineage as build_lineage
 from sqlglot.optimizer import build_scope, normalize_identifiers, optimize, qualify
@@ -28,6 +28,7 @@ __all__ = [
     "extract_table_refs",
     "fingerprint_canonical_sql",
     "fingerprint_sql_duckdb",
+    "semantic_diff_sql_duckdb",
     "parse_one_duckdb",
     "render_sql_duckdb",
 ]
@@ -170,6 +171,28 @@ def fingerprint_canonical_sql(canon: str) -> str:
     return hashlib.sha256(canon.encode("utf-8")).hexdigest()
 
+
+def semantic_diff_sql_duckdb(
+    before_sql: str,
+    after_sql: str,
+    *,
+    schema: SchemaMapping | None = None,
+) -> tuple[str, ...]:
+    """Return a semantic diff between two SQL strings (DuckDB dialect).
+
+    Useful for upgrade-gate tests and debugging compiler changes.
+    """
+    before = canonicalize_expression_duckdb(parse_one_duckdb(before_sql), schema=schema)
+    after = canonicalize_expression_duckdb(parse_one_duckdb(after_sql), schema=schema)
+    actions = semantic_diff(before, after)
+    return tuple(str(a) for a in actions)
```

### Optional test usage idea

In your upgrade-gate tests, when you detect mismatch, include `semantic_diff_sql_duckdb(expected, actual)` in the assertion error message. That makes library upgrades far easier to review and approve.

---

## Patch set 3 — First-class Prometheus metrics using deps you already ship

You already have a nice internal `QueryMetrics` object and `log_query_metrics()`. Right now it logs only. Since you already depend on `prometheus-client`, you can turn that into **real operational metrics** with minimal surface-area changes.

### Entry points

* `src/codeintel/serving/metrics.py` (central place: perfect)
* Add `/metrics` to:

  * FastAPI app: `src/codeintel/serving/http/app.py`
  * FastMCP-only HTTP transport: `src/codeintel/serving/mcp/app.py`

### Diff 3A — Emit Prometheus metrics from your existing `log_query_metrics`

```diff
diff --git a/src/codeintel/serving/metrics.py b/src/codeintel/serving/metrics.py
index 5555555..6666666 100644
--- a/src/codeintel/serving/metrics.py
+++ b/src/codeintel/serving/metrics.py
@@ -1,6 +1,7 @@
 """Query performance metrics and logging."""
 
 from __future__ import annotations
 
 import json
 import logging
@@ -9,6 +10,22 @@ from dataclasses import dataclass
 from typing import Any
 
 LOG = logging.getLogger(__name__)
 
+try:
+    from prometheus_client import Counter, Histogram
+
+    _PROM_AVAILABLE = True
+except ImportError:
+    _PROM_AVAILABLE = False
+
+if _PROM_AVAILABLE:
+    CODEINTEL_QUERY_TOTAL = Counter(
+        "codeintel_query_total",
+        "Total number of CodeIntel semantic queries.",
+        labelnames=("endpoint",),
+    )
+    CODEINTEL_QUERY_DURATION_MS = Histogram(
+        "codeintel_query_duration_ms",
+        "Duration of CodeIntel semantic queries (ms).",
+        labelnames=("endpoint",),
+    )
+    CODEINTEL_QUERY_ROWCOUNT = Histogram(
+        "codeintel_query_rowcount",
+        "Row counts returned by CodeIntel semantic queries.",
+        labelnames=("endpoint",),
+    )
@@ -79,6 +96,14 @@ def log_query_metrics(metrics: QueryMetrics) -> None:
     LOG.info(
         "query_metrics %s",
         json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
     )
+
+    # Keep labels low-cardinality (endpoint is good; avoid view_id/query_hash labels).
+    if _PROM_AVAILABLE:
+        CODEINTEL_QUERY_TOTAL.labels(endpoint=metrics.endpoint).inc()
+        CODEINTEL_QUERY_DURATION_MS.labels(endpoint=metrics.endpoint).observe(
+            float(metrics.duration_ms)
+        )
+        CODEINTEL_QUERY_ROWCOUNT.labels(endpoint=metrics.endpoint).observe(
+            float(metrics.row_count)
+        )
```

### Diff 3B — Add `/metrics` to FastAPI (`create_serving_app`)

```diff
diff --git a/src/codeintel/serving/http/app.py b/src/codeintel/serving/http/app.py
index 7777777..8888888 100644
--- a/src/codeintel/serving/http/app.py
+++ b/src/codeintel/serving/http/app.py
@@ -12,6 +12,7 @@ from fastapi.middleware.cors import CORSMiddleware
 from fastapi.openapi.utils import get_openapi
 from fastmcp.server.event_store import EventStore
 from starlette.middleware.gzip import GZipMiddleware
 from starlette.middleware.trustedhost import TrustedHostMiddleware
+from starlette.responses import Response
 
@@ -120,6 +121,26 @@ def create_serving_app(
    app.include_router(build_http_router(features))
 
+    # --- Prometheus metrics endpoint (optional; dependency already present) ---
+    try:
+        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
+
+        @app.get("/metrics", include_in_schema=False)
+        async def metrics_endpoint() -> Response:  # noqa: RUF029
+            payload = generate_latest()
+            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
+
+    except ImportError:
+        # prometheus-client not installed / not enabled in this environment
+        pass
+
     @app.get("/health", include_in_schema=False)
     def health() -> dict[str, str]:
         return {"status": "ok"}
```

### Diff 3C — Add `/metrics` to FastMCP HTTP transport (so it exists even without FastAPI)

```diff
diff --git a/src/codeintel/serving/mcp/app.py b/src/codeintel/serving/mcp/app.py
index 9999999..aaaaaaa 100644
--- a/src/codeintel/serving/mcp/app.py
+++ b/src/codeintel/serving/mcp/app.py
@@ -106,6 +106,7 @@ def build_mcp_app(
    register_prompts(mcp, settings=settings, kernel=ops)
    register_resources(mcp, ops, store, settings=settings)
     _register_health_routes(mcp, ops)
+    _register_metrics_routes(mcp)
 
     return mcp
 
@@ -132,6 +133,26 @@ def _register_health_routes(mcp: FastMCP, ops: ServingOperations) -> None:
     async def mcp_ready(_request: Request) -> Response:  # noqa: RUF029
         try:
             ops.db.current_pointer()
             return PlainTextResponse("ready")
         except RuntimeError:
             return PlainTextResponse("not ready", status_code=503)
 
+
+def _register_metrics_routes(mcp: FastMCP) -> None:
+    try:
+        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
+    except ImportError:
+        return
+
+    @mcp.custom_route("/metrics", methods=["GET"])
+    async def mcp_metrics(_request: Request) -> Response:  # noqa: RUF029
+        payload = generate_latest().decode("utf-8")
+        return PlainTextResponse(payload, media_type=CONTENT_TYPE_LATEST)
+
 
 __all__ = ["build_mcp_app"]
```

---

## Patch set 4 — Ordered, consistent IDs using `uuid6` (UUIDv7) + single factory

You already depend on `uuid6` but still generate UUIDs in multiple styles across:

* correlation IDs (`serving/http/middleware.py`)
* run IDs (`core/execution/ids.py`)
* debug/error IDs (`core/errors/problem_details.py`, `serving/errors/mapping.py`)
* small short IDs (CLI job IDs, etc.)

The “design excellence” move is: **one ID factory**, optional UUIDv7 for ordered IDs, with graceful fallback.

### Entry points

* Add factory in: `src/codeintel/core/execution/ids.py`
* Use it in:

  * `src/codeintel/serving/http/middleware.py`
  * `src/codeintel/core/errors/problem_details.py`
  * `src/codeintel/serving/errors/mapping.py`

### Diff 4A — Central ID factory

```diff
diff --git a/src/codeintel/core/execution/ids.py b/src/codeintel/core/execution/ids.py
index bbbbbbb..ccccccc 100644
--- a/src/codeintel/core/execution/ids.py
+++ b/src/codeintel/core/execution/ids.py
@@ -7,7 +7,18 @@ with configurable prefixes for different execution contexts.
 from __future__ import annotations
 
-from uuid import uuid4
+from uuid import UUID, uuid4
+
+try:
+    # uuid6 provides uuid7(): time-ordered UUIDs (excellent for logs/indices)
+    from uuid6 import uuid7
+
+    _UUID7_AVAILABLE = True
+except ImportError:
+    uuid7 = None  # type: ignore[assignment]
+    _UUID7_AVAILABLE = False
 
 RUN_PREFIX_PIPELINE = "ci"
 RUN_PREFIX_INGEST = "ingest"
@@ -17,6 +28,24 @@ RUN_PREFIX_ANALYTICS = "analytics"
 RUN_PREFIX_PLAN = "plan"
 
+
+def new_uuid() -> UUID:
+    """Return a new UUID, preferring UUIDv7 when available."""
+    if _UUID7_AVAILABLE and uuid7 is not None:
+        return uuid7()
+    return uuid4()
+
+
+def new_uuid_hex() -> str:
+    """Hex form, useful for headers/correlation IDs."""
+    return new_uuid().hex
+
+
+def new_uuid_str() -> str:
+    """String form, useful for human-facing debug IDs."""
+    return str(new_uuid())
+
+
 def new_run_id(prefix: str = RUN_PREFIX_PIPELINE) -> str:
@@ -39,7 +68,7 @@ def new_run_id(prefix: str = RUN_PREFIX_PIPELINE) -> str:
     >>> len(rid.split("-", 1)[1]) == 32
     True
     """
-    return f"{prefix}-{uuid4().hex}"
+    return f"{prefix}-{new_uuid_hex()}"
 
 __all__ = [
     "RUN_PREFIX_ANALYTICS",
@@ -49,5 +78,8 @@ __all__ = [
     "RUN_PREFIX_PIPELINE",
     "RUN_PREFIX_PLAN",
     "new_run_id",
+    "new_uuid",
+    "new_uuid_hex",
+    "new_uuid_str",
 ]
```

### Diff 4B — Correlation middleware uses the same factory

```diff
diff --git a/src/codeintel/serving/http/middleware.py b/src/codeintel/serving/http/middleware.py
index ddddddd..eeeeeee 100644
--- a/src/codeintel/serving/http/middleware.py
+++ b/src/codeintel/serving/http/middleware.py
@@ -5,11 +5,12 @@ from __future__ import annotations
 
 import time
-import uuid
 from typing import TYPE_CHECKING
 
+from codeintel.core.execution.ids import new_uuid_hex
+
 if TYPE_CHECKING:
     from collections.abc import Awaitable, Callable
 
     from fastapi import Request, Response
@@ -60,7 +61,7 @@ async def correlation_id_and_timing_middleware(
     start = time.perf_counter()
 
     incoming = request.headers.get(CORRELATION_ID_HEADER)
-    correlation_id = incoming.strip() if incoming else uuid.uuid4().hex
+    correlation_id = incoming.strip() if incoming else new_uuid_hex()
     request.state.correlation_id = correlation_id
```

### Diff 4C — ProblemDetails instance IDs and serving debug IDs unify

```diff
diff --git a/src/codeintel/core/errors/problem_details.py b/src/codeintel/core/errors/problem_details.py
index fffffff..1111111 100644
--- a/src/codeintel/core/errors/problem_details.py
+++ b/src/codeintel/core/errors/problem_details.py
@@ -15,7 +15,8 @@ from __future__ import annotations
 import json
 from dataclasses import dataclass, field
 from typing import Any
-from uuid import uuid4
+
+from codeintel.core.execution.ids import new_uuid_str
 
 
 def generate_instance_id() -> str:
@@ -26,7 +27,7 @@ def generate_instance_id() -> str:
     str
         UUID4 string for error instance identification.
     """
-    return str(uuid4())
+    return new_uuid_str()
```

```diff
diff --git a/src/codeintel/serving/errors/mapping.py b/src/codeintel/serving/errors/mapping.py
index 2222222..3333333 100644
--- a/src/codeintel/serving/errors/mapping.py
+++ b/src/codeintel/serving/errors/mapping.py
@@ -6,7 +6,6 @@ from __future__ import annotations
 from dataclasses import dataclass
 from datetime import UTC, datetime
 from typing import TYPE_CHECKING, Any
-from uuid import uuid4
 
 from pydantic import ValidationError
 
+from codeintel.core.execution.ids import new_uuid_str
 from codeintel.serving.errors.catalog import ERROR_CODE_CATALOG
 from codeintel.serving.errors.models import ErrorContext, ErrorInfo, ErrorResponse
 from codeintel.serving.uris import EXPORT_RESOURCE_PREFIX, META_VIEWS_SQL_URI
@@ -44,7 +43,7 @@ def _context_to_details(context: ErrorContext | None) -> dict[str, Any]:
         "commit": context.commit,
         "run_id": context.run_id,
         "request_id": context.request_id,
-        "debug_id": context.debug_id or str(uuid4()),
+        "debug_id": context.debug_id or new_uuid_str(),
         "ts": datetime.now(UTC).isoformat(),
     }
```

---

## Patch set 5 — Faster, clearer NDJSON streaming using `msgspec` (optional fallback)

You already depend on `msgspec` but aren’t using it. Your NDJSON streamer is a perfect “high leverage” spot because:

* it’s performance-sensitive,
* it’s isolated,
* it improves readability (a single encoder definition),
* it’s safe to fallback to stdlib `json`.

### Entry point

* `src/codeintel/serving/http/streaming.py`

### Diff

```diff
diff --git a/src/codeintel/serving/http/streaming.py b/src/codeintel/serving/http/streaming.py
index 4444444..5555555 100644
--- a/src/codeintel/serving/http/streaming.py
+++ b/src/codeintel/serving/http/streaming.py
@@ -8,6 +8,19 @@ from __future__ import annotations
 
 import json
 from typing import TYPE_CHECKING
 
 from starlette.responses import StreamingResponse
 
 from codeintel.serving.export.formats import mime_type_for_export_format
 
 if TYPE_CHECKING:
     from collections.abc import Iterable, Iterator, Mapping
 
+try:
+    import msgspec
+
+    def _enc_hook(obj: object) -> object:
+        # Preserve your current behavior: stringify unknown types (datetime, Decimal, UUID, etc.)
+        return str(obj)
+
+    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder(enc_hook=_enc_hook)
+except ImportError:
+    _MSG_ENCODER = None
+
 
 def ndjson_stream(rows: Iterable[dict[str, object]]) -> Iterator[bytes]:
@@ -23,7 +36,14 @@ def ndjson_stream(rows: Iterable[dict[str, object]]) -> Iterator[bytes]:
         JSON-encoded row followed by newline.
     """
     for row in rows:
-        yield json.dumps(row, default=str).encode("utf-8") + b"\n"
+        if _MSG_ENCODER is not None:
+            yield _MSG_ENCODER.encode(row) + b"\n"
+        else:
+            yield json.dumps(
+                row, default=str, separators=(",", ":"), ensure_ascii=False
+            ).encode("utf-8") + b"\n"
```

---

## Patch set 6 — Use FastMCP `ResourceContent` to set correct MIME + attach metadata

Your export resources currently return `str | bytes` and rely on decorator/static MIME in some cases. FastMCP supports returning a `ResourceContent` object (fine-grained MIME + metadata). This is a “best-in-class” polish move because:

* binary exports get the right `mime_type` consistently,
* you can attach client hints (filename, caching policy, provenance),
* it reduces “guesswork” in clients.

### Entry point

* `src/codeintel/serving/mcp/resources/exports.py` (`read_export`)

### Diff

```diff
diff --git a/src/codeintel/serving/mcp/resources/exports.py b/src/codeintel/serving/mcp/resources/exports.py
index 6666666..7777777 100644
--- a/src/codeintel/serving/mcp/resources/exports.py
+++ b/src/codeintel/serving/mcp/resources/exports.py
@@ -6,6 +6,7 @@ from __future__ import annotations
 
 from datetime import UTC, datetime
 from typing import TYPE_CHECKING
 
+from fastmcp.resources import ResourceContent
 from codeintel.serving.errors import ExportNotFoundError
 from codeintel.serving.export.formats import (
     mime_type_for_export_format,
     suffix_for_export_format,
@@ -96,14 +97,26 @@ def _register_export_read_resource(
     settings: ServingSettings,
 ) -> None:
     @mcp.resource(EXPORT_URI_TEMPLATE)
-    def read_export(export_id: str) -> str | bytes:
+    def read_export(export_id: str) -> ResourceContent:
         artifact = store.get(export_id)
         if artifact.size_bytes > settings.mcp_export_max_full_read_bytes:
             raise ExportFullReadNotAllowedError
-        if artifact.mime_type in {_MIME_PARQUET, _MIME_ARROW}:
-            return artifact.path.read_bytes()
-        return artifact.path.read_text(encoding="utf-8")
+        is_binary = artifact.mime_type in {_MIME_PARQUET, _MIME_ARROW}
+        payload: str | bytes = (
+            artifact.path.read_bytes()
+            if is_binary
+            else artifact.path.read_text(encoding="utf-8")
+        )
+        return ResourceContent(
+            content=payload,
+            mime_type=artifact.mime_type,
+            meta={
+                "export_id": export_id,
+                "row_count": artifact.row_count,
+                "size_bytes": artifact.size_bytes,
+            },
+        )
```

---

## Suggested implementation order (lowest risk → highest leverage)

1. **Patch set 1 (SQL fingerprint unification)**
   Small change, large “design consistency” gain.

2. **Patch set 3A + 3B (Prometheus from `log_query_metrics` + `/metrics` in FastAPI)**
   Very low risk, immediate operational value.

3. **Patch set 3C (Prometheus endpoint in FastMCP-only HTTP mode)**
   Ensures parity no matter how you deploy.

4. **Patch set 4 (uuid6 + single ID factory)**
   Great polish; do it once and benefit everywhere.

5. **Patch set 5 (msgspec NDJSON)**
   Optional, very contained, easy to benchmark.

6. **Patch set 6 (FastMCP ResourceContent)**
   Improves client experience + correctness for binary exports.

---

## Reference docs (the advanced feature notes align with these)





If you want, I can produce the same diff-style plan for the **next tier** of “already-in-deps but currently underleveraged” features I noticed in your tree (e.g., OpenTelemetry packages you ship but only use on the CLI side, plus a clean way to make tracing/metrics symmetrical across CLI + HTTP + MCP without duplicating code).

# second set of diff-style plans #

Below is a **concrete “diff-style” integration plan** for the “already-in-deps but underleveraged” observability features—specifically: **OpenTelemetry everywhere** (CLI + HTTP + MCP), **symmetrical tracing + metrics**, and **one shared implementation** so you don’t duplicate code across entrypoints.

I’m basing the file/entrypoint touchpoints on your tree (notably: `codeintel/cli/observability/*`, `codeintel/serving/http/app.py`, `codeintel/serving/http/middleware.py`, `codeintel/serving/http/route_utils.py`, `codeintel/serving/mcp/middleware_stack.py`, `codeintel/serving/metrics.py`).

---

## 0) Design target (what this patch set accomplishes)

After these changes:

* **CLI**: every operation executed via `@cli_command(...)` runs inside a span + records duration metrics (and flushes on exit).
* **HTTP**: FastAPI gets **request tracing** (via `opentelemetry-instrumentation-fastapi` you already ship), and your existing `run_in_threadpool_with_metrics()` becomes the *single* place that creates “domain operation” spans + duration metrics.
* **MCP**: one middleware adds a span per MCP message/tool call and records duration metrics (no need to edit each tool).
* **Query metrics** you already log (`QueryMetrics`) become **real OTel metrics** too (and optionally Prometheus-exposed) without changing call sites (because you already centralize via `log_query_metrics()`).

---

## 1) Add a shared observability package (new files)

### 1.1 `src/codeintel/observability/__init__.py` (new)

```diff
+++ src/codeintel/observability/__init__.py
+"""Shared observability utilities (CLI + HTTP + MCP).
+
+Goal: one implementation of tracing/metrics/context that every entrypoint calls.
+"""
+
+from __future__ import annotations
+
+from codeintel.observability.context import (
+    correlation_context,
+    get_correlation_id,
+    set_correlation_id,
+)
+from codeintel.observability.otel import (
+    ObservabilityConfig,
+    bootstrap_observability,
+    get_observability,
+    shutdown_observability,
+)
+from codeintel.observability.operations import (
+    observe_operation,
+    record_operation_metrics,
+    record_query_metrics,
+)
+
+__all__ = [
+    "ObservabilityConfig",
+    "bootstrap_observability",
+    "get_observability",
+    "shutdown_observability",
+    "correlation_context",
+    "get_correlation_id",
+    "set_correlation_id",
+    "observe_operation",
+    "record_operation_metrics",
+    "record_query_metrics",
+]
```

### 1.2 `src/codeintel/observability/otel.py` (new)

This is the **one place** that configures OTel SDK providers + exporters (OTLP, console, Prometheus), and makes initialization idempotent.

```diff
+++ src/codeintel/observability/otel.py
+"""OpenTelemetry bootstrap for CodeIntel.
+
+Centralizes:
+- TracerProvider + exporters (OTLP, optional console)
+- MeterProvider + exporters/readers (OTLP, optional Prometheus)
+- Optional instrumentors (logging/asyncio/threading) you already depend on
+"""
+
+from __future__ import annotations
+
+import logging
+import os
+from dataclasses import dataclass
+from importlib.metadata import PackageNotFoundError, version
+from typing import TYPE_CHECKING
+
+from codeintel.core.singleton import SingletonHolder
+
+LOG = logging.getLogger(__name__)
+
+if TYPE_CHECKING:
+    from opentelemetry.trace import Tracer
+    from opentelemetry.metrics import Meter
+
+
+def _bool_env(name: str, *, default: bool) -> bool:
+    raw = os.environ.get(name)
+    if raw is None:
+        return default
+    raw = raw.strip().lower()
+    return raw in {"1", "true", "yes", "on"}
+
+
+def _pkg_version() -> str:
+    try:
+        return version("codeintel")
+    except PackageNotFoundError:
+        return "unknown"
+
+
+@dataclass(frozen=True, slots=True)
+class ObservabilityConfig:
+    """Runtime config for observability bootstrap."""
+
+    enabled: bool = True
+    service_name: str = "codeintel"
+    otlp_endpoint: str | None = None
+    export_traces: bool = True
+    export_metrics: bool = True
+    console_export: bool = False
+    prometheus_enabled: bool = False
+
+    @staticmethod
+    def from_env(*, default_service_name: str) -> "ObservabilityConfig":
+        # Standard-ish OpenTelemetry knobs + your existing CodeIntel envs used in cli/observability/_telemetry.py
+        sdk_disabled = _bool_env("OTEL_SDK_DISABLED", default=False)
+        service_name = os.environ.get("OTEL_SERVICE_NAME", "").strip() or default_service_name
+        otlp = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() or None
+
+        export_traces = _bool_env("CODEINTEL_EXPORT_TRACES", default=True)
+        export_metrics = _bool_env("CODEINTEL_EXPORT_METRICS", default=True)
+        console_export = _bool_env("CODEINTEL_CONSOLE_TELEMETRY", default=False)
+
+        # New (optional) switch: export metrics for Prometheus scraping
+        prometheus_enabled = _bool_env("CODEINTEL_PROMETHEUS_METRICS", default=False)
+
+        return ObservabilityConfig(
+            enabled=not sdk_disabled,
+            service_name=service_name,
+            otlp_endpoint=otlp,
+            export_traces=export_traces,
+            export_metrics=export_metrics,
+            console_export=console_export,
+            prometheus_enabled=prometheus_enabled,
+        )
+
+
+@dataclass(frozen=True, slots=True)
+class ObservabilityRuntime:
+    enabled: bool
+    tracer: "Tracer | None"
+    meter: "Meter | None"
+    _shutdown: callable | None
+
+
+class _ObsHolder(SingletonHolder[ObservabilityRuntime]):
+    pass
+
+
+def bootstrap_observability(cfg: ObservabilityConfig) -> ObservabilityRuntime:
+    """Idempotently configure OTel providers + exporters."""
+
+    def _init() -> ObservabilityRuntime:
+        if not cfg.enabled:
+            return ObservabilityRuntime(enabled=False, tracer=None, meter=None, _shutdown=None)
+
+        try:
+            from opentelemetry import metrics, trace
+            from opentelemetry.sdk.resources import Resource
+            from opentelemetry.sdk.trace import TracerProvider
+            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
+            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
+
+            from opentelemetry.sdk.metrics import MeterProvider
+            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
+            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
+        except Exception as exc:
+            LOG.warning("OpenTelemetry unavailable; observability disabled (%s)", exc)
+            return ObservabilityRuntime(enabled=False, tracer=None, meter=None, _shutdown=None)
+
+        # Optional instrumentors you already ship (safe to no-op if missing).
+        try:
+            from opentelemetry.instrumentation.logging import LoggingInstrumentor
+            LoggingInstrumentor().instrument(set_logging_format=False)
+        except Exception:
+            pass
+        try:
+            from opentelemetry.instrumentation.threading import ThreadingInstrumentor
+            ThreadingInstrumentor().instrument()
+        except Exception:
+            pass
+        try:
+            from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
+            AsyncioInstrumentor().instrument()
+        except Exception:
+            pass
+
+        resource = Resource.create(
+            {
+                "service.name": cfg.service_name,
+                "service.version": _pkg_version(),
+            }
+        )
+
+        # ---- Tracing
+        tracer_provider = TracerProvider(resource=resource)
+        if cfg.export_traces and cfg.otlp_endpoint:
+            tracer_provider.add_span_processor(
+                BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint))
+            )
+        if cfg.console_export:
+            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
+
+        trace.set_tracer_provider(tracer_provider)
+        tracer = trace.get_tracer("codeintel")
+
+        # ---- Metrics
+        metric_readers = []
+        if cfg.export_metrics and cfg.otlp_endpoint:
+            metric_readers.append(
+                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=cfg.otlp_endpoint))
+            )
+
+        if cfg.prometheus_enabled:
+            try:
+                from opentelemetry.exporter.prometheus import PrometheusMetricReader
+                metric_readers.append(PrometheusMetricReader())
+            except Exception:
+                LOG.warning("PrometheusMetricReader unavailable; skipping Prometheus export")
+
+        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers or None)
+        metrics.set_meter_provider(meter_provider)
+        meter = metrics.get_meter("codeintel")
+
+        def _shutdown() -> None:
+            try:
+                tracer_provider.shutdown()
+            except Exception:
+                pass
+            try:
+                meter_provider.shutdown()
+            except Exception:
+                pass
+
+        return ObservabilityRuntime(enabled=True, tracer=tracer, meter=meter, _shutdown=_shutdown)
+
+    return _ObsHolder.get(_init)
+
+
+def get_observability() -> ObservabilityRuntime:
+    existing = _ObsHolder.get_or_none()
+    if existing is not None:
+        return existing
+    # Safe default: disabled until explicitly bootstrapped by an entrypoint.
+    return ObservabilityRuntime(enabled=False, tracer=None, meter=None, _shutdown=None)
+
+
+def shutdown_observability() -> None:
+    runtime = _ObsHolder.get_or_none()
+    if runtime is None or runtime._shutdown is None:
+        return
+    runtime._shutdown()
```

### 1.3 `src/codeintel/observability/context.py` (new)

This gives you **one correlation-id context** across CLI + HTTP + MCP, and optionally puts it into OTel baggage so child spans can “see” it.

```diff
+++ src/codeintel/observability/context.py
+"""Correlation-id context shared across transports."""
+
+from __future__ import annotations
+
+import contextlib
+import contextvars
+from typing import Iterator
+
+_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
+    "codeintel_correlation_id",
+    default=None,
+)
+
+
+def get_correlation_id() -> str | None:
+    return _CORRELATION_ID.get()
+
+
+def set_correlation_id(value: str | None) -> None:
+    _CORRELATION_ID.set(value)
+
+
+@contextlib.contextmanager
+def correlation_context(correlation_id: str) -> Iterator[None]:
+    token = _CORRELATION_ID.set(correlation_id)
+
+    # Optional: propagate via OTel baggage if available.
+    baggage_token = None
+    try:
+        from opentelemetry import baggage
+        from opentelemetry.context import attach, detach
+
+        baggage_token = attach(baggage.set_baggage("correlation_id", correlation_id))
+    except Exception:
+        baggage_token = None
+
+    try:
+        yield
+    finally:
+        _CORRELATION_ID.reset(token)
+        if baggage_token is not None:
+            try:
+                from opentelemetry.context import detach
+                detach(baggage_token)
+            except Exception:
+                pass
```

### 1.4 `src/codeintel/observability/operations.py` (new)

This is the shared “one-liner” API you’ll use everywhere: `observe_operation(...)` and `record_query_metrics(...)`.

```diff
+++ src/codeintel/observability/operations.py
+"""Shared span + metric helpers for CodeIntel operations."""
+
+from __future__ import annotations
+
+import time
+from contextlib import contextmanager
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, Iterator, Protocol
+
+from codeintel.observability.context import get_correlation_id
+from codeintel.observability.otel import get_observability
+
+if TYPE_CHECKING:
+    from opentelemetry.trace import Span
+
+
+class QueryMetricsLike(Protocol):
+    endpoint: str
+    duration_ms: float
+    correlation_id: str
+    view_id: str | None
+    row_count: int
+    truncated: bool
+    engine: str | None
+    query_hash: str | None
+    schema_hash: str | None
+
+
+@dataclass(slots=True)
+class _Instruments:
+    op_calls: object
+    op_duration_ms: object
+    query_calls: object
+    query_duration_ms: object
+    query_row_count: object
+    query_truncated: object
+
+
+_INSTRUMENTS: _Instruments | None = None
+
+
+def _get_instruments() -> _Instruments | None:
+    global _INSTRUMENTS
+    if _INSTRUMENTS is not None:
+        return _INSTRUMENTS
+
+    obs = get_observability()
+    if not obs.enabled or obs.meter is None:
+        return None
+
+    meter = obs.meter
+    _INSTRUMENTS = _Instruments(
+        op_calls=meter.create_counter(
+            "codeintel.operation.calls",
+            unit="1",
+            description="Count of CodeIntel operations across CLI/HTTP/MCP",
+        ),
+        op_duration_ms=meter.create_histogram(
+            "codeintel.operation.duration_ms",
+            unit="ms",
+            description="Operation duration (ms) across CLI/HTTP/MCP",
+        ),
+        query_calls=meter.create_counter(
+            "codeintel.query.calls",
+            unit="1",
+            description="Count of semantic queries/exports across transports",
+        ),
+        query_duration_ms=meter.create_histogram(
+            "codeintel.query.duration_ms",
+            unit="ms",
+            description="Query duration (ms) across transports",
+        ),
+        query_row_count=meter.create_histogram(
+            "codeintel.query.row_count",
+            unit="1",
+            description="Row counts returned/exported",
+        ),
+        query_truncated=meter.create_counter(
+            "codeintel.query.truncated",
+            unit="1",
+            description="Count of truncated query responses",
+        ),
+    )
+    return _INSTRUMENTS
+
+
+def record_operation_metrics(
+    *,
+    component: str,
+    operation: str,
+    duration_ms: float,
+    success: bool,
+) -> None:
+    instruments = _get_instruments()
+    if instruments is None:
+        return
+    attrs = {
+        "codeintel.component": component,  # cli | http | mcp
+        "codeintel.operation": operation,
+        "codeintel.success": bool(success),
+    }
+    instruments.op_calls.add(1, attributes=attrs)
+    instruments.op_duration_ms.record(duration_ms, attributes=attrs)
+
+
+@contextmanager
+def observe_operation(
+    *,
+    component: str,
+    operation: str,
+    attributes: dict[str, object] | None = None,
+) -> Iterator["Span | None"]:
+    """Create a child span (if tracing enabled) + record duration metrics."""
+    obs = get_observability()
+    cid = get_correlation_id()
+
+    span_cm = None
+    span = None
+    if obs.enabled and obs.tracer is not None:
+        span_cm = obs.tracer.start_as_current_span(f"{component}.{operation}")
+
+    start = time.perf_counter()
+    success = False
+    try:
+        if span_cm is None:
+            yield None
+        else:
+            with span_cm as active_span:
+                span = active_span
+                if cid:
+                    span.set_attribute("codeintel.correlation_id", cid)
+                span.set_attribute("codeintel.component", component)
+                span.set_attribute("codeintel.operation", operation)
+                if attributes:
+                    for k, v in attributes.items():
+                        span.set_attribute(k, v)
+                yield span
+        success = True
+    except Exception as exc:
+        if span is not None:
+            try:
+                span.record_exception(exc)
+            except Exception:
+                pass
+        raise
+    finally:
+        duration_ms = (time.perf_counter() - start) * 1000
+        record_operation_metrics(
+            component=component,
+            operation=operation,
+            duration_ms=duration_ms,
+            success=success,
+        )
+
+
+def record_query_metrics(metrics: QueryMetricsLike) -> None:
+    """Turn your existing QueryMetrics into OTel metrics + span attributes."""
+    instruments = _get_instruments()
+    obs = get_observability()
+
+    attrs = {
+        "codeintel.endpoint": metrics.endpoint,
+        "codeintel.view_id": metrics.view_id or "",
+        "codeintel.engine": metrics.engine or "",
+        "codeintel.query_hash": metrics.query_hash or "",
+        "codeintel.schema_hash": metrics.schema_hash or "",
+    }
+
+    if instruments is not None:
+        instruments.query_calls.add(1, attributes=attrs)
+        instruments.query_duration_ms.record(metrics.duration_ms, attributes=attrs)
+        instruments.query_row_count.record(metrics.row_count, attributes=attrs)
+        if metrics.truncated:
+            instruments.query_truncated.add(1, attributes=attrs)
+
+    # Attach to current span if present.
+    if obs.enabled:
+        try:
+            from opentelemetry import trace
+            span = trace.get_current_span()
+            if span is not None:
+                span.set_attribute("codeintel.query.endpoint", metrics.endpoint)
+                span.set_attribute("codeintel.query.row_count", metrics.row_count)
+                span.set_attribute("codeintel.query.truncated", bool(metrics.truncated))
+                if metrics.view_id:
+                    span.set_attribute("codeintel.query.view_id", metrics.view_id)
+                if metrics.query_hash:
+                    span.set_attribute("codeintel.query.hash", metrics.query_hash)
+                if metrics.schema_hash:
+                    span.set_attribute("codeintel.query.schema_hash", metrics.schema_hash)
+        except Exception:
+            pass
```

---

## 2) Wire CLI into the shared OTel bootstrap + per-command spans

### 2.1 Update `src/codeintel/cli/execution/bootstrap.py`

Call `bootstrap_observability()` once when CLI config says telemetry is enabled.

```diff
--- src/codeintel/cli/execution/bootstrap.py
+++ src/codeintel/cli/execution/bootstrap.py
@@
 from codeintel.cli.config import load_config as load_cli_config
 from codeintel.cli.observability._observability import configure_structured_logging
+from codeintel.observability.otel import ObservabilityConfig, bootstrap_observability
@@
 def bootstrap_cli(
@@
         use_structured = structured_logging or active_config.telemetry.enabled
         _configure_logging(verbosity, active_config, structured=use_structured)
 
+        # OTel bootstrap (idempotent).
+        bootstrap_observability(
+            ObservabilityConfig(
+                enabled=active_config.telemetry.enabled,
+                service_name=active_config.telemetry.service_name,
+                otlp_endpoint=active_config.telemetry.endpoint,
+                export_traces=True,
+                export_metrics=True,
+            )
+        )
+
         _register_signal_handlers()
```

### 2.2 Update `src/codeintel/cli/commands/decorators.py`

Wrap handler execution in a span + duration metrics, and flush on exit.

```diff
--- src/codeintel/cli/commands/decorators.py
+++ src/codeintel/cli/commands/decorators.py
@@
 from codeintel.cli.execution.bootstrap import bootstrap_cli
+from codeintel.observability import observe_operation, shutdown_observability
@@
 def _execute_handler_command[R](
@@
     bootstrap_cli(verbosity=infra.verbosity)
@@
     with builder.build() as ctx:
-        try:
-            result = handler(ctx)
-        except Exception:
-            LOG.exception("Handler %s raised exception", operation_id)
-            raise
+        try:
+            with observe_operation(
+                component="cli",
+                operation=operation_id,
+                attributes={"codeintel.output_format": str(infra.output_format)},
+            ):
+                result = handler(ctx)
+        except Exception:
+            LOG.exception("Handler %s raised exception", operation_id)
+            raise
+        finally:
+            # Ensure exporters flush even if the process exits quickly.
+            shutdown_observability()
```

> If you also have “new-style” command execution in this file, apply the same pattern around `command.execute(ctx)`.

---

## 3) Wire HTTP serving: bootstrap once, instrument FastAPI, and add “domain spans” in your centralized threadpool wrapper

### 3.1 Update `src/codeintel/serving/http/app.py`

* Bootstrap OTel in the app factory (works for reload + multi-worker factory mode).
* Instrument FastAPI (outer middleware).
* Optionally expose `/metrics` when `CODEINTEL_PROMETHEUS_METRICS=1`.

```diff
--- src/codeintel/serving/http/app.py
+++ src/codeintel/serving/http/app.py
@@
 from codeintel.serving.http.middleware import correlation_id_and_timing_middleware
 from codeintel.serving.http.routes.v1 import export, health, meta, search, semantic
 from codeintel.serving.http.state import ServingState
+from codeintel.observability.otel import ObservabilityConfig, bootstrap_observability
@@
 def create_serving_app(settings: ServingSettings) -> FastAPI:
@@
     app.state.serving = state
 
     _install_exception_handlers(app)
     _install_middlewares(app, settings)
     _install_routes(app, settings)
+
+    # OTel: configure providers + exporters for this process.
+    bootstrap_observability(ObservabilityConfig.from_env(default_service_name="codeintel-serving"))
+
+    # OTel: instrument FastAPI (you already depend on opentelemetry-instrumentation-fastapi).
+    try:
+        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
+        FastAPIInstrumentor.instrument_app(app)
+    except Exception:
+        pass
+
+    # Optional Prometheus scrape endpoint if PrometheusMetricReader is enabled.
+    if os.environ.get("CODEINTEL_PROMETHEUS_METRICS"):
+        try:
+            from fastapi import Response
+            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
+
+            @app.get("/metrics", include_in_schema=False)
+            async def metrics() -> Response:
+                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
+        except Exception:
+            pass
 
     return app
```

### 3.2 Update `src/codeintel/serving/http/route_utils.py`

This is your “thin adapter” central choke point. Add a child span + operation metrics here so **every endpoint using this wrapper** is automatically instrumented.

```diff
--- src/codeintel/serving/http/route_utils.py
+++ src/codeintel/serving/http/route_utils.py
@@
 from codeintel.serving.http.middleware import get_correlation_id
 from codeintel.serving.metrics import QueryMetrics, log_query_metrics
+from codeintel.observability import observe_operation
@@
 async def run_in_threadpool_with_metrics(
@@
 ) -> T:
@@
     correlation_id = get_correlation_id(request)
     start = time.perf_counter()
     try:
-        result = await run_in_threadpool(fn, *args, **kwargs)
+        # Child span under the HTTP request span (if tracing is enabled).
+        op = f"{request.method} {request.url.path}"
+        with observe_operation(
+            component="http",
+            operation=op,
+            attributes={
+                "http.method": request.method,
+                "http.route": request.url.path,
+                "codeintel.correlation_id": correlation_id,
+            },
+        ):
+            result = await run_in_threadpool(fn, *args, **kwargs)
     except Exception:
         duration_ms = (time.perf_counter() - start) * 1000
         schedule_query_metrics(background, error_metrics(duration_ms, correlation_id))
         raise
```

### 3.3 Update `src/codeintel/serving/http/middleware.py`

Use the shared correlation context and attach correlation_id to the current span when present.

Also (optional but nice): switch correlation IDs to `uuid7` (you already ship `uuid6` but don’t use it today).

```diff
--- src/codeintel/serving/http/middleware.py
+++ src/codeintel/serving/http/middleware.py
@@
-import uuid
+import uuid
@@
 from starlette.types import ASGIApp
+from codeintel.observability import correlation_context
@@
 def _generate_correlation_id() -> str:
-    return f"req-{uuid.uuid4().hex}"
+    # Optional: if you want sortable IDs and you already depend on uuid6:
+    # from uuid6 import uuid7
+    # return f"req-{uuid7().hex}"
+    return f"req-{uuid.uuid4().hex}"
@@
 async def correlation_id_and_timing_middleware(request: Request, call_next: Callable) -> Response:
@@
     request.state.correlation_id = correlation_id
@@
-    try:
-        response = await call_next(request)
+    try:
+        with correlation_context(correlation_id):
+            # If FastAPI OTel middleware is active, this sets correlation_id on the current request span.
+            try:
+                from opentelemetry import trace
+                span = trace.get_current_span()
+                if span is not None:
+                    span.set_attribute("codeintel.correlation_id", correlation_id)
+            except Exception:
+                pass
+
+            response = await call_next(request)
     finally:
         duration_ms = (time.perf_counter() - start) * 1000
```

---

## 4) Wire MCP: one middleware adds spans + operation metrics for every MCP call

### 4.1 Add `src/codeintel/observability/mcp.py` (new)

```diff
+++ src/codeintel/observability/mcp.py
+"""FastMCP middleware for OpenTelemetry spans + operation metrics."""
+
+from __future__ import annotations
+
+import time
+from typing import TYPE_CHECKING
+
+from fastmcp.server.middleware.middleware import Middleware
+
+from codeintel.observability import correlation_context
+from codeintel.observability.otel import get_observability
+from codeintel.observability.operations import record_operation_metrics
+
+if TYPE_CHECKING:
+    from fastmcp.server.middleware.middleware import CallNext, MiddlewareContext
+
+
+class McpOpenTelemetryMiddleware(Middleware):
+    """Create a span per MCP message and record duration metrics."""
+
+    async def on_message(
+        self,
+        context: "MiddlewareContext[object]",
+        call_next: "CallNext[object, object]",
+    ) -> object:
+        obs = get_observability()
+
+        method = context.method or "unknown"
+        tool_name = None
+        if method == "tools/call":
+            msg = getattr(context, "message", None)
+            tool_name = getattr(msg, "name", None) if msg is not None else None
+
+        op = f"{method}:{tool_name}" if tool_name else method
+
+        # Use FastMCP session_id as a correlation id fallback.
+        fastmcp_ctx = context.fastmcp_context
+        session_id = None
+        if fastmcp_ctx is not None:
+            try:
+                session_id = getattr(fastmcp_ctx, "session_id", None)
+            except RuntimeError:
+                session_id = None
+
+        correlation_id = session_id or "mcp-unknown"
+
+        start = time.perf_counter()
+        success = False
+        try:
+            with correlation_context(correlation_id):
+                if obs.enabled and obs.tracer is not None:
+                    with obs.tracer.start_as_current_span(f"mcp.{op}") as span:
+                        span.set_attribute("codeintel.component", "mcp")
+                        span.set_attribute("mcp.method", method)
+                        if tool_name:
+                            span.set_attribute("mcp.tool_name", str(tool_name))
+                        span.set_attribute("codeintel.correlation_id", correlation_id)
+                        result = await call_next(context)
+                else:
+                    result = await call_next(context)
+
+            success = True
+            return result
+        except Exception as exc:
+            if obs.enabled:
+                try:
+                    from opentelemetry import trace
+                    span = trace.get_current_span()
+                    if span is not None:
+                        span.record_exception(exc)
+                except Exception:
+                    pass
+            raise
+        finally:
+            duration_ms = (time.perf_counter() - start) * 1000
+            record_operation_metrics(
+                component="mcp",
+                operation=op,
+                duration_ms=duration_ms,
+                success=success,
+            )
+
+
+__all__ = ["McpOpenTelemetryMiddleware"]
```

### 4.2 Update `src/codeintel/serving/mcp/middleware_stack.py`

Insert the new middleware at the top so it wraps everything (including error mapping).

```diff
--- src/codeintel/serving/mcp/middleware_stack.py
+++ src/codeintel/serving/mcp/middleware_stack.py
@@
 from codeintel.serving.mcp.middleware_errors import CodeIntelErrorMappingMiddleware
+from codeintel.observability.mcp import McpOpenTelemetryMiddleware
@@
 def build_mcp_middleware(settings: ServingSettings) -> list[Middleware]:
     middleware: list[Middleware] = []
 
+    # OTel spans + duration metrics for every MCP message.
+    middleware.append(McpOpenTelemetryMiddleware())
+
     middleware.append(CodeIntelErrorMappingMiddleware())
```

### 4.3 Ensure MCP standalone mode also bootstraps OTel

Update `src/codeintel/serving/mcp/server.py`:

```diff
--- src/codeintel/serving/mcp/server.py
+++ src/codeintel/serving/mcp/server.py
@@
 from codeintel.serving.settings import ServingSettings
+from codeintel.observability.otel import ObservabilityConfig, bootstrap_observability
@@
 def create_mcp_server(settings: ServingSettings) -> FastMCP:
+    bootstrap_observability(ObservabilityConfig.from_env(default_service_name="codeintel-serving"))
     app = build_mcp_app(settings)
     return FastMCP(name="codeintel", description="CodeIntel MCP Server", app=app)
```

---

## 5) Convert your existing `QueryMetrics` logs into real OTel metrics (zero callsite changes)

### Update `src/codeintel/serving/metrics.py`

Just add one line: `record_query_metrics(metrics)`.

```diff
--- src/codeintel/serving/metrics.py
+++ src/codeintel/serving/metrics.py
@@
 import logging
 from dataclasses import dataclass
+
+from codeintel.observability import record_query_metrics
@@
 def log_query_metrics(metrics: QueryMetrics) -> None:
+    # Emit OTel metrics + attach span attributes if tracing is active.
+    record_query_metrics(metrics)
     LOG.info(
         "query_metrics endpoint=%s view_id=%s duration_ms=%.1f row_count=%d truncated=%s correlation_id=%s engine=%s",
         metrics.endpoint,
         metrics.view_id,
         metrics.duration_ms,
         metrics.row_count,
         metrics.truncated,
         metrics.correlation_id,
         metrics.engine,
     )
```

Because **both HTTP and MCP already call** `log_query_metrics(...)`, this instantly makes query/export metrics symmetrical across transports.

---

## 6) What you get immediately (and what to validate)

### Tracing

* **HTTP**: request spans appear (FastAPI instrumentation), with child spans around your threadpool-executed ops (from `run_in_threadpool_with_metrics()`).
* **MCP**: a span per MCP tool call / message (middleware) + correlation ID.
* **CLI**: a span per `operation_id` executed by `@cli_command`.

### Metrics

* `codeintel.operation.*` metrics for CLI/HTTP/MCP uniformly.
* `codeintel.query.*` metrics for query-like actions uniformly.
* If you set `CODEINTEL_PROMETHEUS_METRICS=1`, you can scrape `/metrics` from FastAPI (and you’ll see the OTel-metered metrics because `PrometheusMetricReader` registers into the Prometheus client registry).

### Quick validation steps

* Run HTTP with:

  * `OTEL_EXPORTER_OTLP_ENDPOINT=...`
  * `OTEL_SERVICE_NAME=codeintel-serving`
  * `CODEINTEL_PROMETHEUS_METRICS=1`
* Hit `/v1/semantic/query` and an MCP tool via `/mcp` mount, verify:

  * Trace shows an HTTP span → child op span → MCP span (if MCP is hit)
  * Metrics show `codeintel.operation.calls` with `component=http|mcp`
  * Logs include correlation id and trace context (if you keep `include_trace=True` structured logs)

---

## 7) Optional “next 10% polish” (still using already-in-deps)

If you want to push this from “good” to “best-in-class” without adding new deps:

1. **Instrument outbound HTTP** (if you use `httpx` / `requests` anywhere) by calling:

   * `opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor().instrument()`
   * `opentelemetry.instrumentation.requests.RequestsInstrumentor().instrument()`
     inside `bootstrap_observability()` (guarded with try/except like the others).

2. **DuckDB DBAPI spans**: you already ship `opentelemetry-instrumentation-dbapi`. You can wrap DuckDB connections in `codeintel/storage/backend/duckdb_session.py` (in `_open_primary_connection`) so query execution is traced consistently. (I didn’t include this in the main patch because it’s slightly more nuanced, but it’s a very high ROI next step.)

3. **Make `/metrics` auth-aware** by reusing your `require_api_key` dependency if needed.

---

If you want, I can produce a **second diff-style patch set** that adds **DuckDB query spans** (DBAPI instrumentation) at the single connection-creation point (`DuckDBSession` / pool), so your traces show: HTTP/MCP span → semantic operation span → DuckDB span with query timing.

# DuckDB query spans diff-style patch #

Below is a concrete, “diff-style” patch set that adds **DuckDB query spans** at the **single connection-creation point** (`DuckDBSession.open/open_reader`) so your traces can naturally look like:

**HTTP/MCP span → semantic operation span → DuckDB (DB span w/ timing + SQL)**

### Why this patch uses `TracedCursorProxy` (not `instrument_connection`)

CodeIntel overwhelmingly uses `DuckDBPyConnection.execute(...)` / `executemany(...)` directly (not `cursor().execute(...)`). DuckDB’s Python API documents that `DuckDBPyConnection.cursor()` **creates a duplicate connection** (i.e., not a lightweight cursor) ([DuckDB][1]), while `execute`/`executemany` live directly on `DuckDBPyConnection` ([DuckDB][1]).

OpenTelemetry’s DB-API instrumentation creates spans by wrapping an object’s `execute`/`executemany` with a tracer (see `TracedCursorProxy.execute/executemany`) ([OpenTelemetry Python Contrib][2]). So we apply **that exact proxy** to the connection object itself, at the centralized creation point.

---

## Patch 1: Add a centralized DuckDB DBAPI tracing helper

```diff
diff --git a/src/codeintel/storage/backend/duckdb_tracing.py b/src/codeintel/storage/backend/duckdb_tracing.py
new file mode 100644
index 0000000..b7f9c2a
--- /dev/null
+++ b/src/codeintel/storage/backend/duckdb_tracing.py
@@
+"""OpenTelemetry DB-API spans for DuckDB connections.
+
+CodeIntel uses DuckDB via DuckDBPyConnection.execute/executemany directly.
+DuckDB's cursor() returns a *duplicate connection* (not a lightweight cursor),
+so "wrap connection.cursor()" style DB-API instrumentation won't capture most
+queries in this codebase.
+
+Instead, we wrap the DuckDBPyConnection object itself using OpenTelemetry's
+DB-API TracedCursorProxy, which instruments execute/executemany.
+"""
+
+from __future__ import annotations
+
+import os
+from typing import TYPE_CHECKING, cast
+
+import duckdb
+
+if TYPE_CHECKING:
+    from codeintel.storage.gateway.config import StorageConfig
+    from codeintel.storage.gateway.protocol import DuckDBConnection
+
+__all__ = ["maybe_instrument_duckdb_connection"]
+
+_OTEL_DUCKDB_SPANS_ENV = "CODEINTEL_OTEL_DUCKDB_SPANS"
+_OTEL_CAPTURE_PARAMETERS_ENV = "CODEINTEL_OTEL_DUCKDB_CAPTURE_PARAMETERS"
+_OTEL_SQLCOMMENTER_ENV = "CODEINTEL_OTEL_DUCKDB_SQLCOMMENTER"
+_OTEL_SQLCOMMENTER_ATTR_ENV = "CODEINTEL_OTEL_DUCKDB_SQLCOMMENTER_ATTRIBUTES"
+
+
+def _env_truthy(name: str, *, default: bool = False) -> bool:
+    raw = os.environ.get(name)
+    if raw is None:
+        return default
+    normalized = raw.strip().lower()
+    if normalized in {"1", "true", "yes", "on"}:
+        return True
+    if normalized in {"0", "false", "no", "off"}:
+        return False
+    return default
+
+
+def _otel_sdk_enabled() -> bool:
+    # Match your existing CLI convention: OTEL_SDK_DISABLED=true disables tracing.
+    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() != "true"
+
+
+def maybe_instrument_duckdb_connection(
+    con: "DuckDBConnection",
+    *,
+    config: "StorageConfig",
+) -> "DuckDBConnection":
+    """Return a DuckDB connection that emits OpenTelemetry DB spans when enabled.
+
+    Safe to call even when OpenTelemetry isn't installed/configured; it will
+    return the original connection unchanged.
+    """
+    if not _otel_sdk_enabled():
+        return con
+    if not _env_truthy(_OTEL_DUCKDB_SPANS_ENV, default=True):
+        return con
+
+    try:
+        from opentelemetry.instrumentation.dbapi import (
+            DatabaseApiIntegration,
+            TracedCursorProxy,
+            get_traced_cursor_proxy,
+        )
+    except Exception:
+        # Keep storage usable even if OTEL deps are not present.
+        return con
+
+    # Avoid double-wrapping (e.g., if a future refactor calls this twice).
+    if isinstance(con, TracedCursorProxy):
+        return con
+
+    integration = DatabaseApiIntegration(
+        name="codeintel.storage.duckdb",
+        database_system="duckdb",
+        capture_parameters=_env_truthy(_OTEL_CAPTURE_PARAMETERS_ENV, default=False),
+        enable_commenter=_env_truthy(_OTEL_SQLCOMMENTER_ENV, default=False),
+        enable_attribute_commenter=_env_truthy(_OTEL_SQLCOMMENTER_ATTR_ENV, default=False),
+        connect_module=duckdb,
+    )
+
+    # Populate anything we can from the connection, then pin db.name to the db_path.
+    integration.get_connection_attributes(con)
+    integration.database = str(config.db_path)
+
+    # Add CodeIntel context (super useful in multi-snapshot / multi-repo deployments).
+    if config.repo:
+        integration.span_attributes["codeintel.repo"] = config.repo
+    if config.commit:
+        integration.span_attributes["codeintel.commit"] = config.commit
+    integration.span_attributes["codeintel.storage.read_only"] = config.read_only
+
+    # Wrap the *connection* object so con.execute()/executemany() create DB spans.
+    return cast("DuckDBConnection", get_traced_cursor_proxy(con, integration))
```

What this buys you:

* **Every** `con.execute(...)` / `con.executemany(...)` in the codebase becomes a DB span automatically, with timing and statement attributes, using OTel’s DB-API conventions ([OpenTelemetry Python Contrib][2]).
* Instrumentation is centralized and **opt-out** via `OTEL_SDK_DISABLED=true` or `CODEINTEL_OTEL_DUCKDB_SPANS=false`.

---

## Patch 2: Hook it into the single canonical connection creation point

```diff
diff --git a/src/codeintel/storage/backend/duckdb_session.py b/src/codeintel/storage/backend/duckdb_session.py
index 4d2d9ef..4e9bb0f 100644
--- a/src/codeintel/storage/backend/duckdb_session.py
+++ b/src/codeintel/storage/backend/duckdb_session.py
@@
 import duckdb
 
+from codeintel.storage.backend.duckdb_tracing import maybe_instrument_duckdb_connection
 from codeintel.storage.gateway.extensions import (
     load_extensions_from_env,
     load_required_extensions,
 )
@@ class DuckDBSession:
     def open(self) -> DuckDBConnection:
@@
         _bootstrap_duckdb_secrets_from_env(con)
         _register_fsspec_filesystems_from_env()
         _run_init_sql_from_env(con)
-        return con
+        return maybe_instrument_duckdb_connection(con, config=self.config)
@@
     def open_reader(self) -> DuckDBConnection:
@@
         _bootstrap_duckdb_secrets_from_env(con)
         _register_fsspec_filesystems_from_env()
         _run_init_sql_from_env(con)
-        return con
+        return maybe_instrument_duckdb_connection(con, config=cfg)
```

Note: I’m applying instrumentation **at the end** of bootstrapping so the “startup noise” (PRAGMAs, extension loads, schema DDL) doesn’t show up as orphan traces when pooled reader connections are created at process start. You still get spans for the real serving/ingestion queries.

---

## Patch 3: Add a focused test (optional but recommended)

```diff
diff --git a/tests/observability/test_duckdb_dbapi_spans.py b/tests/observability/test_duckdb_dbapi_spans.py
new file mode 100644
index 0000000..2b3ad61
--- /dev/null
+++ b/tests/observability/test_duckdb_dbapi_spans.py
@@
+from __future__ import annotations
+
+from pathlib import Path
+
+from opentelemetry import trace
+from opentelemetry.sdk.trace import TracerProvider
+from opentelemetry.sdk.trace.export import SimpleSpanProcessor
+from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
+
+from codeintel.storage.backend.duckdb_session import DuckDBSession
+from codeintel.storage.gateway.config import StorageConfig
+
+
+def test_duckdb_execute_emits_db_spans(monkeypatch) -> None:
+    monkeypatch.setenv("CODEINTEL_OTEL_DUCKDB_SPANS", "true")
+    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
+
+    exporter = InMemorySpanExporter()
+
+    # If a provider already exists, attach to it; otherwise install one.
+    provider = trace.get_tracer_provider()
+    if not isinstance(provider, TracerProvider):
+        provider = TracerProvider()
+        trace.set_tracer_provider(provider)
+    provider.add_span_processor(SimpleSpanProcessor(exporter))
+
+    session = DuckDBSession(
+        StorageConfig(db_path=Path(":memory:"), repo="test-repo", commit="abc123")
+    )
+    con = session.open()
+
+    tracer = trace.get_tracer(__name__)
+    with tracer.start_as_current_span("parent") as parent:
+        parent_id = parent.get_span_context().span_id
+        con.execute("SELECT 1").fetchone()
+
+    spans = exporter.get_finished_spans()
+    db_spans = [s for s in spans if s.attributes.get("db.system") == "duckdb"]
+    assert db_spans, "expected at least one DuckDB DB span"
+
+    span = db_spans[-1]
+    assert span.parent is not None
+    assert span.parent.span_id == parent_id
+    assert span.attributes.get("db.name") == ":memory:"
+    assert "SELECT 1" in str(span.attributes.get("db.statement"))
+    assert span.attributes.get("codeintel.repo") == "test-repo"
+    assert span.attributes.get("codeintel.commit") == "abc123"
+    assert span.attributes.get("codeintel.storage.read_only") is False
```

---

## Expected behavior in traces

Once your HTTP/MCP layer has an active span (from your earlier “symmetrical tracing” changes), each DuckDB call will show as a nested **CLIENT** span with:

* `db.system = "duckdb"` ([OpenTelemetry Python Contrib][2])
* `db.name = "<db_path>"` (we set it from `StorageConfig.db_path`)
* `db.statement = "<SQL>"` ([OpenTelemetry Python Contrib][2])
* `codeintel.repo`, `codeintel.commit`, `codeintel.storage.read_only` (custom attrs)

Span name will typically be the SQL operation (`SELECT`, `INSERT`, `CREATE`, …) because the DBAPI instrumentation extracts the first token from the statement ([OpenTelemetry Python Contrib][2]).

---

## Runtime toggles

No new dependencies (you already ship the OTel packages). Runtime controls:

* Disable all tracing: `OTEL_SDK_DISABLED=true` (consistent with your CLI telemetry behavior)
* Disable *just* DuckDB DB spans: `CODEINTEL_OTEL_DUCKDB_SPANS=false`
* (Optional) capture SQL parameters: `CODEINTEL_OTEL_DUCKDB_CAPTURE_PARAMETERS=true`

  * I left this **off by default** because it can leak sensitive values.
* (Optional) sqlcommenter: `CODEINTEL_OTEL_DUCKDB_SQLCOMMENTER=true`

  * Mostly useful if you log SQL statements elsewhere; for embedded DuckDB it’s not required for propagation.

---

## One important caveat

Because DuckDB’s `cursor()` is a “duplicate connection” API ([DuckDB][1]), **don’t** refactor CodeIntel to use `cursor().execute(...)` purely for tracing—you’d risk executing on a different connection and breaking transactional expectations. This patch avoids that by instrumenting the connection’s `execute/executemany` directly via OTel’s DB-API proxy ([OpenTelemetry Python Contrib][2]).

---

If you want the next incremental patch after this: I can layer in a **statement redaction hook** (e.g., hash SQL + keep operation name) so you can keep timings without emitting full SQL text to your tracing backend—useful if any queries can include user-controlled strings.

[1]: https://duckdb.org/docs/stable/clients/python/reference/ "Python Client API – DuckDB"
[2]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/_modules/opentelemetry/instrumentation/dbapi.html "opentelemetry.instrumentation.dbapi — OpenTelemetry Python Contrib  documentation"

# statement redaction hook detailed implementation #

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
