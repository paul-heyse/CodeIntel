
Absolutely — here’s a concrete, “go-forward” redesign that does exactly what you asked:

* **Build side** publishes an immutable **read-only serving snapshot** (DuckDB file + manifest + semantic registry)
* **Serving side** uses a **ServingDBManager** that:

  * opens **read-only** connections to the current snapshot
  * supports **hot-swap** when `current.json` changes
  * exposes a small “semantic query kernel” that drives **both FastMCP + FastAPI**

And for deployment:

* **Local MCP**: `mcp.run()` (STDIO default) ([FastMCP][1])
* **Remote MCP over HTTP**: `mcp.run(transport="http", host=..., port=...)` (FastMCP runs an HTTP server; MCP endpoint at `/mcp`) ([FastMCP][1])
* **Unified FastAPI + MCP**: mount FastMCP ASGI app into FastAPI via `mcp.http_app(...)` and pass lifespan correctly ([FastMCP][2])

Below I’ll lay out:

1. the **new serving architecture** (files + responsibilities)
2. the **publisher** (build/)
3. the **ServingDBManager** (serving/)
4. the **Semantic registry + query kernel**
5. **FastMCP + FastAPI integration** (local + remote)

---

## 1) Proposed new serving architecture under `src/codeintel/serving`

The goal is to make `serving/` *tiny*, stable, and independent of legacy dataset-contract plumbing (which PR‑73 is removing anyway). Serving should know only:

* **where the current snapshot is**
* **what semantic views exist**
* **how to query them safely**
* **how to expose the tools via MCP/HTTP**

### New/updated tree (aggressive simplification)

```text
src/codeintel/build/serving/
  __init__.py
  publisher.py                 # checkpoint + copy + atomic publish + retention
  serving_manifest.py          # dataclasses for current.json

src/codeintel/serving/
  __init__.py
  settings.py                  # env-driven serving settings
  db_manager.py                # ServingDBManager (read-only snapshot + hot swap)
  kernel.py                    # SemanticQueryKernel (catalog/describe/query)
  semantic/
    __init__.py
    registry.py                # loads semantic_registry.json (or from DB)
    models.py                  # SemanticViewSpec, FilterSpec, QuerySpec, QueryResult
  mcp/
    __init__.py
    app.py                     # build FastMCP instance from kernel
    tools.py                   # the actual tools: catalog/describe/query
    server.py                  # entrypoint: stdio or http transport
  http/
    __init__.py
    app.py                     # FastAPI app, mounts MCP app, adds /health etc.
```

> You can keep your existing `serving/backend`, `serving/services`, `serving/operations` around temporarily, but the point of this redesign is: **the new kernel replaces most of that** once semantic views are the primary API.

---

## 2) Build-side publisher: checkpoint + immutable snapshot + atomic “current”

### Serving snapshot layout on disk

Pick a serving root directory (configurable):

```text
<serve_dir>/
  current.json                         # atomic pointer to current snapshot
  snapshots/
    <run_id>/
      codeintel.duckdb                 # immutable snapshot DB
      semantic_registry.json           # semantic views catalog (DAG-derived)
      schema_manifest.json             # (optional) full schema manifest
      build_spec.json                  # (optional) build DAG spec
```

### `ServingSnapshotManifest`

```python
# src/codeintel/build/serving/serving_manifest.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

@dataclass(frozen=True)
class ServingSnapshotManifest:
    run_id: str
    repo: str
    commit: str
    created_at: str  # ISO timestamp
    db_path: str
    semantic_registry_path: str
    schema_manifest_path: str | None = None
    build_spec_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_path(cls, path: Path) -> "ServingSnapshotManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
```

### Publisher implementation

This is designed to be called at the end of a successful build run (after PR‑73: you’ll already have schema manifest + semantic registry generation available).

```python
# src/codeintel/build/serving/publisher.py
from __future__ import annotations

import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from codeintel.build.serving.serving_manifest import ServingSnapshotManifest
from codeintel.storage.gateway.protocol import StorageGateway

def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

def _try_symlink(link: Path, target: Path) -> bool:
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        return True
    except OSError:
        return False

def publish_serving_snapshot(
    *,
    gateway: StorageGateway,          # writer gateway used during build
    run_id: str,
    serve_dir: Path,
    semantic_registry_path: Path,     # file produced by DAG compile step
    schema_manifest_path: Path | None = None,
    build_spec_path: Path | None = None,
    keep_last: int = 10,
) -> ServingSnapshotManifest:
    """
    Publish an immutable read-only serving snapshot.

    Steps:
    - CHECKPOINT to flush WAL into the DB file
    - copy DB file to snapshots/<run_id>/codeintel.duckdb
    - copy semantic_registry + optional manifests
    - atomically update serve_dir/current.json to point at the new snapshot
    """
    db_path = gateway.config.db_path
    if not db_path.is_file():
        raise FileNotFoundError(f"Build DB not found: {db_path}")

    # 1) Ensure DB file is consistent on disk
    gateway.con.execute("CHECKPOINT")  # flush WAL
    gateway.con.commit()

    snap_dir = serve_dir / "snapshots" / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 2) Copy DB
    snap_db = snap_dir / "codeintel.duckdb"
    shutil.copy2(db_path, snap_db)

    # 3) Copy registries/manifests (these should be deterministic build artifacts)
    snap_registry = snap_dir / "semantic_registry.json"
    shutil.copy2(semantic_registry_path, snap_registry)

    snap_schema_manifest = None
    if schema_manifest_path is not None:
        snap_schema_manifest = snap_dir / "schema_manifest.json"
        shutil.copy2(schema_manifest_path, snap_schema_manifest)

    snap_build_spec = None
    if build_spec_path is not None:
        snap_build_spec = snap_dir / "build_spec.json"
        shutil.copy2(build_spec_path, snap_build_spec)

    manifest = ServingSnapshotManifest(
        run_id=run_id,
        repo=gateway.config.repo or "unknown",
        commit=gateway.config.commit or "unknown",
        created_at=datetime.now(timezone.utc).isoformat(),
        db_path=str(snap_db),
        semantic_registry_path=str(snap_registry),
        schema_manifest_path=str(snap_schema_manifest) if snap_schema_manifest else None,
        build_spec_path=str(snap_build_spec) if snap_build_spec else None,
    )

    # 4) Atomic publish pointer
    current_path = serve_dir / "current.json"
    _atomic_write_text(current_path, manifest.to_json())

    # Optional convenience: current.duckdb symlink
    current_db_link = serve_dir / "current.duckdb"
    if not _try_symlink(current_db_link, snap_db):
        # fallback: just copy a small file "current_db_path.txt"
        _atomic_write_text(serve_dir / "current_db_path.txt", str(snap_db))

    # 5) Retention (delete older snapshots)
    if keep_last > 0:
        snaps_root = serve_dir / "snapshots"
        if snaps_root.exists():
            dirs = sorted([p for p in snaps_root.iterdir() if p.is_dir()],
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)
            for old in dirs[keep_last:]:
                shutil.rmtree(old, ignore_errors=True)

    return manifest
```

This publisher becomes your one “promotion step” from **build DB** → **serving DB**.

---

## 3) ServingDBManager: read-only snapshot + hot swap

This is the heart of the serving redesign. It:

* reads `<serve_dir>/current.json`
* opens read-only DuckDB connections to that snapshot
* optionally hot-swaps if `current.json` changes (either per-request check or timed)

### Settings

```python
# src/codeintel/serving/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class ServingSettings:
    serve_dir: Path
    hot_swap: bool = True
    pool_size: int = 4

    mcp_transport: str = "stdio"  # "stdio" or "http"
    host: str = "127.0.0.1"
    port: int = 8000

    auth_token: str | None = None  # for remote serving if you want a bearer token gate

    @classmethod
    def from_env(cls) -> "ServingSettings":
        serve_dir = Path(os.environ.get("CODEINTEL_SERVE_DIR", ".codeintel/serve")).resolve()
        hot_swap = os.environ.get("CODEINTEL_SERVE_HOTSWAP", "1") == "1"
        pool_size = int(os.environ.get("CODEINTEL_SERVE_POOL_SIZE", "4"))
        transport = os.environ.get("CODEINTEL_MCP_TRANSPORT", "stdio")
        host = os.environ.get("CODEINTEL_HOST", "127.0.0.1")
        port = int(os.environ.get("CODEINTEL_PORT", "8000"))
        token = os.environ.get("CODEINTEL_AUTH_TOKEN")
        return cls(
            serve_dir=serve_dir,
            hot_swap=hot_swap,
            pool_size=pool_size,
            mcp_transport=transport,
            host=host,
            port=port,
            auth_token=token,
        )
```

### DB manager (simple + reliable: per-request connection)

For your scale and “hardness” goals, I recommend **starting with “connection per request”** because it avoids thread-safety complexity. You can add pooling later.

```python
# src/codeintel/serving/db_manager.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import duckdb

from codeintel.build.serving.serving_manifest import ServingSnapshotManifest

@dataclass(frozen=True)
class ServingSnapshot:
    manifest: ServingSnapshotManifest
    manifest_path: Path

    @property
    def db_path(self) -> Path:
        return Path(self.manifest.db_path)

    @property
    def repo(self) -> str:
        return self.manifest.repo

    @property
    def commit(self) -> str:
        return self.manifest.commit

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


class ServingDBManager:
    def __init__(self, *, serve_dir: Path, hot_swap: bool = True) -> None:
        self._serve_dir = serve_dir
        self._manifest_path = serve_dir / "current.json"
        self._hot_swap = hot_swap
        self._cached_mtime_ns: int | None = None
        self._snapshot: ServingSnapshot | None = None

    def load_current(self) -> ServingSnapshot:
        if not self._manifest_path.is_file():
            raise FileNotFoundError(f"Missing serving manifest: {self._manifest_path}")

        mtime = self._manifest_path.stat().st_mtime_ns
        if self._snapshot is None or (self._hot_swap and mtime != self._cached_mtime_ns):
            manifest = ServingSnapshotManifest.from_path(self._manifest_path)
            self._snapshot = ServingSnapshot(manifest=manifest, manifest_path=self._manifest_path)
            self._cached_mtime_ns = mtime

        return self._snapshot

    @contextmanager
    def connect(self):
        """
        Open a read-only connection to the current snapshot.

        If hot_swap is enabled, this will pick up new snapshots automatically
        when current.json changes.
        """
        snap = self.load_current()
        con = duckdb.connect(str(snap.db_path), read_only=True)
        try:
            yield con, snap
        finally:
            con.close()
```

> Later, you can upgrade `connect()` into a small pool keyed by `snap.db_path`.
> But this version is extremely robust and already supports hot swap.

---

## 4) Semantic registry + query kernel (the unifying “semantic layer API”)

### Semantic registry format

The registry should be generated from the Hamilton DAG (post‑PR73: you’ll have a canonical schema provider; semantic views are just another set of materialized assets).

Example `semantic_registry.json`:

```json
{
  "version": "v1",
  "views": [
    {
      "id": "functions.summary",
      "table": "docs.v_function_summary",
      "description": "Summarized per-function metadata for agent inspection",
      "primary_key": ["goid_h128"],
      "columns": ["goid_h128", "module", "qualname", "risk_score", "is_exported"]
    }
  ]
}
```

### Serving-side registry loader

```python
# src/codeintel/serving/semantic/registry.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class SemanticViewSpec:
    id: str
    table: str                 # "schema.table" in DuckDB
    description: str | None
    primary_key: tuple[str, ...]
    columns: tuple[str, ...]

@dataclass(frozen=True)
class SemanticRegistry:
    version: str
    views: tuple[SemanticViewSpec, ...]

    @classmethod
    def load(cls, path: Path) -> "SemanticRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        views = []
        for v in payload["views"]:
            views.append(
                SemanticViewSpec(
                    id=v["id"],
                    table=v["table"],
                    description=v.get("description"),
                    primary_key=tuple(v.get("primary_key", [])),
                    columns=tuple(v.get("columns", [])),
                )
            )
        return cls(version=payload.get("version", "v1"), views=tuple(views))

    def by_id(self, view_id: str) -> SemanticViewSpec:
        for v in self.views:
            if v.id == view_id:
                return v
        raise KeyError(view_id)
```

### Query kernel: catalog/describe/query (no raw SQL accepted from agent)

This kernel is the single “business logic” interface used by both MCP and HTTP.

```python
# src/codeintel/serving/kernel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.serving.db_manager import ServingDBManager
from codeintel.serving.semantic.registry import SemanticRegistry, SemanticViewSpec

@dataclass(frozen=True)
class QueryResult:
    view_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool
    snapshot: dict[str, str]

class SemanticQueryKernel:
    def __init__(self, *, db: ServingDBManager) -> None:
        self._db = db

    def catalog(self) -> dict[str, Any]:
        # registry is stored next to the DB snapshot; load it via manifest pointer
        with self._db.connect() as (con, snap):
            registry_path = snap.manifest.semantic_registry_path
            reg = SemanticRegistry.load(Path(registry_path))
            return {
                "version": reg.version,
                "snapshot": {"repo": snap.repo, "commit": snap.commit, "run_id": snap.run_id},
                "views": [
                    {
                        "id": v.id,
                        "table": v.table,
                        "description": v.description,
                        "primary_key": list(v.primary_key),
                        "columns": list(v.columns),
                    }
                    for v in reg.views
                ],
            }

    def describe(self, view_id: str) -> dict[str, Any]:
        with self._db.connect() as (_con, snap):
            reg = SemanticRegistry.load(Path(snap.manifest.semantic_registry_path))
            v = reg.by_id(view_id)
            return {
                "id": v.id,
                "table": v.table,
                "description": v.description,
                "primary_key": list(v.primary_key),
                "columns": list(v.columns),
                "snapshot": {"repo": snap.repo, "commit": snap.commit, "run_id": snap.run_id},
            }

    def query_view(
        self,
        *,
        view_id: str,
        filters: list[dict[str, Any]] | None = None,  # simple structured filters
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> QueryResult:
        from pathlib import Path

        filters = filters or []
        order_by = order_by or []

        with self._db.connect() as (con, snap):
            reg = SemanticRegistry.load(Path(snap.manifest.semantic_registry_path))
            view = reg.by_id(view_id)

            cols = list(select) if select else list(view.columns)
            for c in cols:
                if c not in view.columns:
                    raise ValueError(f"Unknown column {c!r} for view {view_id!r}")

            schema, table = view.table.split(".", 1)
            quoted_table = f'"{schema}"."{table}"'
            quoted_cols = ", ".join([f'"{c}"' for c in cols])

            where_sql, params = _compile_filters(filters, allowed_cols=set(view.columns))

            order_sql = ""
            if order_by:
                for c in order_by:
                    if c.lstrip("-") not in view.columns:
                        raise ValueError(f"Unknown order_by column {c!r}")
                parts = []
                for c in order_by:
                    if c.startswith("-"):
                        parts.append(f'"{c[1:]}" DESC')
                    else:
                        parts.append(f'"{c}" ASC')
                order_sql = " ORDER BY " + ", ".join(parts)

            sql = (
                f"SELECT {quoted_cols} FROM {quoted_table}"
                + (f" WHERE {where_sql}" if where_sql else "")
                + order_sql
                + " LIMIT ? OFFSET ?"
            )
            params = [*params, int(limit), int(offset)]

            arrow = con.execute(sql, params).fetch_arrow_table()
            rows = arrow.to_pylist()  # list[dict]

            return QueryResult(
                view_id=view_id,
                columns=cols,
                rows=rows,
                truncated=len(rows) >= limit,
                snapshot={"repo": snap.repo, "commit": snap.commit, "run_id": snap.run_id},
            )


def _compile_filters(filters: list[dict[str, Any]], *, allowed_cols: set[str]) -> tuple[str, list[Any]]:
    """
    Very small structured filter compiler:
      [{"col":"module","op":"eq","value":"foo"}, {"col":"risk_score","op":"gte","value":0.8}]
    """
    clauses = []
    params: list[Any] = []
    for f in filters:
        col = f.get("col")
        op = f.get("op")
        value = f.get("value")
        if col not in allowed_cols:
            raise ValueError(f"Unknown filter column: {col!r}")

        # allowed ops (expand later)
        if op == "eq":
            clauses.append(f'"{col}" = ?')
            params.append(value)
        elif op == "gte":
            clauses.append(f'"{col}" >= ?')
            params.append(value)
        elif op == "lte":
            clauses.append(f'"{col}" <= ?')
            params.append(value)
        elif op == "in":
            if not isinstance(value, list):
                raise ValueError("in requires list value")
            placeholders = ", ".join(["?"] * len(value))
            clauses.append(f'"{col}" IN ({placeholders})')
            params.extend(value)
        else:
            raise ValueError(f"Unsupported op: {op!r}")

    return (" AND ".join(clauses), params)
```

This gives you a **strictly bounded query interface** that still supports a lot of expressive filtering, without ever accepting raw SQL or code from the agent.

---

## 5) FastMCP + FastAPI integration (local + remote)

FastMCP gives you:

* default `mcp.run()` (STDIO) ([FastMCP][1])
* HTTP transport with `mcp.run(transport="http", host=..., port=...)` and endpoint at `/mcp` ([FastMCP][1])
* the ability to mount FastMCP inside FastAPI via `mcp.http_app(...)` ([FastMCP][2])

### MCP app builder (tools wired to kernel)

```python
# src/codeintel/serving/mcp/app.py
from __future__ import annotations
from fastmcp import FastMCP

from codeintel.serving.kernel import SemanticQueryKernel

def build_mcp_app(*, kernel: SemanticQueryKernel) -> FastMCP:
    mcp = FastMCP("CodeIntel", json_response=True)

    @mcp.tool
    def semantic_catalog() -> dict:
        """List available semantic views (Hamilton-managed semantic layer)."""
        return kernel.catalog()

    @mcp.tool
    def semantic_describe(view_id: str) -> dict:
        """Describe a semantic view: columns, PK, description."""
        return kernel.describe(view_id)

    @mcp.tool
    def semantic_query(
        view_id: str,
        filters: list[dict] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        """Query a semantic view using structured filters (no SQL)."""
        result = kernel.query_view(
            view_id=view_id,
            filters=filters,
            select=select,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )
        return {
            "view_id": result.view_id,
            "columns": result.columns,
            "rows": result.rows,
            "truncated": result.truncated,
            "snapshot": result.snapshot,
        }

    return mcp
```

### MCP server entrypoint: stdio or http

```python
# src/codeintel/serving/mcp/server.py
from __future__ import annotations

from codeintel.serving.db_manager import ServingDBManager
from codeintel.serving.kernel import SemanticQueryKernel
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.settings import ServingSettings

def main() -> None:
    settings = ServingSettings.from_env()
    db = ServingDBManager(serve_dir=settings.serve_dir, hot_swap=settings.hot_swap)
    kernel = SemanticQueryKernel(db=db)
    mcp = build_mcp_app(kernel=kernel)

    # STDIO by default; run http for remote deployments
    if settings.mcp_transport == "http":
        # FastMCP HTTP transport over Streamable HTTP. :contentReference[oaicite:6]{index=6}
        mcp.run(transport="http", host=settings.host, port=settings.port)
    else:
        # STDIO default. :contentReference[oaicite:7]{index=7}
        mcp.run()
```

This matches your desire: **use the uvicorn functionality integrated into FastMCP** for a small remote deployment (i.e., `transport="http"`). ([FastMCP][1])

---

## Optional: a combined FastAPI app that mounts MCP (recommended if you also want /health, /metrics, auth)

FastMCP docs show mounting the MCP ASGI app using `http_app` and ensuring lifespan is passed. ([FastMCP][2])

```python
# src/codeintel/serving/http/app.py
from __future__ import annotations
from fastapi import FastAPI
from contextlib import asynccontextmanager

from codeintel.serving.settings import ServingSettings
from codeintel.serving.db_manager import ServingDBManager
from codeintel.serving.kernel import SemanticQueryKernel
from codeintel.serving.mcp.app import build_mcp_app

def create_app() -> FastAPI:
    settings = ServingSettings.from_env()

    db = ServingDBManager(serve_dir=settings.serve_dir, hot_swap=settings.hot_swap)
    kernel = SemanticQueryKernel(db=db)
    mcp = build_mcp_app(kernel=kernel)

    # Create MCP ASGI app at /mcp
    mcp_app = mcp.http_app(path="/mcp")  # returns ASGI app :contentReference[oaicite:10]{index=10}

    # If you need your own lifespan too, combine it with mcp_app.lifespan.
    app = FastAPI(title="CodeIntel Serving", lifespan=mcp_app.lifespan)  # :contentReference[oaicite:11]{index=11}

    @app.get("/health")
    def health() -> dict:
        # Simple check: can load manifest
        snap = db.load_current()
        return {"status": "ok", "repo": snap.repo, "commit": snap.commit, "run_id": snap.run_id}

    # Mount MCP under /mcp (or /agent, /tools, etc.)
    app.mount("", mcp_app)  # MCP endpoint will be /mcp :contentReference[oaicite:12]{index=12}

    return app

app = create_app()
```

Run with uvicorn:

```bash
uvicorn codeintel.serving.http.app:app --host 0.0.0.0 --port 8000
```

Or run MCP-only via FastMCP `mcp.run(transport="http")` as above. ([FastMCP][1])

---

## Hot swap behavior (what you’ll get “for free”)

With the design above:

* build publishes a new snapshot → replaces `<serve_dir>/current.json`
* serving on next request calls `load_current()` → sees `mtime` changed → uses new snapshot

So deployments are easy:

* you can update snapshots without restarting the server
* or for extra safety, you can still restart workers on publish

---

## How this “ties directly into your semantic view registry”

The key is: **semantic registry is a build artifact derived from the DAG**, then published alongside the snapshot DB.

That means:

* Serving tools are driven by `semantic_registry.json`
* Not by hardcoded operations catalogs
* And definitely not by `config/datasets`

This is the correct direction for PR‑73+.

---

## The one build-side prerequisite you’ll want post‑PR73

Make sure your build pipeline *materializes* (or creates) all semantic views/tables **before** publishing the snapshot, because your read-only serving snapshot cannot create new views.

FastMCP/HTTP + read-only DB then becomes extremely simple and safe.

---

If you want, next I can also propose (in the same style) the **exact “semantic layer” conventions** to encode in Hamilton tags so `semantic_registry.json` can be compiled automatically from the DAG (e.g., `output_kind=semantic`, `semantic_id=...`, `entity=function/module/...`, `grain=per_function/per_module/...`, etc.). That’s the piece that makes the semantic layer truly “Hamilton-native” and eliminates manual registry edits entirely.

[1]: https://gofastmcp.com/deployment/running-server "Running Your Server - FastMCP"
[2]: https://gofastmcp.com/integrations/fastapi "FastAPI  FastMCP - FastMCP"


# semantic layer conventions #

Below is a concrete, Hamilton-native “semantic layer” tagging convention that lets you **compile `semantic_registry.json` automatically from the DAG**—no hand-maintained registry files, and no drift between what’s served and what’s built.

I’ll give you:

1. The **tag taxonomy** (exact tag keys + expected values)
2. A **decorator helper** (`@semantic_view(...)`) that applies tags consistently
3. The **compiler** that scans the Hamilton graph + SchemaProvider and writes `semantic_registry.json` deterministically
4. A suggested **semantic ID & entity/grain convention** (so it scales)
5. Optional “best-in-class” extras: join hints, default ordering/limits, sensitivity, and deprecation

---

## 1) Semantic layer: the minimal tag contract

You already standardized baseline tags like:

* `domain`
* `target`
* `node_type` (e.g., `compute`, `materialize`, `loader.query`, …)
* often `table_key` for dataset nodes

Add the following semantic tags. The idea is:

> A semantic view is a **materialized** output node (table or view) with `output_kind="semantic"` and a stable `semantic_id`.

### Required tags

| Tag key           | Type | Example                     | Meaning                                        |
| ----------------- | ---: | --------------------------- | ---------------------------------------------- |
| `output_kind`     |  str | `"semantic"`                | Marks this as a semantic-layer output          |
| `semantic_id`     |  str | `"function.summary"`        | Stable API identifier (doesn’t change lightly) |
| `semantic_kind`   |  str | `"table"` or `"view"`       | What it is in DuckDB snapshot                  |
| `table_key`       |  str | `"docs.v_function_summary"` | DuckDB object to query                         |
| `semantic_entity` |  str | `"function"`                | Core entity the rows represent                 |
| `semantic_grain`  |  str | `"per_function"`            | Row granularity (“grain”)                      |

### Strongly recommended tags (for quality + usability)

| Tag key                     |       Type | Example                                    | Meaning                                                      |
| --------------------------- | ---------: | ------------------------------------------ | ------------------------------------------------------------ |
| `semantic_primary_key`      |    CSV str | `"goid_h128"`                              | Primary key columns                                          |
| `semantic_columns`          |    CSV str | `"goid_h128,module,qualname,risk_score"`   | Exposed columns (subset); default is “all columns in schema” |
| `semantic_description`      |        str | `"Per-function summary for agent queries"` | Human description                                            |
| `mcp_visible`               |  `"1"/"0"` | `"1"`                                      | Whether it is exposed to MCP tools                           |
| `semantic_default_order_by` |    CSV str | `"-risk_score,qualname"`                   | Default ordering for queries                                 |
| `semantic_default_limit`    | int as str | `"200"`                                    | Default limit when querying this view                        |

### Best-in-class optional tags (scale & governance)

| Tag key                |      Type | Example                                                | Meaning                                            |
| ---------------------- | --------: | ------------------------------------------------------ | -------------------------------------------------- |
| `semantic_joins`       |  JSON str | `[{"to":"module.summary","on":[["module","module"]]}]` | Join hints for agents (semantic-to-semantic joins) |
| `semantic_sensitivity` |       str | `"internal"` / `"public"` / `"secret"`                 | Controls exposure policies                         |
| `semantic_deprecated`  | `"1"/"0"` | `"0"`                                                  | Mark deprecated views                              |
| `semantic_replaced_by` |       str | `"function.summary_v2"`                                | Successor semantic ID                              |
| `semantic_examples`    |  JSON str | `[{...}]`                                              | Example queries (optional)                         |

This is enough to compile a registry that’s genuinely useful to agents and stable over time.

---

## 2) Naming conventions that keep things readable at scale

### `semantic_id`

Make it a stable dotted name:

* `<entity>.<name>` for simple sets

  * `function.summary`
  * `module.summary`
  * `symbol.definitions`
  * `edge.calls`
* `<entity>.<name>.<variant>` for variants

  * `function.summary.v1`
  * `function.summary.public`

Keep it **stable**. Treat it like a public API.

### `semantic_entity`

Use a controlled vocabulary so your catalog stays coherent:

Suggested entities (tailor to your dataset):

* `function`, `module`, `file`, `symbol`, `subsystem`
* `edge_call`, `edge_import`, `edge_export`
* `risk_factor`, `metric`, `test`, `coverage`
* `asset` (Phase 4 asset catalog views)

### `semantic_grain`

Make it explicit and consistent:

* `per_function`, `per_module`, `per_file`, `per_symbol`, `per_subsystem`
* `per_edge` (for edge tables)
* `per_test`, `per_run`, `per_asset`

Agents do much better when grain is explicit because it prevents category mistakes.

---

## 3) A Hamilton-native decorator to enforce tag consistency

Put this in something like:

`src/codeintel/build/hamilton/native/semantic_decorators.py`

```python
from __future__ import annotations

import json
from typing import Callable, Iterable, Sequence, TypeVar

from hamilton.function_modifiers import tag

F = TypeVar("F", bound=Callable[..., object])

def _csv(items: Sequence[str] | None) -> str | None:
    if not items:
        return None
    return ",".join(items)

def semantic_view(
    *,
    semantic_id: str,
    table_key: str,
    entity: str,
    grain: str,
    kind: str = "view",  # "table" or "view"
    primary_key: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    description: str | None = None,
    joins: list[dict] | None = None,  # semantic join hints
    mcp_visible: bool = True,
    sensitivity: str = "internal",  # internal|public|secret
    default_order_by: Sequence[str] | None = None,
    default_limit: int | None = 200,
    deprecated: bool = False,
    replaced_by: str | None = None,
) -> Callable[[F], F]:
    """
    Apply the canonical semantic-layer tags to a Hamilton node (usually a materializer).
    """
    tags: dict[str, str] = {
        "output_kind": "semantic",
        "semantic_id": semantic_id,
        "semantic_kind": kind,
        "table_key": table_key,
        "semantic_entity": entity,
        "semantic_grain": grain,
        "mcp_visible": "1" if mcp_visible else "0",
        "semantic_sensitivity": sensitivity,
        "semantic_deprecated": "1" if deprecated else "0",
    }

    if description:
        tags["semantic_description"] = description
    if primary_key:
        tags["semantic_primary_key"] = _csv(list(primary_key))  # type: ignore[arg-type]
    if columns:
        tags["semantic_columns"] = _csv(list(columns))  # type: ignore[arg-type]
    if joins:
        tags["semantic_joins"] = json.dumps(joins, sort_keys=True)
    if default_order_by:
        tags["semantic_default_order_by"] = _csv(list(default_order_by))  # type: ignore[arg-type]
    if default_limit is not None:
        tags["semantic_default_limit"] = str(int(default_limit))
    if replaced_by:
        tags["semantic_replaced_by"] = replaced_by

    def _wrap(fn: F) -> F:
        return tag(**tags)(fn)

    return _wrap
```

### Usage example (materializer node)

```python
from __future__ import annotations
import ibis.expr.types as ir

from codeintel.build.hamilton.native.semantic_decorators import semantic_view
from codeintel.build.hamilton.native.materializers import materialize_view  # your helper
from codeintel.build.hamilton.env import BuildEnv

@semantic_view(
    semantic_id="function.summary",
    table_key="docs.v_function_summary",
    kind="view",
    entity="function",
    grain="per_function",
    primary_key=("goid_h128",),
    description="Per-function semantic summary for agent queries",
    default_order_by=("-risk_score", "qualname"),
    default_limit=200,
    joins=[
        {"to": "module.summary", "on": [["module", "module"]]},
        {"to": "function.calls_out", "on": [["goid_h128", "caller_goid_h128"]]},
    ],
)
def t__semantic__function_summary(env: BuildEnv, q__analytics__function_metrics: ir.Table) -> object:
    expr = (
        q__analytics__function_metrics
        # ... build Ibis expr for view/table ...
    )
    return materialize_view(env, table_key="docs.v_function_summary", expr=expr)
```

The important part: **the tags fully describe how serving should expose this semantic view**.

---

## 4) Compile `semantic_registry.json` automatically from the DAG

### Registry schema (recommended)

Keep it stable and easy for agents:

```json
{
  "version": "v1",
  "views": [
    {
      "id": "function.summary",
      "kind": "view",
      "table": "docs.v_function_summary",
      "entity": "function",
      "grain": "per_function",
      "description": "...",
      "primary_key": ["goid_h128"],
      "columns": ["goid_h128", "module", "qualname", "risk_score"],
      "joins": [{"to":"module.summary","on":[["module","module"]]}],
      "defaults": {"limit": 200, "order_by": ["-risk_score", "qualname"]},
      "sensitivity": "internal",
      "deprecated": false,
      "replaced_by": null
    }
  ]
}
```

### Compiler implementation

Put this in:

`src/codeintel/build/serving/semantic_registry_compile.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.core.schemas.provider import SchemaProvider
from codeintel.build.hamilton.driver_factory import build_driver

# Tag keys
TAG_OUTPUT_KIND = "output_kind"
TAG_NODE_TYPE = "node_type"
TAG_TABLE_KEY = "table_key"
TAG_SEM_ID = "semantic_id"
TAG_SEM_KIND = "semantic_kind"
TAG_SEM_ENTITY = "semantic_entity"
TAG_SEM_GRAIN = "semantic_grain"
TAG_SEM_PK = "semantic_primary_key"
TAG_SEM_COLS = "semantic_columns"
TAG_SEM_DESC = "semantic_description"
TAG_SEM_JOINS = "semantic_joins"
TAG_MCP_VISIBLE = "mcp_visible"
TAG_DEFAULT_ORDER = "semantic_default_order_by"
TAG_DEFAULT_LIMIT = "semantic_default_limit"
TAG_SENSITIVITY = "semantic_sensitivity"
TAG_DEPRECATED = "semantic_deprecated"
TAG_REPLACED_BY = "semantic_replaced_by"

NODE_TYPE_MATERIALIZE = "materialize"
OUTPUT_KIND_SEMANTIC = "semantic"

def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_json(s: str | None) -> Any:
    if not s:
        return None
    return json.loads(s)

@dataclass(frozen=True)
class SemanticRegistry:
    version: str
    views: list[dict[str, Any]]

    def to_json_text(self) -> str:
        # Deterministic output: sort keys and ensure stable ordering.
        payload = {"version": self.version, "views": self.views}
        return json.dumps(payload, indent=2, sort_keys=True)

def compile_semantic_registry(
    *,
    schema_provider: SchemaProvider,
    version: str = "v1",
) -> SemanticRegistry:
    """
    Introspects the Hamilton graph and produces a semantic_registry.json payload.
    """
    rt = build_driver(mode="auto")
    views: list[dict[str, Any]] = []

    # NOTE: exact graph introspection APIs depend on your driver wrapper.
    # The idea: iterate nodes, read tags.
    for node_name, node in rt.dr.graph.nodes.items():  # adjust if your API differs
        tags = getattr(node, "tags", {}) or {}

        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_MATERIALIZE:
            continue
        if tags.get(TAG_OUTPUT_KIND) != OUTPUT_KIND_SEMANTIC:
            continue
        if tags.get(TAG_MCP_VISIBLE, "1") != "1":
            continue

        semantic_id = tags.get(TAG_SEM_ID)
        table_key = tags.get(TAG_TABLE_KEY)
        if not semantic_id or not table_key:
            # hard fail: semantic outputs must be fully specified
            raise ValueError(f"Semantic node {node_name} missing semantic_id/table_key")

        kind = tags.get(TAG_SEM_KIND, "view")
        entity = tags.get(TAG_SEM_ENTITY, "unknown")
        grain = tags.get(TAG_SEM_GRAIN, "unknown")

        # Resolve columns: explicit tag subset OR SchemaProvider (canonical)
        explicit_cols = _split_csv(tags.get(TAG_SEM_COLS))
        table_schema = schema_provider.require_table_schema(table_key)
        all_cols = [c.name for c in table_schema.columns]
        cols = explicit_cols if explicit_cols else all_cols

        # PK + joins + defaults
        pk = _split_csv(tags.get(TAG_SEM_PK))
        joins = _parse_json(tags.get(TAG_SEM_JOINS)) or []
        default_order = _split_csv(tags.get(TAG_DEFAULT_ORDER))
        default_limit = int(tags.get(TAG_DEFAULT_LIMIT, "200"))

        view_entry = {
            "id": semantic_id,
            "kind": kind,
            "table": table_key,
            "entity": entity,
            "grain": grain,
            "description": tags.get(TAG_SEM_DESC) or (getattr(node, "documentation", None) or None),
            "primary_key": pk,
            "columns": cols,
            "joins": joins,
            "defaults": {"limit": default_limit, "order_by": default_order},
            "sensitivity": tags.get(TAG_SENSITIVITY, "internal"),
            "deprecated": tags.get(TAG_DEPRECATED, "0") == "1",
            "replaced_by": tags.get(TAG_REPLACED_BY),
        }
        views.append(view_entry)

    # Deterministic ordering by semantic_id
    views.sort(key=lambda v: v["id"])

    return SemanticRegistry(version=version, views=views)

def write_semantic_registry(
    *,
    schema_provider: SchemaProvider,
    out_path: Path,
    version: str = "v1",
) -> None:
    reg = compile_semantic_registry(schema_provider=schema_provider, version=version)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(reg.to_json_text(), encoding="utf-8")
```

### Where to call it

You typically generate this as part of your build artifacts *before publishing*:

* After the semantic views/tables are materialized
* Before `publish_serving_snapshot(...)` copies artifacts into snapshot dir

So the publisher just copies the file.

---

## 5) “Semantic layer” conventions that prevent future drift

These are the rules I recommend enforcing with CI gate tests:

### Rule 1 — Every `output_kind="semantic"` node must declare `semantic_id` and `table_key`

No implicit naming. No best-effort guessing.

### Rule 2 — SchemaProvider must resolve every semantic `table_key`

If a semantic view is queryable, it must have a schema in the schema authority. This gives you hardness.

### Rule 3 — `semantic_id` is stable API

If you replace a view, don’t delete it silently:

* set `semantic_deprecated=1`
* set `semantic_replaced_by=<new id>`

### Rule 4 — Semantic views should live in one schema

E.g., `docs.*` (or `semantic.*`). This makes serving allowlists trivial.

### Rule 5 — Join hints should reference semantic IDs, not raw tables

Agents should learn your semantic surface, not internal storage tables.

---

## 6) Example semantic join hints for agent ergonomics

A join hint object like:

```json
{"to":"module.summary","on":[["module","module"]]}
```

Means:

* from this view’s column `module`
* join to `module.summary`’s column `module`

You can add richer hints later:

```json
{
  "to": "function.calls_out",
  "on": [["goid_h128","caller_goid_h128"]],
  "kind": "left",
  "note": "Outgoing call edges from this function"
}
```

---

## 7) What to expose first: a “starter pack” semantic surface

To make the MCP service immediately valuable to agents, I’d expose a small set of semantic views with clear grains:

* `function.summary` (per_function)
* `module.summary` (per_module)
* `function.calls_out` (per_edge_call)
* `module.depends_on` (per_edge_import or per_edge_call aggregated)
* `module.exports` (per_module or per_export_target)
* `risk_factors.by_function` (per_function)
* `tests.coverage_by_function` (per_function)

Keep it under ~15 views initially; expand once you see real agent query patterns.

---

## 8) CI tests and snapshot suggestion

Once you implement the compiler, add:

* `tests/build/hamilton/test_semantic_registry_compile.py`

  * compiles registry
  * asserts deterministic ordering
  * asserts all entries have required fields
  * asserts SchemaProvider resolves columns

And add a CLI snapshot (optional) that prints the registry:

* `codeintel serving semantic-registry --format json`
* snapshot: `semantic_registry.json`

This makes semantic drift reviewable, just like schema drift.

---

If you want next: I can propose a **precise tag taxonomy** for “entity keys” and “join keys” (e.g., standard columns like `repo`, `commit`, `module`, `goid_h128`, `urn`) and a recommended set of **semantic schemas** (`docs.*` vs `semantic.*`), so all your semantic views look and feel uniform to agents from day one.
