
Below is a detailed implementation plan to normalize DB usage around `StorageGateway`, with concrete code snippets and how it touches:

* `analytics.function_history`
* `analytics.history_timeseries`
* `storage.gateway`
* CLI + tests

The goal is:

> All analytics entrypoints talk to **DuckDB via `StorageGateway`**, not raw `duckdb.DuckDBPyConnection`, and the only place you configure `read_only/apply_schema/ensure_views/validate_schema` is `StorageConfig` + `open_gateway` .

---

## 1. Current state (why we’re doing this)

Most analytics already do the “right thing”:

* They accept `gateway: StorageGateway` and do `con = gateway.con` before running SQL.
  Example: coverage analytics uses `ensure_schema` + `DELETE` + `INSERT…SELECT` with `gateway.con` .
* Ingestion and analytics steps all carry a `PipelineContext` that has a `gateway: StorageGateway` field and pass that to analytics functions .

Two places still operate on **bare connections**:

1. `analytics.function_history.compute_function_history`:

   ```python
   def compute_function_history(
       con: duckdb.DuckDBPyConnection,
       cfg: FunctionHistoryConfig,
       *,
       runner: ToolRunner | None = None,
       context: AnalyticsContext | None = None,
   ) -> None:
       ...
       ensure_schema(con, "analytics.function_history")
       con.execute("DELETE FROM analytics.function_history WHERE repo = ? AND commit = ?", [cfg.repo, cfg.commit])
       ...
       spans_by_path = _load_function_spans(con, cfg.repo, cfg.commit)
       ...
       con.executemany("INSERT INTO analytics.function_history (...)", insert_rows)
   :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
   ```

2. `analytics.history_timeseries.compute_history_timeseries`:

   ```python
   DBResolver = Callable[[str], duckdb.DuckDBPyConnection]

   def compute_history_timeseries(
       history_con: duckdb.DuckDBPyConnection,
       cfg: HistoryTimeseriesConfig,
       db_resolver: DBResolver,
       *,
       runner: ToolRunner | None = None,
   ) -> None:
       ensure_schema(history_con, "analytics.history_timeseries")
       history_con.execute("DELETE FROM analytics.history_timeseries WHERE repo = ?", [cfg.repo])
       selection = _select_entities(cfg, db_resolver)
       ...
       for commit in cfg.commits:
           con_ci = db_resolver(commit)
           ...
           rows.extend(_collect_function_rows_for_commit(cfg, con_ci, commit_ctx=commit_ctx, selection=selection.functions))
           ...
       history_con.executemany("INSERT INTO analytics.history_timeseries (...)", rows)
   :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
   ```

The CLI and tests mirror that connection-based style:

* CLI’s `history-timeseries` subcommand builds a write `StorageGateway` but passes `gateway.con` + a raw `duckdb.connect` resolver into `compute_history_timeseries` .
* `tests/analytics/test_history_timeseries.py` uses `duckdb.connect` directly in `_resolve_db` and passes `gateway.con` for the output DB .
* `tests/analytics/test_function_history.py` uses a `StorageGateway` fixture but still calls `compute_function_history(con, cfg, runner=runner)` with `con = gateway.con` .

We’re going to:

1. Make **`StorageGateway` the public boundary** for `function_history` and `history_timeseries`.
2. Keep an internal connection-based core for `history_timeseries` so you don’t duplicate logic.
3. Centralize “open snapshot DB for commit” in `storage.gateway`.

---

## 2. Part A — Refactor `function_history` to accept `StorageGateway`

### 2.1. Change the public signature

Today:

```python
def compute_function_history(
    con: duckdb.DuckDBPyConnection,
    cfg: FunctionHistoryConfig,
    *,
    runner: ToolRunner | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    ...
:contentReference[oaicite:10]{index=10}
```

We want:

```python
from codeintel.storage.gateway import StorageGateway

def compute_function_history(
    gateway: StorageGateway,
    cfg: FunctionHistoryConfig,
    *,
    runner: ToolRunner | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """
    Populate `analytics.function_history` for the given repo/commit snapshot.

    Parameters
    ----------
    gateway:
        StorageGateway bound to the CodeIntel DuckDB database.
    cfg:
        Function history configuration.
    runner:
        Optional shared ToolRunner for git invocations.
    context:
        Optional shared analytics context to enforce snapshot consistency.
    """
    con = gateway.con

    if context is not None and (context.repo != cfg.repo or context.commit != cfg.commit):
        log.warning(
            "function_history context mismatch: context=%s@%s cfg=%s@%s",
            context.repo,
            context.commit,
            cfg.repo,
            cfg.commit,
        )

    ensure_schema(con, "analytics.function_history")
    con.execute(
        "DELETE FROM analytics.function_history WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    spans_by_path = _load_function_spans(con, cfg.repo, cfg.commit)
    ...
:contentReference[oaicite:11]{index=11}
```

Changes inside the module:

* Import `StorageGateway` instead of `duckdb.DuckDBPyConnection` (you can drop the direct duckdb import if `_load_function_spans` still uses it for typing).
* Take `gateway` as the first argument; call `con = gateway.con` and reuse the rest of the logic unchanged.

### 2.2. Optional: keep a tiny compatibility shim (if you care)

If you want to maintain the old API for a while:

```python
import duckdb

def compute_function_history_with_con(
    con: duckdb.DuckDBPyConnection,
    cfg: FunctionHistoryConfig,
    *,
    runner: ToolRunner | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """Compatibility wrapper around the StorageGateway-based implementation."""
    from codeintel.storage.gateway import StorageConfig, open_gateway

    gateway = open_gateway(
        StorageConfig(
            db_path=Path(con.database_name),  # or use a separate helper if this is awkward
            read_only=False,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
    )
    compute_function_history(gateway, cfg, runner=runner, context=context)
```

But realistically, you control all call sites and can just update them, so the shim is optional.

### 2.3. Update pipeline steps (if needed)

`steps.py` already imports `compute_function_history` . Somewhere lower in the file you’ll have a step like:

```python
@dataclass
class FunctionHistoryStep:
    name: str = "function_history"
    deps: Sequence[str] = ("function_analytics",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        con = ctx.gateway.con
        cfg = FunctionHistoryConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_history(con, cfg, runner=ctx.tool_runner, context=_analytics_context(ctx))
```

Change it to:

```python
@dataclass
class FunctionHistoryStep:
    name: str = "function_history"
    deps: Sequence[str] = ("function_analytics",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = FunctionHistoryConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_history(
            gateway,
            cfg,
            runner=ctx.tool_runner,
            context=_analytics_context(ctx),
        )
```

The important bit is: **steps never reach for `.con` directly** except for tiny local helpers like `_seed_catalog_modules` — all real analytics calls get the gateway.

### 2.4. Update tests for function_history

`tests/analytics/test_function_history.py` currently does:

```python
def test_function_history_populates_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    repo = "demo/repo"
    commit = "abc123"
    gateway = fresh_gateway
    con = gateway.con
    insert_function_metrics(gateway, [...])
    insert_modules(gateway, [...])
    ...
    cfg = FunctionHistoryConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_history(con, cfg, runner=runner)
:contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
```

Change to:

```python
    cfg = FunctionHistoryConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_history(gateway, cfg, runner=runner)
    rows = gateway.con.execute("SELECT * FROM analytics.function_history").fetchall()
```

And similarly in `test_function_history_respects_min_threshold`:

```python
    cfg = FunctionHistoryConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        overrides=FunctionHistoryConfig.Overrides(min_lines_threshold=10),
    )
    compute_function_history(gateway, cfg, runner=runner)
    rows = gateway.con.execute(
        "SELECT commit_count, lines_added FROM analytics.function_history"
    ).fetchall()
:contentReference[oaicite:15]{index=15}
```

This keeps tests aligned with the “StorageGateway as DB boundary” rule.

---

## 3. Part B — Normalize `history_timeseries` around gateways

`history_timeseries` is special: it aggregates **across many snapshot DBs**. So we have two distinct DB roles:

1. The **history DB** (where `analytics.history_timeseries` lives) — you already open this via `StorageConfig` + `open_gateway` in the CLI .
2. The **per-commit snapshot DBs** (one per commit), currently opened via `duckdb.connect(snapshot_dir / f"codeintel-{commit}.duckdb", read_only=True)` in both CLI and tests  .

We’ll:

* Wrap per-commit DB opening in a helper that returns a **`StorageGateway`**.
* Add a gateway-based *wrapper* around the existing `compute_history_timeseries` rather than rewriting its internals.

### 3.1. Add a snapshot resolver helper in `storage.gateway`

In `src/codeintel/storage/gateway.py`, near `StorageConfig` / `open_gateway`, add:

```python
from collections.abc import Callable

...

SnapshotGatewayResolver = Callable[[str], StorageGateway]
"""Callable returning a StorageGateway for a given commit."""
```

Then implement:

```python
def build_snapshot_gateway_resolver(
    *,
    db_dir: Path,
    repo: str | None = None,
) -> SnapshotGatewayResolver:
    """
    Build a resolver that opens per-commit snapshot databases as StorageGateways.

    Parameters
    ----------
    db_dir:
        Directory containing per-commit DuckDB snapshots, named
        ``codeintel-<commit>.duckdb``.
    repo:
        Optional repository slug to attach to StorageConfig; if provided, it is
        recorded in the gateway config for observability.

    Returns
    -------
    SnapshotGatewayResolver
        Callable that returns a read-only StorageGateway for the given commit.
    """

    def _resolve(commit: str) -> StorageGateway:
        db_path = db_dir / f"codeintel-{commit}.duckdb"
        if not db_path.is_file():
            message = f"Missing snapshot database for commit {commit}: {db_path}"
            raise FileNotFoundError(message)
        cfg = StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
            repo=repo,
            commit=commit,
        )
        return open_gateway(cfg)

    return _resolve
```

This pushes all the path + DuckDB flags logic into the storage layer, instead of duplicating `duckdb.connect(..., read_only=True)` in CLI/tests.

### 3.2. Add a gateway-based wrapper in `history_timeseries.py`

Keep the existing “core” function as-is, but treat it as an internal implementation detail:

```python
# existing imports
import duckdb
from collections.abc import Callable, Iterable
...
from codeintel.config.models import HistoryTimeseriesConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.tool_runner import ToolRunner

log = logging.getLogger(__name__)

DBResolver = Callable[[str], duckdb.DuckDBPyConnection]
...
def compute_history_timeseries(
    history_con: duckdb.DuckDBPyConnection,
    cfg: HistoryTimeseriesConfig,
    db_resolver: DBResolver,
    *,
    runner: ToolRunner | None = None,
) -> None:
    ...
:contentReference[oaicite:19]{index=19} :contentReference[oaicite:20]{index=20}
```

Now add:

```python
from codeintel.storage.gateway import StorageGateway, SnapshotGatewayResolver

def compute_history_timeseries_gateways(
    history_gateway: StorageGateway,
    cfg: HistoryTimeseriesConfig,
    snapshot_resolver: SnapshotGatewayResolver,
    *,
    runner: ToolRunner | None = None,
) -> None:
    """
    Gateway-based wrapper around compute_history_timeseries.

    Parameters
    ----------
    history_gateway:
        StorageGateway for the destination history DuckDB database.
    cfg:
        History aggregation configuration.
    snapshot_resolver:
        Callable returning a StorageGateway bound to the per-commit snapshot DB.
    runner:
        Optional ToolRunner for git timestamp lookups.
    """

    history_con = history_gateway.con

    def _db_resolver(commit: str) -> duckdb.DuckDBPyConnection:
        # The caller controls gateway lifetime; we only borrow the connection.
        snapshot_gateway = snapshot_resolver(commit)
        return snapshot_gateway.con

    compute_history_timeseries(
        history_con,
        cfg,
        _db_resolver,
        runner=runner,
    )
```

You now have a **canonical, gateway-based entrypoint** (`compute_history_timeseries_gateways`) that all orchestrators/CLI should use, while `compute_history_timeseries` stays as the internal implementation that all tests and helper functions already depend on.

If you want to fully hide the connection-based function from external callers, you can:

* Rename it to `_compute_history_timeseries_raw` and remove it from `__all__`.
* Keep only `compute_history_timeseries_gateways` as the public export.

### 3.3. Update CLI `history-timeseries` command to use gateways

CLI currently:

```python
def _cmd_history_timeseries(args: argparse.Namespace) -> int:
    ...
    def _resolve_db(commit: str) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(snapshot_dir / f"codeintel-{commit}.duckdb"), read_only=True)
    ...
    storage_cfg = StorageConfig(
        db_path=args.output_db,
        read_only=False,
        apply_schema=True,
        ensure_views=True,
    )
    gateway = open_gateway(storage_cfg)
    try:
        compute_history_timeseries(gateway.con, cfg, _resolve_db, runner=runner)
    ...
:contentReference[oaicite:21]{index=21}
```

Refactor to:

```python
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    open_gateway,
    build_snapshot_gateway_resolver,
)
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
...

def _cmd_history_timeseries(args: argparse.Namespace) -> int:
    ...
    storage_cfg = StorageConfig(
        db_path=args.output_db,
        read_only=False,
        apply_schema=True,
        ensure_views=True,
    )
    gateway = open_gateway(storage_cfg)

    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=args.db_dir,
        repo=args.repo,
    )

    try:
        compute_history_timeseries_gateways(
            gateway,
            cfg,
            snapshot_resolver,
            runner=runner,
        )
    except FileNotFoundError:
        LOG.exception("Missing snapshot database for history_timeseries")
        return 1
    except duckdb.Error:  # pragma: no cover - surfaced to caller
        LOG.exception("Failed to compute history_timeseries")
        return 1
    LOG.info(
        "history_timeseries written to %s for %d commits",
        args.output_db,
        len(args.commits),
    )
    return 0
```

Now:

* The output DB is opened via `StorageConfig` + `open_gateway` (as before).
* Snapshots are opened via **`StorageConfig` inside `build_snapshot_gateway_resolver`**, using consistent flags (`read_only=True`, no schema application, no views, no validation) .
* The analytics entrypoint only sees **gateways**, not raw connections.

### 3.4. Update tests for `history_timeseries`

`tests/analytics/test_history_timeseries.py` currently:

```python
gateway = open_gateway(
    StorageConfig(
        db_path=output_db,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
        read_only=False,
    )
)
...
def _resolve_db(commit: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(snapshot_dir / f"codeintel-{commit}.duckdb"), read_only=True)
...
compute_history_timeseries(gateway.con, cfg, _resolve_db, runner=runner)
:contentReference[oaicite:23]{index=23} :contentReference[oaicite:24]{index=24}
```

Update to:

```python
from codeintel.storage.gateway import StorageConfig, open_gateway, build_snapshot_gateway_resolver
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
...

gateway = open_gateway(
    StorageConfig(
        db_path=output_db,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
        read_only=False,
    )
)

snapshot_resolver = build_snapshot_gateway_resolver(
    db_dir=snapshot_dir,
    repo=repo,
)

...

compute_history_timeseries_gateways(
    gateway,
    cfg,
    snapshot_resolver,
    runner=runner,
)
```

This keeps the test logic identical (same snapshot DB layout), but now the analytics call is **100% gateway-based**.

---

## 4. Patterns for new analytics code

To make this stick (and to help your AI agents):

1. **All analytics entrypoints must accept a `StorageGateway`.**

   * Per-snapshot analytics functions (like `compute_function_metrics_and_types`, `compute_config_data_flow`, `compute_function_history`) should have signatures like:

     ```python
     def compute_X(
         gateway: StorageGateway,
         cfg: SomeAnalyticsConfig,
         *,
         context: AnalyticsContext | None = None,
         graph_ctx: GraphContext | None = None,
     ) -> None: ...
     ```

   * Cross-snapshot analytics like `history_timeseries` should accept:

     * a **destination gateway** (`history_gateway`), and
     * a **resolver** that returns gateways for other DBs (`SnapshotGatewayResolver`).

2. **Only inner helpers deal with `duckdb.DuckDBPyConnection`.**

   * It’s fine for low-level internal helpers (e.g. `_select_top_functions`, `_load_function_spans`) to take `con: duckdb.DuckDBPyConnection` — they’re called from within a gateway-based entrypoint and never used directly by orchestration.
   * Top-level orchestration (`steps.py`, CLI, MCP server) must never call `duckdb.connect` themselves; they should always go through `StorageConfig` + `open_gateway` or a resolver built on top of it.

3. **Configuration lives in `StorageConfig`, not ad-hoc calls.**

   * Want read-only? Set `read_only=True` and appropriate `apply_schema/ensure_views/validate_schema` on `StorageConfig`.
   * Need to ensure views exist? Set `ensure_views=True` and let `open_gateway` call `create_all_views` for you .
   * For snapshot DBs that are already prepared (as in `create_snapshot_db`), you can set `apply_schema=False`, `ensure_views=False`, `validate_schema=False` (as in `build_snapshot_gateway_resolver`).

4. **DuckDB best practices are centralized in your docs.**

   * You already have an extensive `docs/python_library_reference/duckdb.md` explaining why you want consistent DB access patterns and where direct `duckdb.connect` is appropriate (REPL, experiments) vs not in library code .
   * You can add a short “CodeIntel-specific” section there that says: *“Inside CodeIntel analytics, always use `StorageGateway` rather than `duckdb.connect` directly.”*

---

## 5. Suggested implementation order

If you want to stage this:

1. **Round 1 – Function history**

   * Change `compute_function_history` to take `StorageGateway`.
   * Update pipeline step and tests accordingly.
   * Run tests (`test_function_history`) and a pipeline run to sanity check.

2. **Round 2 – History timeseries**

   * Add `SnapshotGatewayResolver` + `build_snapshot_gateway_resolver` in `storage.gateway`.
   * Add `compute_history_timeseries_gateways` wrapper in `analytics.history_timeseries`.
   * Update CLI and tests to use the new gateway-based wrapper.
   * Keep the raw `compute_history_timeseries` as an internal core (or mark it private).

3. **Round 3 – Documentation & AGENTS guidance**

   * Add a short section in `docs/improvement_plans/metadata_set6_chatgptpro.md` or a new `docs/improvement_plans/db_access_normalization.md` documenting:

     * “Gateway at the boundary, connections inside only.”
     * Example signatures and usage.

If you’d like, I can turn this into that markdown improvement-plan file (with TODO checkboxes) that you can drop directly into `docs/improvement_plans/` for the agents to follow.
