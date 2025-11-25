You can treat temporal analytics as “another analytics layer on top of your existing git‑powered hotspots + per‑function metrics,” and wire it into the same DuckDB / Prefect / docs_export pipeline you already have.

I’ll split this into:

1. `analytics.function_history` – per‑function compressed git history (6.1).
2. `analytics.history_timeseries` – optional time‑bucketed metrics over commits (6.2).
3. Orchestration + docs views.

---

## 1. `analytics.function_history` – per‑GOID history (6.1)

### 1.1 Table schema

Add a new table to `config/schemas/tables.py` under the `analytics` schema.

```python
"analytics.function_history": TableSchema(
    schema="analytics",
    name="function_history",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("urn", "VARCHAR", nullable=False),
        Column("rel_path", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=False),
        Column("qualname", "VARCHAR", nullable=False),

        Column("created_in_commit", "VARCHAR", nullable=True),
        Column("created_at", "TIMESTAMP", nullable=True),

        Column("last_modified_commit", "VARCHAR", nullable=True),
        Column("last_modified_at", "TIMESTAMP", nullable=True),

        Column("age_days", "INTEGER", nullable=True),

        Column("commit_count", "INTEGER", nullable=False),
        Column("author_count", "INTEGER", nullable=False),

        Column("lines_added", "BIGINT", nullable=False),
        Column("lines_deleted", "BIGINT", nullable=False),

        Column("churn_score", "DOUBLE", nullable=False),
        Column("stability_bucket", "VARCHAR", nullable=False),  # stable, churning, new_hot, legacy_hot

        Column("history_window_start", "TIMESTAMP", nullable=True),
        Column("history_window_end", "TIMESTAMP", nullable=True),

        Column("created_at_row", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Per-function compressed git history & churn metrics derived from file history and GOID spans.",
)
```

Notes:

* `repo`/`commit` are the analysis snapshot you’re running (the “current” commit, not the historical ones). Your pipeline already carries these through almost every table.
* `history_window_*` lets you track “last 90 days vs whole repo history” if you want to cap lookback later.
* `stability_bucket` is derived from `commit_count`, `age_days`, `churn_score`.

### 1.2 Config model

Add to `config/models.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FunctionHistoryConfig:
    repo: str
    commit: str
    # Limit git lookback; None = full history
    max_history_days: int | None = 365
    min_lines_threshold: int = 1  # ignore trivial one-line changes
    default_branch: str = "HEAD"

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "FunctionHistoryConfig":
        return cls(repo=repo, commit=commit)
```

### 1.3 Reuse hotspot git plumbing

You already have a git integration for `analytics.hotspots` that computes per‑file `commit_count`, `author_count`, `lines_added`, `lines_deleted`, `score` by walking git history.

Refactor that code into reusable helpers, e.g. `analytics/git_history.py`:

```python
@dataclass
class FileCommitDelta:
    commit_hash: str
    author_email: str
    author_ts: datetime
    old_path: str
    new_path: str
    # line ranges & deltas in new file coords
    added_spans: list[tuple[int, int]]   # [start, end] inclusive
    deleted_spans: list[tuple[int, int]]
    lines_added: int
    lines_deleted: int

def iter_file_history(
    repo_root: Path,
    rel_path: str,
    *,
    max_history_days: int | None,
    default_branch: str,
) -> Iterable[FileCommitDelta]:
    """
    Iterate commits (newest→oldest or vice versa) touching this file,
    yielding per-commit hunks normalized to current-file line numbers.
    """
    ...
```

Implementation can continue to shell out to `git log -p` like your hotspot code does, just now returning structured `FileCommitDelta` instead of aggregating directly.

### 1.4 Per‑function aggregation algorithm

Create `analytics/function_history.py`:

```python
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import duckdb

from codeintel.config.models import FunctionHistoryConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.analytics.git_history import iter_file_history

log = logging.getLogger(__name__)
```

#### 1.4.1 Load function spans

We already have per‑function spans in `analytics.function_metrics` (or `core.goids` if you prefer).

SQL:

```sql
SELECT
  fm.repo,
  fm.commit,
  fm.function_goid_h128,
  fm.urn,
  fm.rel_path,
  m.module,
  fm.qualname,
  fm.start_line,
  fm.end_line,
  fm.loc
FROM analytics.function_metrics fm
JOIN core.modules m
  ON m.repo = fm.repo
 AND m.commit = fm.commit
 AND m.path = fm.rel_path
WHERE fm.repo = ? AND fm.commit = ?;
```

Materialize this into a Python structure keyed by `rel_path`:

```python
@dataclass
class FuncSpan:
    goid: int
    urn: str
    module: str
    qualname: str
    start: int
    end: int
    loc: int

functions_by_path: dict[str, list[FuncSpan]]
```

#### 1.4.2 Map file commits to functions

For each `rel_path`:

1. Call `iter_file_history(repo_root, rel_path, cfg.max_history_days, cfg.default_branch)`.

2. For each `FileCommitDelta`:

   * Compute intersection with each function span:

   ```python
   for fs in functions_by_path[rel_path]:
       added = sum(
           max(0, min(end, fs.end) - max(start, fs.start) + 1)
           for (start, end) in delta.added_spans
       )
       deleted = sum(
           max(0, min(end, fs.end) - max(start, fs.start) + 1)
           for (start, end) in delta.deleted_spans
       )
       if added + deleted < cfg.min_lines_threshold:
           continue
       ...
   ```

3. Maintain per‑function accumulators:

```python
@dataclass
class FuncHistoryAgg:
    first_commit: str | None = None
    first_ts: datetime | None = None
    last_commit: str | None = None
    last_ts: datetime | None = None
    commit_count: int = 0
    authors: set[str] = field(default_factory=set)
    lines_added: int = 0
    lines_deleted: int = 0
```

4. Update:

```python
agg = agg_by_goid[fs.goid]
agg.commit_count += 1
agg.authors.add(delta.author_email)
agg.lines_added += added
agg.lines_deleted += deleted

if agg.first_ts is None or delta.author_ts < agg.first_ts:
    agg.first_ts = delta.author_ts
    agg.first_commit = delta.commit_hash

if agg.last_ts is None or delta.author_ts > agg.last_ts:
    agg.last_ts = delta.author_ts
    agg.last_commit = delta.commit_hash
```

This gives you **per‑function history** across the git window, keyed by GOID at the current commit.

#### 1.4.3 Compute churn & stability

Define:

```python
total_churn = agg.lines_added + agg.lines_deleted
loc = max(fs.loc, 1)

# Normalized churn over the window, scaled to ~[0,1]
raw_churn = total_churn / loc

# Optionally log-transform or squash:
churn_score = min(raw_churn / 10.0, 1.0)  # 10x LOC changes ~1.0
```

Age:

```python
if agg.first_ts is not None:
    age_days = (analysis_ts - agg.first_ts).days
else:
    age_days = None
```

Stability buckets (tune thresholds per repo later):

```python
def classify_stability(
    *,
    age_days: int | None,
    commit_count: int,
    churn_score: float,
    window_days: int | None,
) -> str:
    if age_days is None:
        return "new_hot" if churn_score > 0 else "unknown"

    recent_window_days = window_days or 365
    is_new = age_days <= 30
    is_hot = churn_score >= 0.5 or commit_count >= 5

    if is_new and is_hot:
        return "new_hot"
    if not is_new and not is_hot and commit_count <= 2:
        return "stable"
    if total_churn := churn_score >= 0.2:
        return "churning"
    return "legacy_hot" if age_days > recent_window_days and is_hot else "stable"
```

Record `history_window_start = analysis_ts - timedelta(days=cfg.max_history_days)` (if set) and `history_window_end = analysis_ts` (UTC now at run).

The nice bit: this is **purely git‑based**, no extra analyses beyond hotspots.

### 1.5 Writing to DuckDB

`compute_function_history`:

```python
def compute_function_history(con: duckdb.DuckDBPyConnection, cfg: FunctionHistoryConfig) -> None:
    ensure_schema(con, "analytics.function_history")
    con.execute(
        "DELETE FROM analytics.function_history WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    # 1) load function spans
    rows = con.execute(FUNCTION_SPANS_SQL, [cfg.repo, cfg.commit]).fetchall()
    # build functions_by_path ...

    # 2) iterate git history per file & aggregate into agg_by_goid ...

    now = datetime.now(tz=UTC)
    window_start = None
    if cfg.max_history_days is not None:
        window_start = now - timedelta(days=cfg.max_history_days)

    insert_rows = []
    for fs in all_function_spans:
        agg = agg_by_goid.get(fs.goid, FuncHistoryAgg())
        if agg.first_ts:
            age_days = (now - agg.first_ts).days
        else:
            age_days = None

        churn_score = compute_churn_score(agg, fs.loc)
        stability = classify_stability(
            age_days=age_days,
            commit_count=agg.commit_count,
            churn_score=churn_score,
            window_days=cfg.max_history_days,
        )

        insert_rows.append((
            cfg.repo,
            cfg.commit,
            fs.goid,
            fs.urn,
            fs.rel_path,
            fs.module,
            fs.qualname,
            agg.first_commit,
            agg.first_ts,
            agg.last_commit,
            agg.last_ts,
            age_days,
            agg.commit_count,
            len(agg.authors),
            agg.lines_added,
            agg.lines_deleted,
            churn_score,
            stability,
            window_start,
            now,
            now,
        ))

    con.executemany(
        """
        INSERT INTO analytics.function_history (
            repo, commit,
            function_goid_h128, urn, rel_path, module, qualname,
            created_in_commit, created_at,
            last_modified_commit, last_modified_at,
            age_days,
            commit_count, author_count,
            lines_added, lines_deleted,
            churn_score, stability_bucket,
            history_window_start, history_window_end,
            created_at_row
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
```

### 1.6 Integration with profiles & views

You already use `function_profile.*` as the main “denormalized” per‑function profile.

* Extend `analytics/profiles.py` (or wherever you build `function_profile`) to **left join** `analytics.function_history` on `(repo, commit, function_goid_h128)` and project:

  * `created_in_commit`, `created_at`
  * `last_modified_commit`, `last_modified_at`
  * `age_days`
  * `commit_count`, `author_count`
  * `lines_added`, `lines_deleted`
  * `churn_score`, `stability_bucket`

* In `docs_export/storage/views.py`, add a view `docs.v_function_history` or extend `docs.v_function_architecture` to expose these fields:

```sql
CREATE OR REPLACE VIEW docs.v_function_history AS
SELECT
  fp.repo,
  fp.commit,
  fp.function_goid_h128,
  fp.urn,
  fp.rel_path,
  fp.module,
  fp.qualname,
  fh.created_in_commit,
  fh.created_at,
  fh.last_modified_commit,
  fh.last_modified_at,
  fh.age_days,
  fh.commit_count,
  fh.author_count,
  fh.lines_added,
  fh.lines_deleted,
  fh.churn_score,
  fh.stability_bucket
FROM analytics.function_profile fp
LEFT JOIN analytics.function_history fh
  ON fh.repo = fp.repo
 AND fh.commit = fp.commit
 AND fh.function_goid_h128 = fp.function_goid_h128;
```

### 1.7 Orchestration step

In `orchestration/steps.py`, add:

```python
@dataclass
class FunctionHistoryStep:
    name: str = "function_history"
    deps: Sequence[str] = ("function_analytics", "hotspots")  # reuses AST + git config

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = FunctionHistoryConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_function_history(con, cfg)
```

Wire this into the step graph so it runs after:

* `goids`, `function_metrics`
* `ast_metrics` / `hotspots` (so git tooling is configured, though we don’t strictly depend on hotspots table itself).

Expose in docs export:

```python
JSONL_DATASETS["analytics.function_history"] = "function_history.jsonl"
PARQUET_DATASETS["analytics.function_history"] = "function_history.parquet"
```

---

## 2. `analytics.history_timeseries` – entity metrics over time (6.2)

This is heavier and spans **multiple commits**. Given your existing design of “one DuckDB per `<repo, commit>` run,” we treat this as a separate aggregation step that reads multiple snapshots (or multiple Parquet exports) and writes a consolidated history table.

### 2.1 Identity across commits: `entity_stable_id`

Because GOIDs include the commit in their hash, a function gets a new `goid_h128` every commit.

Define a stable key independent of commit:

```python
def make_entity_stable_id(
    *,
    repo: str,
    rel_path: str,
    language: str,
    kind: str,
    qualname: str,
) -> str:
    raw = f"{repo}:{rel_path}:{language}:{kind}:{qualname}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
```

We’ll compute this for both **functions** and **modules**.

### 2.2 Table schema

Single generic table, dimensioned by entity kind + time bucket:

```python
"analytics.history_timeseries": TableSchema(
    schema="analytics",
    name="history_timeseries",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("entity_kind", "VARCHAR", nullable=False),  # "function" | "module"
        Column("entity_stable_id", "VARCHAR", nullable=False),

        # For functions: function_goid_h128 at that commit; null for modules
        Column("function_goid_h128", "DECIMAL(38,0)"),
        Column("module", "VARCHAR"),      # module name or module path
        Column("rel_path", "VARCHAR", nullable=False),
        Column("language", "VARCHAR", nullable=False),
        Column("qualname", "VARCHAR"),    # function qualname; null for modules

        Column("commit", "VARCHAR", nullable=False),
        Column("commit_ts", "TIMESTAMP", nullable=False),

        # Metrics (extend as needed)
        Column("loc", "INTEGER"),
        Column("cyclomatic_complexity", "INTEGER"),
        Column("coverage_ratio", "DOUBLE"),
        Column("static_error_count", "INTEGER"),
        Column("typedness_bucket", "VARCHAR"),
        Column("risk_score", "DOUBLE"),
        Column("risk_level", "VARCHAR"),

        Column("bucket_label", "VARCHAR", nullable=True),  # optional: weekly/monthly bucket label
        Column("created_at_row", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "entity_kind", "entity_stable_id", "commit"),
    description="Per-commit metrics for selected functions/modules for temporal analysis.",
)
```

This design lets you:

* Query **per commit**, **per entity**, or aggregate by `bucket_label` for weekly/monthly summaries.

### 2.3 Config model

In `config/models.py`:

```python
@dataclass(frozen=True)
class HistoryTimeseriesConfig:
    repo: str
    # List of commits to include in the history; caller passes them in
    commits: tuple[str, ...]
    # Which entities to track & how many
    entity_kind: str = "function"          # "function" | "module" | "both"
    max_entities: int = 500                # cap to top-N by importance
    # How to choose top-N (risk, centrality, hotspots, etc.)
    selection_strategy: str = "risk_score" # or "call_pagerank", "hotspot_score"

    @classmethod
    def from_args(
        cls, *, repo: str, commits: Sequence[str], **kwargs: Any
    ) -> "HistoryTimeseriesConfig":
        return cls(repo=repo, commits=tuple(commits), **kwargs)
```

You’ll likely drive this config from a dedicated `history-timeseries` CLI command that passes a commit list.

### 2.4 Source of snapshot metrics

For each commit we want:

* For **functions**:

  * `analytics.function_profile` (or at minimum `function_metrics`, `coverage_functions`, `goid_risk_factors`).
* For **modules**:

  * `analytics.module_profile` (if present) or a rollup from `function_profile`/`file_profile`.

The easiest path is:

* Require that for each commit `Ci` in the history, you have **either**:

  * a DuckDB DB at `build/db/codeintel-<Ci>.duckdb`, or
  * exported Parquet for `function_profile` / `module_profile` under a commit‑scoped directory.

The aggregator opens each DB/Parquet, extracts the metrics, and writes into a **single** history DB (the one for “current commit” or a dedicated `history.duckdb`).

### 2.5 Selecting “most important” entities

You don’t want timeseries for every function; you want top‑N.

Strategy:

1. Use the **latest commit in the history list** as the “current” commit `C0`.

2. In its DB, compute a **selection set**:

   * If entity_kind includes `function`:

     ```sql
     SELECT
       function_goid_h128,
       rel_path,
       module,
       language,
       qualname,
       risk_score,
       call_pagerank,       -- from graph_metrics_functions_ext if available
       churn_score,         -- from function_history if available
       coverage_ratio
     FROM analytics.function_profile
     WHERE repo = ? AND commit = ?
     ORDER BY risk_score DESC
     LIMIT ?;
     ```

     Or use a composite score: high risk, high centrality, low coverage, high churn.

   * For modules, similar using `module_profile` + import graph metrics.

3. For each selected entity, compute `entity_stable_id` using `(repo, rel_path, language, kind, qualname)`.

Store this selection in memory as you process other commits.

### 2.6 Aggregation algorithm

Implement `analytics/history_timeseries.py`:

```python
def compute_history_timeseries(
    history_con: duckdb.DuckDBPyConnection,
    cfg: HistoryTimeseriesConfig,
    db_resolver: Callable[[str], duckdb.DuckDBPyConnection],
) -> None:
    """
    history_con: connection to the DB where history_timeseries will be stored.
    db_resolver: function commit -> DuckDB connection for that commit's snapshot.
    """
    ensure_schema(history_con, "analytics.history_timeseries")
    history_con.execute(
        "DELETE FROM analytics.history_timeseries WHERE repo = ?",
        [cfg.repo],
    )

    now = datetime.now(tz=UTC)
    selection = select_top_entities_for_history(cfg, db_resolver)  # returns list of EntitySelection
    rows: list[tuple] = []

    for commit in cfg.commits:
        con_ci = db_resolver(commit)
        commit_ts = fetch_commit_timestamp(cfg.repo, commit)  # via git

        if cfg.entity_kind in ("function", "both"):
            rows.extend(
                _collect_function_rows_for_commit(
                    cfg=cfg,
                    con_ci=con_ci,
                    commit=commit,
                    commit_ts=commit_ts,
                    selection=selection.functions,
                    now=now,
                )
            )

        if cfg.entity_kind in ("module", "both"):
            rows.extend(
                _collect_module_rows_for_commit(
                    cfg=cfg,
                    con_ci=con_ci,
                    commit=commit,
                    commit_ts=commit_ts,
                    selection=selection.modules,
                    now=now,
                )
            )

    history_con.executemany(
        """
        INSERT INTO analytics.history_timeseries (
            repo, entity_kind, entity_stable_id,
            function_goid_h128, module, rel_path, language, qualname,
            commit, commit_ts,
            loc, cyclomatic_complexity, coverage_ratio,
            static_error_count, typedness_bucket,
            risk_score, risk_level,
            bucket_label,
            created_at_row
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
```

#### 2.6.1 Per‑commit function rows

Given a commit DB connection `con_ci`:

```sql
SELECT
  fp.function_goid_h128,
  fp.rel_path,
  fp.module,
  fp.language,
  fp.qualname,
  fp.loc,
  fp.cyclomatic_complexity,
  fp.coverage_ratio,
  fp.static_error_count,
  fp.typedness_bucket,
  fp.risk_score,
  fp.risk_level
FROM analytics.function_profile fp
WHERE fp.repo = ? AND fp.commit = ?;
```

For each row where `(repo, rel_path, language, 'function', qualname)` matches a selected entity, compute `entity_stable_id` and append:

```python
bucket_label = commit_ts.strftime("%Y-%m-%d")  # or ISO week/month label

rows.append((
    cfg.repo,
    "function",
    entity_stable_id,
    function_goid_h128,
    module,
    rel_path,
    language,
    qualname,
    commit,
    commit_ts,
    loc,
    cyclomatic_complexity,
    coverage_ratio,
    static_error_count,
    typedness_bucket,
    risk_score,
    risk_level,
    bucket_label,
    now,
))
```

Modules similarly using `analytics.module_profile` columns: total LOC, coverage, high‑risk function count, etc.

### 2.7 Orchestration & CLI

Because this is cross‑commit, it’s best as a **separate flow**, not part of the single‑commit pipeline.

* Add a small CLI command, e.g. in `cli/main.py`:

  ```bash
  codeintel history-timeseries \
    --repo my-repo \
    --commits <sha1> <sha2> <sha3> \
    --db-dir build/db \
    --output-db build/db/history.duckdb
  ```

* Implement `db_resolver(commit)` as:

  ```python
  def db_resolver(commit: str) -> duckdb.DuckDBPyConnection:
      db_path = db_root / f"codeintel-{commit}.duckdb"
      return duckdb.connect(str(db_path), read_only=True)
  ```

* Optionally also expose a Prefect flow `history_timeseries_flow` that:

  * loops across commits,
  * ensures snapshot DBs exist (or runs a reduced pipeline to generate them),
  * then calls `compute_history_timeseries`.

### 2.8 Docs views

Two handy views:

**Function history timeseries:**

```sql
CREATE OR REPLACE VIEW docs.v_function_history_timeseries AS
SELECT
  h.repo,
  h.entity_stable_id,
  h.commit,
  h.commit_ts,
  h.rel_path,
  h.module,
  h.qualname,
  h.loc,
  h.cyclomatic_complexity,
  h.coverage_ratio,
  h.static_error_count,
  h.typedness_bucket,
  h.risk_score,
  h.risk_level,
  h.bucket_label
FROM analytics.history_timeseries h
WHERE h.entity_kind = 'function';
```

**Module history timeseries:**

```sql
CREATE OR REPLACE VIEW docs.v_module_history_timeseries AS
SELECT
  h.repo,
  h.entity_stable_id,
  h.commit,
  h.commit_ts,
  h.module,
  h.rel_path,
  h.coverage_ratio,
  h.risk_score,
  h.risk_level,
  h.bucket_label
FROM analytics.history_timeseries h
WHERE h.entity_kind = 'module';
```

LLM‑side usage examples:

* “Show me the last 6 commits for function X’s risk_score and coverage.”
* “Plot coverage_ratio and risk_score over time for module Y.”
* “Flag functions whose risk_score increased by >0.3 over the last N commits.”

---

## 3. Wiring everything into your existing system

### 3.1 New analytics modules

Under `src/codeintel/codeintel/analytics/`:

* `git_history.py` – extracted utility from hotspots for per‑file commit deltas.
* `function_history.py` – per‑GOID aggregation (6.1).
* `history_timeseries.py` – cross‑commit aggregator (6.2).

Each uses the same DuckDB client + schema registry as your other analytics, and leans on `function_profile` / `module_profile` so they stay in sync with the rest of your architecture views.

### 3.2 Export

Add to `docs_export/export_jsonl.py` / `export_parquet.py`:

```python
JSONL_DATASETS.update({
    "analytics.function_history": "function_history.jsonl",
    "analytics.history_timeseries": "history_timeseries.jsonl",
})

PARQUET_DATASETS.update({
    "analytics.function_history": "function_history.parquet",
    "analytics.history_timeseries": "history_timeseries.parquet",
})
```

So the new datasets show up in `Document Output/` alongside `goid_risk_factors.jsonl`, `function_profile.jsonl`, etc.

---

### 3.3 How an agent benefits

With these in place, an AI agent can:

* See **per‑function git history** directly in `docs.v_function_architecture` / `docs.v_function_history`:

  * “This function is 800 lines, high risk, **and** edited by 6 authors in the last month → maybe don’t do a massive refactor here.”

* Use `docs.v_function_history_timeseries` to reason about trends:

  * “Coverage went from 80% → 30% after commit X; maybe the tests were broken or disabled.”
  * “Risk_score has been trending up across last 5 commits; consider splitting this function.”

* Prefer **stable, long‑lived** functions for new integrations:

  * `stability_bucket = 'stable'` + high coverage + low churn.

If you’d like, I can next sketch the **DuckDB SQL** for computing `stability_bucket` percentiles inside the DB (instead of Python), or show a couple of example queries an MCP tool could expose (e.g., “find high‑risk, high‑churn functions in subsystem X”).
