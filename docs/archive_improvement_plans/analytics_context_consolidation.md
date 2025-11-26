You’ve already done 80% of the hard work here: `AnalyticsContext` is a really solid “snapshot spine”; the refactor is mostly about **using it everywhere in a consistent way** and tightening up a few stragglers.

I’ll walk through:

1. What `AnalyticsContext` actually is today
2. What “canonical entrypoint” means in practice
3. Concrete code changes in `analytics.context`
4. How to refit analytics modules to lean on it
5. How to integrate this cleanly in `steps.py` and the Prefect flow
6. A short “rules of the road” section for future agents

---

## 1. Baseline: what `AnalyticsContext` already does

`src/codeintel/analytics/context.py` already gives you a best-in-class snapshot object:

* **Config:** `AnalyticsContextConfig` controls repo/commit, repo_root, catalog provider, graph / AST budgets, sample seed, and whether to load symbol graphs, plus an optional `metrics_hook` .

* **Stats:** `AnalyticsContextStats` captures counts for ASTs and all graphs + truncation signals .

* **Timing:** `AnalyticsResourceCounters` tracks per-resource load times (catalog, module_map, call/import/symbol graphs, ASTs) .

* **The context itself:**

  ```python
  @dataclass(frozen=True)
  class AnalyticsContext:
      """Shared analytics artifacts for a repo/commit snapshot."""

      repo: str
      commit: str
      repo_root: Path
      catalog: FunctionCatalogProvider
      module_map: dict[str, str]
      function_ast_map: dict[int, FunctionAst]
      missing_function_goids: set[int]
      call_graph: nx.DiGraph
      import_graph: nx.DiGraph | None
      symbol_module_graph: nx.Graph | None
      symbol_function_graph: nx.Graph | None
      created_at: datetime
      snapshot_id: str
      stats: AnalyticsContextStats
      resources: AnalyticsResourceCounters
  :contentReference[oaicite:3]{index=3}
  ```

* **Builder:** `build_analytics_context(gateway, cfg)`:

  * Uses `FunctionCatalogService.from_db` (unless you supply a provider) and `load_module_map` .
  * Loads call/import/symbol graphs through `nx_views` with trimming based on `max_*` budgets, logging metrics/truncation  .
  * Loads function ASTs via `load_function_asts` with a `FunctionAstLoadRequest` (repo, commit, repo_root, max_functions) .
  * Assembles `AnalyticsContext` + `stats` + `resources` and fires `metrics_hook` if provided  .

Pipeline code already caches it:

```python
@dataclass
class PipelineContext:
    ...
    analytics_context: AnalyticsContext | None = None
    ...

def _analytics_context(ctx: PipelineContext) -> AnalyticsContext:
    if ctx.analytics_context is None:
        ctx.analytics_context = build_analytics_context(
            ctx.gateway,
            AnalyticsContextConfig(
                repo=ctx.repo,
                commit=ctx.commit,
                repo_root=ctx.repo_root,
            ),
        )
        ctx.function_catalog = ctx.analytics_context.catalog
    return ctx.analytics_context
:contentReference[oaicite:10]{index=10}
```

So: the plumbing is there. The refactor is about **making all analytics modules and orchestration consistently treat this as “the one true snapshot.”**

---

## 2. What “canonical entrypoint” actually means

Concretely, the design target is:

1. **One context per `<repo, commit>` per process.**
   Both the CLI pipeline (`steps.py`) and Prefect flows should build it once and pass it into all analysis stages that need catalog, module map, graphs, or ASTs.

2. **Standard analytics function signatures:**

   ```python
   def compute_X(
       gateway: StorageGateway,
       cfg: SomeAnalyticsConfig,
       *,
       context: AnalyticsContext | None = None,
       graph_ctx: GraphContext | None = None,  # when graph-heavy
       ...
   ) -> None: ...
   ```

   You already follow this pattern in a lot of modules: `config_data_flow`, `dependencies`, `data_model_usage`, `semantic_roles`, `function_contracts`, `graph_metrics`, `graph_metrics_ext`, `module_graph_metrics_ext`, `symbol_graph_metrics`, `config_graph_metrics`, `cfg_dfg_metrics`, `subsystem_graph_metrics`, `coverage_analytics` etc.      

3. **When you have a context, never re-load catalog/graphs/ASTs from DB.**
   Use `context.catalog`, `context.module_map`, `context.function_ast_map`, `context.call_graph`, `context.import_graph`, etc., rather than `FunctionCatalogService.from_db` or `load_call_graph`/`load_import_graph`.

4. **Context creation is centralized in orchestration only.**

   * In `steps.py`: via `_analytics_context(ctx)` as shown above .
   * In Prefect flow: via a lightweight helper that mirrors `_analytics_context`.

5. **Modules may still fall back for ad-hoc usage.**
   It’s fine for an analytics function to do `context or build_analytics_context(...)` if it’s meant to be called from tests/REPL, but the *pipeline* code should always pass a context so that fallback path is “rare, not routine.”

---

## 3. Centralizing context helpers in `analytics.context`

Right now, each module that supports `context` repeats the same little pattern:

```python
shared_context = context or build_analytics_context(
    gateway,
    AnalyticsContextConfig(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog_provider=catalog_provider,
    ),
)
:contentReference[oaicite:18]{index=18}
```

You do this in `entrypoints`, `config_data_flow`, `dependencies`, `function_contracts`, `data_model_usage`, `semantic_roles` etc.     

### 3.1 Add a small shared helper

In `analytics/context.py`, add:

```python
# src/codeintel/analytics/context.py

def ensure_analytics_context(
    gateway: StorageGateway,
    *,
    cfg: AnalyticsContextConfig,
    context: AnalyticsContext | None = None,
) -> AnalyticsContext:
    """
    Return an existing AnalyticsContext or build one from the provided config.

    This helper centralizes the `context or build_analytics_context` pattern
    so analytics modules don't need to duplicate it.

    Parameters
    ----------
    gateway:
        Storage gateway exposing the DuckDB connection.
    cfg:
        AnalyticsContextConfig specifying repo, commit, and budgets.
    context:
        Optional pre-built context to reuse.
    """
    if context is not None:
        # Optional safety: assert we're not mixing snapshots.
        if context.repo != cfg.repo or context.commit != cfg.commit:
            raise ValueError(
                f"AnalyticsContext mismatch: {context.repo}@{context.commit} "
                f"vs {cfg.repo}@{cfg.commit}"
            )
        return context
    return build_analytics_context(gateway, cfg)
```

Now every analytics module can call this instead of inlining the pattern.

---

## 4. Refitting analytics modules to rely on the context

### 4.1 Example: `config_data_flow`

Current top-level pattern:

```python
def compute_config_data_flow(
    gateway: StorageGateway,
    cfg: ConfigDataFlowConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    ...
    shared_context = context or build_analytics_context(
        gateway,
        AnalyticsContextConfig(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
        ),
    )
    entrypoints = _entrypoints(con, cfg.repo, cfg.commit)
    call_graph = shared_context.call_graph
    ast_by_goid = shared_context.function_ast_map
    missing = shared_context.missing_function_goids
:contentReference[oaicite:24]{index=24}
```

Refactor to use the helper and make the “canonical” intent explicit:

```python
from codeintel.analytics.context import AnalyticsContext, AnalyticsContextConfig, ensure_analytics_context

def compute_config_data_flow(
    gateway: StorageGateway,
    cfg: ConfigDataFlowConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    ...
    shared_context = ensure_analytics_context(
        gateway,
        cfg=AnalyticsContextConfig(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
        ),
        context=context,
    )

    entrypoints = _entrypoints(con, cfg.repo, cfg.commit)
    call_graph = shared_context.call_graph
    ast_by_goid = shared_context.function_ast_map
    missing = shared_context.missing_function_goids
    ...
```

This removes duplicated construction logic and makes it obvious that the **first choice** is “use the passed context”.

You’d make the same mechanical change in:

* `analytics.entrypoints.build_entrypoints` 
* `analytics.dependencies.build_external_dependency_calls` 
* `analytics.function_contracts.compute_function_contracts` 
* `analytics.data_model_usage.compute_data_model_usage` 
* `analytics.semantic_roles.compute_semantic_roles` 

Each becomes a trivial call to `ensure_analytics_context` rather than rolling its own.

### 4.2 Example: `subsystems` gaining `context` awareness

Right now, `build_subsystems` takes only `gateway` and `cfg`, and always reloads the import graph from `graph.import_graph_edges` via `load_import_graph` .

Update its signature and implementation:

```python
from codeintel.analytics.context import AnalyticsContext

def build_subsystems(
    gateway: StorageGateway,
    cfg: SubsystemsConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    """Populate analytics.subsystems and analytics.subsystem_modules for a repo/commit."""
    con = gateway.con
    ensure_schema(con, "analytics.subsystems")
    ensure_schema(con, "analytics.subsystem_modules")

    con.execute(
        "DELETE FROM analytics.subsystems WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    modules, tags_by_module = _load_modules(gateway, cfg)
    if not modules:
        log.info("No modules available for subsystem inference; skipping.")
        return

    # If we have a context, reuse its import graph; otherwise, load from DB.
    if context is not None and (context.repo != cfg.repo or context.commit != cfg.commit):
        log.warning(
            "subsystems context mismatch: context=%s@%s cfg=%s@%s",
            context.repo,
            context.commit,
            cfg.repo,
            cfg.commit,
        )
        import_graph = load_import_graph(gateway, cfg.repo, cfg.commit)
    else:
        import_graph = context.import_graph if context is not None else load_import_graph(
            gateway, cfg.repo, cfg.commit
        )

    affinity_graph = _build_weighted_graph(gateway, cfg, modules)
    adjacency = _graph_to_adjacency(affinity_graph)
    seed_labels = _seed_labels_from_tags(tags_by_module)
    labels = _label_propagation_nx(affinity_graph, seed_labels)
    labels = _reassign_small_clusters(labels, adjacency, cfg.min_modules)
    labels = _limit_clusters(labels, adjacency, cfg.max_subsystems)
    clusters = _clusters_from_labels(labels)

    risk_stats = _aggregate_risk(gateway, cfg, labels)
    now = datetime.now(UTC)
    ctx = SubsystemBuildContext(
        cfg=cfg,
        labels=labels,
        tags_by_module=tags_by_module,
        import_graph=import_graph,
        risk_stats=risk_stats,
        now=now,
    )
    subsystem_rows, membership_rows = _build_rows(clusters, ctx)
    ...
```

Now subsystem inference participates fully in the “shared snapshot” rather than always hitting DuckDB.

### 4.3 Example: making test analytics context-aware (optional)

Some test analytics don’t strictly need `AnalyticsContext`, but you *can* use it beneficially:

* `test_graph_metrics` already takes `context: AnalyticsContext | None` and reuses its call graph when present .
* `build_test_profile` and `build_behavioral_coverage` currently parse test files from disk using `repo_root` and `parse_python_module` directly  .

You could extend their signatures:

```python
def build_test_profile(
    gateway: StorageGateway,
    cfg: TestProfileConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None: ...
```

and then, when building the AST index, prefer `context.function_ast_map` for tests that have GOIDs instead of reparsing from disk. That’s a nice incremental optimization, but not required for correctness—the big wins are on function/graph analytics, which you already wired.

---

## 5. Orchestration: always build & pass context

### 5.1 Pipeline steps (`steps.py`)

You’re already doing the right thing for many steps:

* `FunctionHistoryStep`, `FunctionAnalyticsStep`, `FunctionEffectsStep`, `FunctionContractsStep`, `DataModelUsageStep`, `ConfigDataFlowStep`, `CoverageAnalyticsStep`, `GraphMetricsStep`, `SemanticRolesStep` all call `_analytics_context(ctx)` and pass it through        .

For modules you just made context-aware (e.g. `build_subsystems`), update the step:

```python
@dataclass
class SubsystemsStep:
    ...

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = SubsystemsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        acx = _analytics_context(ctx)
        build_subsystems(gateway, cfg, context=acx)
:contentReference[oaicite:42]{index=42}
```

If you also update `compute_test_coverage_edges` or `build_test_profile` to accept `context`, you can likewise call `_analytics_context(ctx)` in `TestCoverageEdgesStep` / `TestProfileStep` and pass it along.

### 5.2 Prefect flow

The Prefect flow imports the same analytics functions directly but currently doesn’t manage `AnalyticsContext` explicitly .

The preferred option already in use:

1. **Reuse `PipelineContext` inside Prefect.**
   The flow already imports `PipelineContext` and `PIPELINE_STEPS` for one code path . Ensuring the Prefect tasks use those steps means they *automatically* use `_analytics_context`. Use this as the standard going forward.

An alternate option that may be worth considering with future design changes
2. **Add a tiny Prefect-side wrapper:**

   ```python
   from codeintel.analytics.context import AnalyticsContextConfig, build_analytics_context

   def build_prefect_analytics_context(
       gateway: StorageGateway,
       repo_root: Path,
       repo: str,
       commit: str,
   ) -> AnalyticsContext:
       return build_analytics_context(
           gateway,
           AnalyticsContextConfig(
               repo=repo,
               commit=commit,
               repo_root=repo_root,
           ),
       )
   ```

   Then in your flow run body:

   ```python
   gateway = _get_gateway(args.db_path, ...)
   acx = build_prefect_analytics_context(gateway, args.repo_root, args.repo, args.commit)

   # Example usage:
   cfg = FunctionAnalyticsConfig.from_paths(...)
   summary = compute_function_metrics_and_types(
       gateway,
       cfg,
       options=FunctionAnalyticsOptions(context=acx),
   )
   ```

   And similar for `compute_function_contracts`, `compute_data_model_usage`, `compute_config_data_flow`, `compute_semantic_roles`, `compute_graph_metrics*`, etc.

Either way, the **rule** becomes: *anytime we run analytics for a snapshot in a flow-like context, we build a single `AnalyticsContext` and pass it through*.

---

## 6. “Rules of the road” for future code & agents

You can fold this into `AGENTS.md` or a short `docs/analytics_context.md`:

1. **If you need catalog, module_map, function ASTs, or call/import/symbol graphs, you must:**

   * Accept a `context: AnalyticsContext | None` argument, and
   * Inside the function, do:

     ```python
     shared = ensure_analytics_context(
         gateway,
         cfg=AnalyticsContextConfig(
             repo=cfg.repo,
             commit=cfg.commit,
             repo_root=cfg.repo_root,
             catalog_provider=catalog_provider,  # if applicable
         ),
         context=context,
     )
     ```

2. **Do not call graph loaders or catalog constructors directly when you already have a context.**

   * `shared.catalog` instead of `FunctionCatalogService.from_db`.
   * `shared.call_graph` / `shared.import_graph` instead of `load_call_graph` / `load_import_graph`.
   * `shared.function_ast_map` instead of re-parsing AST from disk.

3. **Orchestration is responsible for building the context once.**

   * In `steps.py`, always call `_analytics_context(ctx)` before invoking analytics functions, and pass the resulting object into `context=` or `options=...`.
   * In Prefect, either reuse the pipeline steps or build a single `AnalyticsContext` per run and thread it through.

4. **Snapshot safety:**
   If an analytics function receives a `context` whose `repo/commit` don’t match its `cfg`, either:

   * Log a warning and fall back to DB (as `coverage_analytics` does) , or
   * Raise a `ValueError` (preferred for internal use) to catch bugs early.

---


