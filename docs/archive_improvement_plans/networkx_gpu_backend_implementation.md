
Here’s a concrete implementation plan to turn the NetworkX GPU backend into a **first-class, config-driven feature** instead of “some random environment trick.” I’ll assume:

* You want a **single knob** in config/CLI/Prefect that says “use GPU for graph algorithms when possible.”
* You’re okay with **graceful fallback to CPU** if the GPU backend isn’t available.
* You don’t want AI agents sprinkling `nx_cugraph` imports all over analytics modules.

I’ll structure this around:

1. Design goals & baseline
2. `GraphBackendConfig` in `config.models`
3. `_maybe_enable_nx_gpu` helper
4. Wiring it into CLI & Prefect
5. Integrating with `graphs.nx_views` & GraphService
6. Updating analytics modules to pass `use_gpu`
7. Testing & agent guidance

---

## 1. Design goals & current baseline

From your `networkx_gpu.md` improvement plan (and code):

* Heavy NetworkX users:
  `analytics/graph_metrics.py`, `graph_metrics_ext.py`, `module_graph_metrics_ext.py`,
  `test_graph_metrics.py`, `cfg_dfg_metrics.py`, `symbol_graph_metrics.py`,
  `config_graph_metrics.py`, `subsystem_graph_metrics.py`, `graph_stats.py`.

* The doc proposes:

  * A `GraphBackendConfig` dataclass
  * A `_maybe_enable_nx_gpu(...)` helper
  * Hooking this in **CLI + Prefect**, not deep inside analytics modules.

Our implementation will:

* Add a **config object** that travels through your config + `PipelineContext`.
* Call `_maybe_enable_nx_gpu` **once per process** (CLI + Prefect).
* Thread a `use_gpu: bool` flag down into your graph loading functions and GraphService.

We’ll stay conservative: if GPU backend isn’t available, we **log and fall back to CPU** unless a `strict` flag says “fail if GPU can’t be enabled.”

---

## 2. Add `GraphBackendConfig` to `config.models`

In `src/codeintel/config/models.py`, near your other config dataclasses (`CodeIntelConfig`, `GraphMetricsConfig`, etc.), add:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class GraphBackendConfig:
    """
    Configuration for NetworkX graph backend selection.

    Attributes
    ----------
    use_gpu:
        If True, prefer a GPU-capable backend such as nx-cugraph when available.
    backend:
        Preferred backend identifier. The exact string depends on your
        networkx / nx-cugraph versions; "auto" lets _maybe_enable_nx_gpu pick.
    strict:
        If True, failure to enable the requested backend raises an error
        instead of falling back to CPU.
    """

    use_gpu: bool = False
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    strict: bool = False
```

Now thread this through your **top-level config**. For example, in `CodeIntelConfig`:

```python
@dataclass(frozen=True)
class CodeIntelConfig:
    ...
    graph_backend: GraphBackendConfig = GraphBackendConfig()
    ...
```

And in anything that stands in for runtime options, e.g. `ExportArgs`, `PipelineArgs`, etc., add a field or CLI mapping so you can say:

* `--nx-gpu` → `graph_backend.use_gpu = True`
* `--nx-backend nx-cugraph` → `graph_backend.backend = "nx-cugraph"`

(Exact CLI syntax is up to you; more on that below.)

---

## 3. Implement `_maybe_enable_nx_gpu` helper

Create a small helper module whose only job is to configure NetworkX’s backend **once** per process.

For example: `src/codeintel/cli/nx_backend.py`:

```python
# src/codeintel/cli/nx_backend.py
from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from codeintel.config.models import GraphBackendConfig

LOG = logging.getLogger(__name__)


def _enable_nx_cugraph_backend() -> None:
    """
    Try to switch NetworkX's backend to nx-cugraph (GPU).

    The exact API may differ slightly by NetworkX / nx-cugraph version,
    so keep this function as the single place that is allowed to touch
    those integration details.
    """
    try:
        import nx_cugraph  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Requested GPU backend, but nx_cugraph is not installed."
        ) from exc

    # Depending on versions, you may need to adjust the call below.
    # These are examples; check your installed nx/nx-cugraph docs.
    try:
        # Option 1 (networkx >= 3.4 style)
        # nx.set_default_graph_backend("nx_cugraph")
        # Option 2 (nx_cugraph convenience)
        nx_cugraph.set_default_backend()  # type: ignore[attr-defined]
        LOG.info("NetworkX GPU backend enabled via nx_cugraph.")
    except AttributeError:
        # Fallback to older-style API if needed
        LOG.warning(
            "nx_cugraph.set_default_backend not available; "
            "please adjust _enable_nx_cugraph_backend for your versions."
        )
        # You can add version-specific logic here if needed.


def maybe_enable_nx_gpu(cfg: GraphBackendConfig) -> None:
    """
    Configure NetworkX backend based on GraphBackendConfig.

    Call this once at process startup (CLI / Prefect flow).
    """
    if not cfg.use_gpu:
        LOG.debug("Graph backend: CPU (use_gpu=False).")
        return

    backend = cfg.backend
    LOG.info("Graph backend requested: %s", backend)

    if backend in ("auto", "nx-cugraph"):
        try:
            _enable_nx_cugraph_backend()
        except RuntimeError:
            if cfg.strict:
                LOG.exception("Failed to enable GPU backend (strict=True).")
                raise
            LOG.exception(
                "Failed to enable GPU backend; continuing with CPU backend."
            )
    elif backend == "cpu":
        LOG.info("Graph backend explicitly pinned to CPU.")
    else:
        LOG.warning(
            "Unknown graph backend '%s'; using CPU backend.", backend
        )
```

Key points:

* **Exactly one place** in your code touches `nx_cugraph` / backend API.
* If `use_gpu=False`, this function is a no-op.
* If GPU isn’t available and `strict=False`, you **log and fall back**; if `strict=True`, you raise.

---

## 4. Wire `maybe_enable_nx_gpu` into CLI & Prefect

### 4.1 CLI

In `src/codeintel/cli/main.py` (or wherever you parse args and construct config):

1. **Add CLI flags**:

Using `argparse`, you might add:

```python
def _add_graph_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nx-gpu",
        action="store_true",
        help="Prefer GPU backend for NetworkX (nx-cugraph) when available.",
    )
    parser.add_argument(
        "--nx-backend",
        choices=["auto", "cpu", "nx-cugraph"],
        default="auto",
        help="NetworkX backend selection (default: auto).",
    )
    parser.add_argument(
        "--nx-gpu-strict",
        action="store_true",
        help="Fail instead of falling back to CPU if GPU backend can't be enabled.",
    )
```

Call `_add_graph_backend_args(...)` on relevant subcommands: `pipeline run`, `docs export`, `analytics graph-*` etc.

2. **Map args → `GraphBackendConfig`**:

When building your top-level config or `PipelineContext`, do:

```python
from codeintel.config.models import GraphBackendConfig

def _build_graph_backend_config(args: argparse.Namespace) -> GraphBackendConfig:
    return GraphBackendConfig(
        use_gpu=bool(args.nx_gpu),
        backend=args.nx_backend,
        strict=bool(args.nx_gpu_strict),
    )
```

3. **Call `maybe_enable_nx_gpu` at startup**:

At the beginning of each subcommand handler:

```python
from codeintel.cli.nx_backend import maybe_enable_nx_gpu

def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    graph_backend = _build_graph_backend_config(args)
    maybe_enable_nx_gpu(graph_backend)

    # now build CodeIntelConfig / PipelineContext with graph_backend
    cfg = CodeIntelConfig(..., graph_backend=graph_backend)
    ...
```

Do the same for the docs export CLI entrypoint.

### 4.2 Prefect flow

In `src/codeintel/prefect/export_docs_flow.py` (or wherever you define the flow):

At the top of your `flow` run (or in an initialization task):

```python
from codeintel.config.models import GraphBackendConfig
from codeintel.cli.nx_backend import maybe_enable_nx_gpu

@flow
def export_docs_flow(args: ExportFlowArgs) -> None:
    graph_backend = GraphBackendConfig(
        use_gpu=args.nx_gpu,
        backend=args.nx_backend,
        strict=args.nx_gpu_strict,
    )
    maybe_enable_nx_gpu(graph_backend)
    ...
```

And make sure `ExportFlowArgs` includes the same three booleans/strings.

---

## 5. Integrate with `graphs.nx_views` & GraphService

The GPU backend we just wired is **global**, so most algorithms will automatically run on GPU once the NetworkX backend is set. But you also may want a per-graph `use_gpu` hint (as your doc suggests).

### 5.1 Add a `use_gpu` flag in GraphContext

In `analytics/graph_service.py`, your `GraphContext` currently has things like betweenness sampling `k`, eigen iteration limits, seed, etc.

Add a field:

```python
@dataclass(frozen=True)
class GraphContext:
    repo: str
    commit: str
    pagerank_weight: str | None
    betweenness_k_max: int | None
    eigen_max_iter: int | None
    seed: int | None
    use_gpu: bool = False
```

Adjust `build_graph_context` to accept a `use_gpu` parameter (or read from `GraphBackendConfig` later).

```python
def build_graph_context(
    cfg: GraphMetricsConfig,
    *,
    use_gpu: bool = False,
) -> GraphContext:
    return GraphContext(
        repo=cfg.repo,
        commit=cfg.commit,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_k_max=cfg.betweenness_k_max,
        eigen_max_iter=cfg.eigen_max_iter,
        seed=cfg.seed,
        use_gpu=use_gpu,
    )
```

Now GraphService can log or make GPU-aware decisions if you ever need per-algorithm differences.

### 5.2 Extend `graphs.nx_views` with `use_gpu` hooks

In `src/codeintel/analytics/graphs/nx_views.py`, your graph loaders probably look like:

```python
def load_call_graph(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> nx.DiGraph:
    df = con.execute("SELECT ...").fetch_df()
    G = nx.DiGraph()
    ...
    return G
```

Update their signatures and add a `_maybe_to_gpu_graph` helper:

```python
# src/codeintel/analytics/graphs/nx_views.py
from __future__ import annotations

import logging
from typing import TypeVar

import networkx as nx

LOG = logging.getLogger(__name__)
G = TypeVar("G", bound=nx.Graph)

def _maybe_to_gpu_graph(graph: G, *, use_gpu: bool) -> G:
    """
    Optionally convert a NetworkX graph to a GPU-backed graph.

    If use_gpu is False or the GPU backend is unavailable/unsupported,
    return the original graph.
    """
    if not use_gpu:
        return graph

    try:
        import nx_cugraph  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - environment specific
        LOG.debug("nx_cugraph not installed; leaving graph on CPU.")
        return graph

    try:
        # Depending on your nx_cugraph version, you may be able to wrap the graph
        # or create a new GPU-backed graph. Check your versions' docs.
        # Example / pseudo-code:
        #   gpu_graph = nx_cugraph.from_networkx(graph)
        # For now, we assume the global backend handles this and simply return.
        return graph
    except Exception:  # pragma: no cover - defensive
        LOG.exception("Failed to convert graph to GPU backend; using CPU graph.")
        return graph


def load_call_graph(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.DiGraph:
    df = con.execute(
        "SELECT src, dst, weight FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetch_df()
    G = nx.DiGraph()
    ...
    return _maybe_to_gpu_graph(G, use_gpu=use_gpu)


def load_import_graph(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    *,
    use_gpu: bool = False,
) -> nx.DiGraph:
    ...
    return _maybe_to_gpu_graph(G, use_gpu=use_gpu)

# and similarly for:
# - load_symbol_module_graph
# - load_symbol_function_graph
# - load_test_function_bipartite
# - load_config_module_bipartite
```

In practice, if you rely solely on **global backend selection**, `_maybe_to_gpu_graph` may just log and return the graph; the important part is that **graph-construction code is GPU-aware** and doesn’t need to be edited everywhere later.

---

## 6. Update analytics modules to pass `use_gpu`

Now that you have:

* Global backend control (`maybe_enable_nx_gpu`)
* `GraphBackendConfig` on the config / pipeline context
* `use_gpu` in `GraphContext` and `load_*` functions

…we just need to **thread the flag** through the analytics modules that build graphs.

### 6.1 Example: `analytics/graph_metrics.py`

Assuming its entrypoint is something like:

```python
def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
) -> None:
    con = gateway.con
    graph_ctx = graph_ctx or build_graph_context(cfg)
    ...
```

Add a `graph_backend: GraphBackendConfig | None` argument, and pass `use_gpu` when loading graphs and building `GraphContext`:

```python
from codeintel.config.models import GraphBackendConfig
from codeintel.analytics.graphs.nx_views import load_call_graph, load_import_graph
from codeintel.analytics.graph_service import build_graph_context

def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
    graph_backend: GraphBackendConfig | None = None,
) -> None:
    con = gateway.con
    use_gpu = bool(graph_backend.use_gpu) if graph_backend is not None else False

    graph_ctx = graph_ctx or build_graph_context(cfg, use_gpu=use_gpu)

    if context is not None and context.call_graph is not None:
        call_graph = context.call_graph
    else:
        call_graph = load_call_graph(con, cfg.repo, cfg.commit, use_gpu=use_gpu)

    if context is not None and context.import_graph is not None:
        import_graph = context.import_graph
    else:
        import_graph = load_import_graph(con, cfg.repo, cfg.commit, use_gpu=use_gpu)

    # then call GraphService.centrality_* as before...
```

### 6.2 Other analytics modules

Apply the same pattern to:

* `analytics/graph_metrics_ext.py`
* `analytics/module_graph_metrics_ext.py`
* `analytics/symbol_graph_metrics.py`
* `analytics/config_graph_metrics.py`
* `analytics/test_graph_metrics.py`
* `analytics/subsystem_graph_metrics.py`
* `analytics/graph_stats.py`
* `analytics/cfg_dfg_metrics.py` (for CFG/DFG centralities; they might just pass a `GraphContext` flag)

In each:

1. Add a `graph_backend: GraphBackendConfig | None = None` parameter.
2. Compute `use_gpu` once.
3. Pass `use_gpu` into any `load_*` calls and into `build_graph_context`.

### 6.3 Pipeline / Prefect steps

In `steps.py`, you already have steps like `GraphMetricsStep`, `ConfigGraphMetricsStep`, etc. They typically do:

```python
graph_ctx = build_graph_context(cfg)
compute_graph_metrics(gateway, cfg, context=_analytics_context(ctx), graph_ctx=graph_ctx)
```

Update to something like:

```python
graph_backend = ctx.config.graph_backend  # or ctx.graph_backend

graph_ctx = build_graph_context(cfg, use_gpu=graph_backend.use_gpu)
compute_graph_metrics(
    gateway,
    cfg,
    context=_analytics_context(ctx),
    graph_ctx=graph_ctx,
    graph_backend=graph_backend,
)
```

Do the same for other graph-heavy steps.

---

## 7. Testing & agent guidance

### 7.1 Tests

1. **Unit tests for `maybe_enable_nx_gpu`**:

   * With `use_gpu=False` → ensure it’s a no-op.
   * With `use_gpu=True`, `backend="auto"` and `nx_cugraph` missing → ensure that:

     * No exception when `strict=False`.
     * Exception when `strict=True`.

2. **Graph loader tests**:

   * Ensure `load_call_graph(..., use_gpu=True)` still returns an `nx.DiGraph` and that `_maybe_to_gpu_graph` is called.
   * You can monkeypatch or stub `nx_cugraph` to avoid requiring actual GPU in CI.

3. **End-to-end**:

   * Run a small pipeline with `--nx-gpu` and confirm:

     * No crashes.
     * Metrics tables still populated identically (or tolerably similarly).
     * Logs show GPU backend attempt.

### 7.2 Instructions for AI agents (docs / AGENTS)

Add a short section to `AGENTS.md` or a new `docs/improvement_plans/networkx_gpu_implementation.md`:

* **Never import `nx_cugraph` or call backend APIs from analytics modules.**
  All backend selection must go through `codeintel.cli.nx_backend.maybe_enable_nx_gpu`.

* **If you add a new graph analytics module**:

  * Accept `graph_backend: GraphBackendConfig | None`.
  * Compute `use_gpu = graph_backend.use_gpu if graph_backend else False`.
  * Pass `use_gpu` into `load_*` functions and `build_graph_context`.

* **If you need to change how GPU integration works** (e.g., networkx API changes), update **only**:

  * `GraphBackendConfig` (if new options are needed).
  * `_enable_nx_cugraph_backend` in `nx_backend.py`.
  * `_maybe_to_gpu_graph` in `graphs.nx_views.py`.

---

If you’d like, I can turn this into a concrete markdown improvement plan file (with TODO checkboxes and references to each module) that you can drop into `docs/improvement_plans/networkx_gpu_implementation.md` and point your agents at as the canonical “GPU backend” playbook.
