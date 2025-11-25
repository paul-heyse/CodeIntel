Short version:
Yes, you can safely “turn GPU on everywhere” for NetworkX by enabling the `nx-cugraph` backend; unsupported algorithms just fall back to normal NetworkX.([GitHub][1]) Below is a concrete implementation plan tied to your actual code (the zips) and to the architecture doc.

---

## 1. Where NetworkX lives in your code (and where GPU matters most)

From the repo you uploaded, NetworkX is used in a few clear clusters:

### 1. Graph materialization helpers

**File:** `graphs/graphs/nx_views.py`
**Key functions:**

* `load_call_graph(...) -> nx.DiGraph`
* `load_import_graph(...) -> nx.DiGraph`
* `load_test_function_bipartite(...) -> nx.Graph`
* `load_config_module_bipartite(...) -> nx.Graph`
* `load_symbol_module_graph(...) -> nx.Graph`
* `load_symbol_function_graph(...) -> nx.Graph`

These are the *only* “official” entrypoints to build in‑memory graphs from DuckDB. Every analytics step below calls through here.

These graphs are often big for a non‑trivial repo (call graph, import graph, symbol graphs, test bipartite, config bipartite), so they’re prime GPU candidates.

---

### 2. Heavy offline analytics (high‑value GPU targets)

All of these run inside the `graph_metrics` step in `orchestration/orchestration/steps.py` and the Prefect task `t_graph_metrics` in `orchestration/orchestration/prefect_flow.py`:

```python
# orchestration/steps.py – GraphMetricsStep.run
compute_graph_metrics(gateway, cfg)
compute_graph_metrics_functions_ext(gateway, repo=ctx.repo, commit=ctx.commit)
compute_test_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
compute_cfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
compute_dfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
compute_graph_metrics_modules_ext(gateway, repo=ctx.repo, commit=ctx.commit)
compute_symbol_graph_metrics_modules(gateway, repo=ctx.repo, commit=ctx.commit)
compute_symbol_graph_metrics_functions(gateway, repo=ctx.repo, commit=ctx.commit)
compute_config_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
compute_subsystem_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
compute_subsystem_agreement(gateway, repo=ctx.repo, commit=ctx.commit)
compute_graph_stats(gateway, repo=ctx.repo, commit=ctx.commit)
```

**Files & key functions:**

* `analytics/analytics/graph_metrics.py`

  * `_centrality(graph, max_betweenness_sample)` → `nx.pagerank`, `nx.betweenness_centrality`, `nx.closeness_centrality`
* `analytics/analytics/graph_metrics_ext.py`

  * `_centralities(graph, undirected)` → `nx.betweenness_centrality`, `nx.closeness_centrality`, `nx.harmonic_centrality`, `nx.eigenvector_centrality`, `nx.core_number`, `nx.clustering`, `nx.triangles`
  * `compute_graph_metrics_functions_ext(...)`
* `analytics/analytics/module_graph_metrics_ext.py`

  * `_centralities(...)` → same family of centrality + `structuralholes.constraint/effective_size`
  * `_component_and_community(...)` → `nx.weakly_connected_components`, `nx.strongly_connected_components`, `nx.algorithms.community.asyn_lpa_communities`
  * `compute_graph_metrics_modules_ext(...)`
* `analytics/analytics/test_graph_metrics.py`

  * `_projection_bundle(...)` → `bipartite.weighted_projected_graph`, `nx.clustering`, `nx.betweenness_centrality`
  * `compute_test_graph_metrics(...)`
* `analytics/analytics/cfg_dfg_metrics.py`

  * `_compute_cfg_centralities(...)` → `nx.betweenness_centrality`, `nx.closeness_centrality`, `nx.eigenvector_centrality`
  * `_dfg_centralities(...)` → more betweenness/eigenvector
  * `compute_cfg_metrics(...)`, `compute_dfg_metrics(...)`
* `analytics/analytics/symbol_graph_metrics.py`

  * `_centrality_bundle(...)` → `nx.betweenness_centrality`, `nx.closeness_centrality`, `nx.eigenvector_centrality`, `nx.harmonic_centrality`, `nx.core_number`, `structuralholes.constraint/effective_size`, `nx.algorithms.community.asyn_lpa_communities`, `nx.connected_components`
* `analytics/analytics/config_graph_metrics.py`

  * `_projection_metrics(...)` → `community.asyn_lpa_communities`, `nx.betweenness_centrality`, `nx.closeness_centrality`
* `analytics/analytics/subsystem_graph_metrics.py`

  * `_subsystem_centralities(...)` → centrality over subsystem graph
  * `compute_subsystem_graph_metrics(...)`
* `analytics/analytics/graph_stats.py`

  * `compute_graph_stats(...)` → `nx.connected_components`, `nx.weakly_connected_components`, `nx.strongly_connected_components`, `nx.average_clustering`, `approximation.diameter`, `nx.average_shortest_path_length`

These are the “big whole‑repo graphs” where NX‑>GPU gives maximum win. They all use algorithms that are largely supported by `nx-cugraph`: betweenness, eigenvector, degree centrality, k‑core, clustering, triangles, components, PageRank, etc.([RAPIDS Docs][2])

Algorithms like `closeness_centrality`, `harmonic_centrality`, `structuralholes.*`, `asyn_lpa_communities`, and `average_shortest_path_length` are *not* in the nx‑cugraph supported list, so they’ll stay on CPU even with GPU enabled.([RAPIDS Docs][2])

---

### 3. Smaller / always‑CPU uses

* `graphs/graphs/cfg_builder.py`: uses `nx.DiGraph` to *construct* CFGs; per‑function graphs are tiny.
* `graphs/graphs/validation.py`: uses NetworkX for validation checks.
* `analytics/analytics/subsystems.py`: uses a custom label‑propagation algorithm and various `nx.Graph` helpers for the module‑affinity graph.
* `mcp/mcp/query_service.py`: uses NetworkX in interactive queries (`load_call_graph`, `load_import_graph`) to answer “neighbors” style questions.

Even if GPU is enabled globally, these mostly run on small graphs, and many of the algorithms aren’t backed by cuGraph anyway. It’s fine if they quietly remain CPU‑only.

---

## 2. Global GPU strategy (Stage 1: “zero‑code‑change”)

This stage is basically: install nx‑cugraph, flip one env var before running the pipeline, and let NetworkX’s backend system do its thing.

### 2.1 Dependencies

Add an **optional** dependency for GPU environments (e.g. in your `pyproject.toml` / `requirements-gpu.txt`):

```bash
# CUDA 12 example
pip install nx-cugraph-cu12 --extra-index-url https://pypi.nvidia.com
```

The nx‑cugraph docs recommend `networkx>=3.4` for best backend behavior.([RAPIDS Docs][3])

### 2.2 Configuration surface

Add a simple knob that both humans and the LLM agent can reason about:

* In `config/config/models.py`:

  ```python
  @dataclass(frozen=True)
  class GraphBackendConfig:
      """NetworkX backend preferences."""
      use_gpu: bool = True            # default: try GPU if present
      min_gpu_nodes: int = 5_000      # graphs smaller than this can stay on CPU
  ```

* Add this to:

  * `CodeIntelConfig` (root config),
  * `ExportArgs` in `orchestration/prefect_flow.py`,
  * CLI flags in `cli/cli/main.py` (e.g. `--nx-gpu / --no-nx-gpu`).

The LLM agent should treat this as **the single source of truth** for “should we try to use nx‑cugraph?”.

### 2.3 Enable nx‑cugraph automatically

In *one* place near process startup (good candidate: CLI main before calling `export_docs_flow`, and your server bootstrap), do:

```python
# cli/cli/main.py
def _maybe_enable_nx_gpu(cfg: GraphBackendConfig) -> None:
    if not cfg.use_gpu:
        return
    try:
        import importlib
        importlib.import_module("nx_cugraph")
    except ImportError:
        # GPU backend not installed; soft fallback to CPU
        return

    # Zero-code-change acceleration
    os.environ.setdefault("NX_CUGRAPH_AUTOCONFIG", "True")
    # Optional: be explicit with generic NetworkX backend settings
    os.environ.setdefault("NETWORKX_FALLBACK_TO_NX", "True")
```

`NX_CUGRAPH_AUTOCONFIG=True` tells NetworkX:

* “For any algorithm you support in nx‑cugraph, run it on the GPU; otherwise fall back to default NetworkX on CPU.”([GitHub][1])

This alone will GPU‑accelerate:

* Call‑graph centralities (`graph_metrics.py`, `graph_metrics_ext.py`),
* Import‑graph metrics (`module_graph_metrics_ext.py`, `subsystem_graph_metrics.py`),
* Symbol and config graph centralities (`symbol_graph_metrics.py`, `config_graph_metrics.py`),
* Test projection centralities (`test_graph_metrics.py`),
* Component/k‑core/triangle/cluster stats (`graph_stats.py`, others),

*without* touching your existing `import networkx as nx` calls.

### 2.4 Hook into Prefect and the internal pipeline

You have two execution paths:

1. **Prefect flow** (`export_docs_flow` in `orchestration/prefect_flow.py`):
   Before building `ctx = IngestionContext(...)`, call `_maybe_enable_nx_gpu(args.graph_backend)` once.

2. **Non‑Prefect internal runner** (`GraphMetricsStep.run` in `orchestration/steps.py`):
   At the top of `run`, call `_maybe_enable_nx_gpu(ctx.graph_backend)` if you allow that config to ride along in `PipelineContext`.

You *don’t* need to modify any of the analytics modules to get a first round of GPU acceleration.

---

## 3. Targeted tuning for the biggest graphs (Stage 2: type‑based dispatch)

If later you want to squeeze more performance (and give the LLM agent clear guidance where to do it), you can reduce per‑algorithm conversion overhead by converting each big graph to a GPU graph *once* and reusing it.

The nx‑cugraph docs recommend type‑based dispatch via `nx_cugraph.from_networkx(G)`; the resulting graph type automatically routes supported algorithms to cuGraph.([PyPI][4])

### 3.1 Add a helper in `graphs/nx_views.py`

```python
# graphs/graphs/nx_views.py
def _maybe_to_gpu_graph(graph: nx.Graph | nx.DiGraph, *, use_gpu: bool) -> nx.Graph | nx.DiGraph:
    if not use_gpu:
        return graph
    try:
        import nx_cugraph as nxcg  # type: ignore[import]
    except ImportError:
        log.info("nx_cugraph not installed; using CPU NetworkX")
        return graph
    try:
        return nxcg.from_networkx(graph)
    except Exception:
        log.warning("Failed to convert graph to nx_cugraph; using CPU NetworkX", exc_info=True)
        return graph
```

Then update loaders to accept an optional `use_gpu` flag and call `_maybe_to_gpu_graph`:

```python
def load_call_graph(gateway: StorageGateway, repo: str, commit: str, *, use_gpu: bool = False):
    ...
    graph = nx.DiGraph()
    # build as today ...
    return _maybe_to_gpu_graph(graph, use_gpu=use_gpu)
```

Do the same for:

* `load_import_graph(...)`
* `load_symbol_module_graph(...)`
* `load_symbol_function_graph(...)`
* `load_config_module_bipartite(...)`
* `load_test_function_bipartite(...)`

Call sites that you *don’t* want on GPU (CFG builder, validation, some ad‑hoc diagnostics) can keep the default `use_gpu=False`.

### 3.2 Wire `use_gpu` through the analytics

Update the heavy analytics functions to request GPU graph types via `use_gpu=True` for whole‑repo graphs:

* `analytics/graph_metrics.py`

  * `graph = load_call_graph(gateway, cfg.repo, cfg.commit, use_gpu=cfg_backend.use_gpu)`
* `analytics/graph_metrics_ext.py`

  * `graph = load_call_graph(...)`
* `analytics/module_graph_metrics_ext.py`

  * `graph = load_import_graph(...)`
* `analytics/symbol_graph_metrics.py`

  * `load_symbol_module_graph(...)`, `load_symbol_function_graph(...)`
* `analytics/config_graph_metrics.py`

  * `load_config_module_bipartite(...)`
* `analytics/test_graph_metrics.py`

  * `load_test_function_bipartite(...)`
* `analytics/graph_stats.py`

  * All `load_*` calls there.
* `analytics/subsystem_graph_metrics.py`

  * `import_graph = load_import_graph(..., use_gpu=cfg_backend.use_gpu)`

This makes it explicit to the LLM agent: **“If you’re adding a new whole‑repo graph analytic here, either set `use_gpu=True` or pass `backend="cugraph"` for supported algorithms.”**

You can keep CFG/DFG metrics CPU‑only; the per‑function graphs are tiny and dominated by Python overhead anyway.

---

## 4. What to tell the LLM programming agent (rules of thumb)

Here’s the “contract” you can embed in docs, comments, or system prompts for your AI agent:

1. **Global behavior**

   * Treat `GraphBackendConfig.use_gpu` as the master switch for NetworkX GPU usage.
   * Assume `NX_CUGRAPH_AUTOCONFIG=True` has been set when `use_gpu=True` on a GPU‑capable machine.

2. **When building graphs**

   * Always materialize graphs via helpers in `codeintel.graphs.nx_views` (`load_call_graph`, `load_import_graph`, etc.), *not* ad‑hoc `nx.Graph()` calls for analytics.
   * If your code runs on a whole‑repo graph (call graph, import graph, symbol/ config/test graphs), pass `use_gpu=True` into the `load_*` helper unless there’s a specific reason not to.

3. **When calling algorithms**

   * Prefer standard NetworkX API (e.g. `nx.betweenness_centrality`, `nx.eigenvector_centrality`, `nx.core_number`, `nx.clustering`, `nx.triangles`, `nx.pagerank`). These are supported by nx‑cugraph and will offload automatically.([RAPIDS Docs][2])
   * It’s OK to call algorithms that nx‑cugraph doesn’t support (`nx.closeness_centrality`, `nx.harmonic_centrality`, `nx.algorithms.community.asyn_lpa_communities`, structural holes, dominance, etc.); they will run on CPU even when GPU is enabled.
   * For new *expensive* algorithms over large graphs, either:

     * check they’re in the nx‑cugraph supported list, **or**
     * sample / approximate when graph is huge (you already do this with `CENTRALITY_SAMPLE_LIMIT`, `MAX_BETWEENNESS_NODES`).

4. **When adding new analytics steps**

   * If the step operates on `graph.*` tables and scales with repo size, plug it into `GraphMetricsStep` / `t_graph_metrics` so it runs under the GPU‑enabled phase.
   * Always push results back into `analytics.*` and then into `docs.*`; don’t do runtime graph recomputation in the MCP server for heavy metrics.

---

## 5. About your “no harm enabling it everywhere” assumption

You’re basically right:

* If `nx-cugraph` isn’t installed or a function isn’t supported, NetworkX simply falls back to the standard implementation.([RAPIDS Docs][3])
* For very small graphs (e.g., per‑function CFG/DFG) GPU can be slower because kernel launch overhead dominates; but your architecture already keeps those graphs small and you’re not using especially expensive algorithms there, so it’s fine if they stay CPU.

So the safe default is:

* **Turn on `NX_CUGRAPH_AUTOCONFIG` globally when `GraphBackendConfig.use_gpu=True`.**
* Let the backend system and your `use_gpu` flags in `nx_views` decide what actually uses the GPU.

If you’d like, I can next sketch pseudo‑diffs for the exact files (`nx_views.py`, `prefect_flow.py`, `steps.py`, and `config/models.py`) that the LLM agent should apply, keeping the changes tightly scoped.

[1]: https://github.com/rapidsai/nx-cugraph/?utm_source=chatgpt.com "rapidsai/nx-cugraph: GPU Accelerated Backend for NetworkX"
[2]: https://docs.rapids.ai/api/cugraph/stable/nx_cugraph/supported-algorithms/?utm_source=chatgpt.com "Supported Algorithms — cugraph-docs 25.10.00 ..."
[3]: https://docs.rapids.ai/api/cugraph/stable/nx_cugraph/?utm_source=chatgpt.com "nx-cugraph"
[4]: https://pypi.org/project/nx-cugraph-cu11/?utm_source=chatgpt.com "nx-cugraph-cu11 - GPU Backend for NetworkX"
