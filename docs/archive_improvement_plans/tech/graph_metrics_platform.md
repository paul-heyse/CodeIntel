Graph Metrics Platform
======================

Purpose
-------

Centralize NetworkX metric computation across analytics modules so graphs are loaded once, seeded deterministically, and metrics are comparable across tables.

Service surface
---------------

- `GraphContext`: repo/commit/now plus shared knobs (`betweenness_sample`, `eigen_max_iter`, `pagerank_weight`, `betweenness_weight`, `seed`).
- `GraphBundle`: memoizes graph loaders by name and returns each graph once per execution.
- Metric helpers:
  - `centrality_directed` / `centrality_undirected`: pagerank/betweenness/closeness/harmonic (+ eigen optional) with shared sampling and seeds.
  - `component_metadata`: weak/SCC ids, sizes, cycle flags, and layer per node.
  - `neighbor_stats`: in/out neighbor sets and weighted counts.
  - `projection_metrics`: weighted projection degree/clustering/betweenness for bipartite sides.
  - `structural_metrics`: clustering, triangles, core number, constraint, effective size, community ids for undirected graphs.
  - `to_decimal_id`: coercion helper for DuckDB DECIMAL identifiers.

Caller guidance
---------------

- Build a `GraphContext` from config and pass it to all helpers; keep weight keys consistent (`weight` for current analytics tables).
- Use `GraphBundle` to load call/import/symbol/config/test graphs once; reuse for all table writers in a run.
- Prefer helper outputs over direct NetworkX calls to ensure sampling, seeds, and fallbacks stay consistent.
- Keep table semantics aligned: continue using `Decimal` for GOIDs, and return empty dicts for empty graphs.

Defaults & logging
------------------

- Weights come from `GraphMetricsConfig` (`pagerank_weight` / `betweenness_weight`, default `"weight"`); seeds default to `0` and flow through every helper.
- Betweenness sampling starts from `GraphMetricsConfig.max_betweenness_sample` (default `200`) and is capped per module (call/import + extensions cap at 500, symbol/config/test at 1000, CFG/DFG at 100).
- Eigenvector iterations honor `GraphMetricsConfig.eigen_max_iter` and respect per-module caps where present.
- Empty or skipped projections use `log_empty_graph` / `log_projection_skipped` so operators can spot missing partitions instead of silently continuing.

Modules migrated
----------------

- Core graph metrics (`graph_metrics.py`, `graph_metrics_ext.py`, `module_graph_metrics_ext.py`).
- Symbol metrics (`symbol_graph_metrics.py`).
- CFG/DFG metrics (`cfg_dfg_metrics.py`).
- Subsystem, test bipartite, and config projections (`subsystem_graph_metrics.py`, `test_graph_metrics.py`, `config_graph_metrics.py`).

Extensibility
-------------

- Extend `GraphContext` if new knobs are needed; avoid per-caller constants.
- Add new metric bundles to `graph_service.py` and reuse them across modules.
- Keep seeds deterministic to preserve testability and cross-table comparability.***
