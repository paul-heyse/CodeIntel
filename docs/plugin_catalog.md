# Graph Plugin Catalog

Generated at: 2025-11-29T21:26:54.422328+00:00
Plugin count: 12

| Name | Stage | Severity | Enabled | Isolation | Scope-aware | Depends | Provides | Requires |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfg_metrics | cfg | fatal | yes | no | no | - | - | - |
| config_graph_metrics | config | fatal | yes | no | no | - | - | - |
| core_graph_metrics | core | fatal | yes | no | no | - | - | - |
| dfg_metrics | dfg | fatal | yes | no | no | - | - | - |
| graph_metrics_functions_ext | core | fatal | yes | no | no | - | - | - |
| graph_metrics_modules_ext | core | fatal | yes | no | no | - | - | - |
| graph_stats | stats | fatal | yes | no | no | - | - | - |
| subsystem_agreement | subsystem | fatal | yes | no | no | subsystem_graph_metrics, graph_metrics_modules_ext | - | - |
| subsystem_graph_metrics | subsystem | fatal | yes | no | no | - | - | - |
| symbol_graph_metrics_functions | symbol | fatal | yes | no | no | - | - | - |
| symbol_graph_metrics_modules | symbol | fatal | yes | no | no | - | - | - |
| test_graph_metrics | test | fatal | yes | no | no | - | - | - |

## Plugin Details
### cfg_metrics

- Description: Control-flow graph metrics for functions and blocks.
- Stage: cfg (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: -
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### config_graph_metrics

- Description: Config bipartite/projection graph metrics.
- Stage: config (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.config_graph_metrics_keys, analytics.config_graph_metrics_modules, analytics.config_projection_key_edges, analytics.config_projection_module_edges
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### core_graph_metrics

- Description: Core function/module graph metrics (centrality, neighbors, components).
- Stage: core (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.graph_metrics_functions, analytics.graph_metrics_modules
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### dfg_metrics

- Description: Data-flow graph metrics for functions and blocks.
- Stage: dfg (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: -
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### graph_metrics_functions_ext

- Description: Extended call graph metrics for functions.
- Stage: core (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.graph_metrics_functions_ext
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### graph_metrics_modules_ext

- Description: Extended import graph metrics for modules.
- Stage: core (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.graph_metrics_modules_ext
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### graph_stats

- Description: Global graph statistics for core graphs.
- Stage: stats (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.graph_stats
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### subsystem_agreement

- Description: Check agreement between subsystem labels and import communities.
- Stage: subsystem (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: subsystem_graph_metrics, graph_metrics_modules_ext; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.subsystem_agreement
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### subsystem_graph_metrics

- Description: Subsystem-level condensed import graph metrics.
- Stage: subsystem (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.subsystem_graph_metrics
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### symbol_graph_metrics_functions

- Description: Symbol graph metrics at the function level.
- Stage: symbol (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.symbol_graph_metrics_functions
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### symbol_graph_metrics_modules

- Description: Symbol graph metrics at the module level.
- Stage: symbol (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.symbol_graph_metrics_modules
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

### test_graph_metrics

- Description: Metrics over the test <-> function bipartite graph.
- Stage: test (severity: fatal)
- Enabled by default: yes
- Isolation: no (none)
- Scope-aware: no (supports: -)
- Depends on: -; Provides: -; Requires data: -
- Resource hints: none; Row count tables: analytics.test_graph_metrics_tests, analytics.test_graph_metrics_functions
- Options model: none (default: None)
- Config schema ref: none; Contracts: 0
- Version hash: n/a

## Plan Output Examples
The CLI and MCP plan surfaces return ordered plugins, skip reasons, dependency graph, and enriched metadata. Example:

```json
{
  "plan_id": "example-plan-id",
  "ordered_plugins": [
    "core_graph_metrics",
    "graph_metrics_modules_ext"
  ],
  "skipped_plugins": [
    {
      "name": "graph_stats",
      "reason": "disabled"
    }
  ],
  "dep_graph": {
    "core_graph_metrics": [],
    "graph_metrics_modules_ext": []
  },
  "plugin_metadata": {
    "core_graph_metrics": {
      "stage": "core",
      "severity": "fatal",
      "requires_isolation": false,
      "isolation_kind": null,
      "scope_aware": false,
      "supported_scopes": [],
      "depends_on": [],
      "provides": [],
      "requires": []
    }
  }
}
```

Manifest excerpts include correlation/run IDs, scope, isolation flags, and row counts:

```json
{
  "run_id": "example-run-id",
  "scope": {
    "paths": [
      "src/"
    ],
    "modules": [],
    "time_window": null
  },
  "plugins": [
    {
      "name": "core_graph_metrics",
      "status": "ok",
      "requires_isolation": false,
      "row_counts": {
        "analytics.graph_metrics_functions": 42
      },
      "contracts": []
    }
  ]
}
```

To regenerate this catalog:

- Build JSON/Markdown/HTML: `uv run python scripts/render_graph_plugin_catalog.py`
- Inspect plan ordering: `uv run codeintel graph plugins --plan --json` (shows dep graph, isolation, scope fields).
- Inspect manifest snippets: run graph metrics to produce `build/graph-metrics/manifest.json`.