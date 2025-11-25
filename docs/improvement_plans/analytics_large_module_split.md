Here we go again 😄 — this is a good one.

Below is a concrete, repo-targeted implementation plan for splitting the “god modules” in `src/codeintel/analytics/` into cohesive submodules, with directory layout, step-by-step refactor strategy, and example code stubs you can hand to an AI implementation agent.

I’ll focus on:

* `analytics/functions.py`
* `analytics/subsystems.py`
* `analytics/cfg_dfg_metrics.py`
* `analytics/tests_analytics.py`

and organize them into:

* `analytics/functions/`
* `analytics/subsystems/`
* `analytics/cfg_dfg/`
* `analytics/tests/`

The overall theme:

> **Preserve the external API, move responsibilities into smaller modules, and keep each module “single purpose”: compute X, not compute X + load Y + write Z.**

---

## 0. High-level goals & constraints

Design intent:

* **No semantic changes** initially — just file layout + imports. You can do follow-on cleanup later.
* **Minimize breakage**: keep the top-level “public API” (functions called from orchestration, CLI, Prefect) in stable places (via re-exports).
* **Make domains obvious**: functions-related analytics live under `analytics/functions/`, test-related under `analytics/tests/`, etc.
* **Keep things LLM-friendly**: each file should be small and self-contained enough that a single agent can “load it into working memory” and be confident they understand the whole thing.

---

## 1. Proposed directory layout

Target structure (only showing analytics bits):

```text
src/codeintel/analytics/
    __init__.py

    # Functions analytics
    functions/
        __init__.py
        config.py          # config/option dataclasses for function analytics
        metrics.py         # compute_function_metrics_and_types (main entrypoint)
        typedness.py       # typedness buckets + flags
        parsing.py         # thin adapters around function_parsing + span_resolver
        validation.py      # ValidationReporter and function_validation helpers

    # Subsystems analytics
    subsystems/
        __init__.py
        affinity.py        # module affinity graph and cluster assignment
        edge_stats.py      # subsystem edge statistics
        risk.py            # risk aggregation per subsystem
        materialize.py     # row building and DB writes

    # CFG/DFG analytics
    cfg_dfg/
        __init__.py
        cfg_core.py        # building CFGs per function and non-centrality metrics
        dfg_core.py        # building DFGs per function and non-centrality metrics
        graph_metrics.py   # CFG/DFG graph centralities (using GraphService)
        materialize.py     # row building and DB writes

    # Tests analytics
    tests/
        __init__.py
        coverage_edges.py  # formerly tests_analytics.py
        profiles.py        # formerly test_profiles.py
        graph_metrics.py   # formerly test_graph_metrics.py
```

Plus **thin compatibility shims**:

* `analytics/functions.py`
* `analytics/subsystems.py`
* `analytics/cfg_dfg_metrics.py`
* `analytics/tests_analytics.py`

that just import and delegate to the new packages, so you don’t have to update every caller all at once.

---

## 2. Splitting `analytics/functions.py` → `analytics/functions/`

### 2.1. Responsibilities today

From your current code, `analytics/functions.py` roughly does:

* Defines config/options for the function analytics pass (per-repo + per-commit).
* Computes per-function AST metrics (complexity, statement counts, etc.).
* Computes **typedness** (how fully typed, missing annotations, Any usage).
* Writes rows into `analytics.function_metrics`, `analytics.function_types`.
* Handles **span resolution & parsing** (via `function_parsing` / `span_resolver`).
* Handles **validation** and populates `analytics.function_validation` (via `ValidationReporter`).

That’s too much for one file. We’ll slice it by concern.

### 2.2. Step 1 – Create the package skeleton

Create:

```python
# src/codeintel/analytics/functions/__init__.py
from __future__ import annotations

"""
Function-level analytics: metrics, typedness, validation.

This package replaces the old `analytics.functions` monolith.
Use `compute_function_metrics_and_types` as the main entrypoint.
"""

from .config import FunctionAnalyticsConfig, FunctionAnalyticsOptions  # whatever you use
from .metrics import compute_function_metrics_and_types
from .validation import ValidationReporter

__all__ = [
    "FunctionAnalyticsConfig",
    "FunctionAnalyticsOptions",
    "compute_function_metrics_and_types",
    "ValidationReporter",
]
```

And then **shrink the old module** to a shim:

```python
# src/codeintel/analytics/functions.py  (compat wrapper)
from __future__ import annotations

"""
Backwards-compat wrapper.

New code should import from `codeintel.analytics.functions.*` submodules instead.
"""

from .functions import (
    FunctionAnalyticsConfig,
    FunctionAnalyticsOptions,
    ValidationReporter,
    compute_function_metrics_and_types,
)

__all__ = [
    "FunctionAnalyticsConfig",
    "FunctionAnalyticsOptions",
    "ValidationReporter",
    "compute_function_metrics_and_types",
]
```

This keeps all existing imports like
`from codeintel.analytics.functions import compute_function_metrics_and_types`
working.

### 2.3. Step 2 – Extract config/options into `config.py`

Anything like `FunctionAnalyticsConfig`, `FunctionAnalyticsOptions`, or similar option dataclasses should move to `config.py`:

```python
# src/codeintel/analytics/functions/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from codeintel.storage.gateway import StorageGateway

@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    include_types: bool = True
    include_metrics: bool = True
    include_validation: bool = True

@dataclass(frozen=True)
class FunctionAnalyticsConfig:
    repo: str
    commit: str
    options: FunctionAnalyticsOptions

    @classmethod
    def for_current_snapshot(
        cls,
        repo: str,
        commit: str,
        *,
        include_types: bool = True,
        include_metrics: bool = True,
        include_validation: bool = True,
    ) -> "FunctionAnalyticsConfig":
        return cls(
            repo=repo,
            commit=commit,
            options=FunctionAnalyticsOptions(
                include_types=include_types,
                include_metrics=include_metrics,
                include_validation=include_validation,
            ),
        )
```

Then your orchestrator call sites become simpler:

```python
from codeintel.analytics.functions import compute_function_metrics_and_types, FunctionAnalyticsConfig

cfg = FunctionAnalyticsConfig.for_current_snapshot(repo, commit)
compute_function_metrics_and_types(gateway, cfg, context=analytics_context)
```

### 2.4. Step 3 – Extract typedness into `typedness.py`

All the logic that answers “how typed is this function?” moves here:

```python
# src/codeintel/analytics/functions/typedness.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class TypednessStats:
    total_params: int
    annotated_params: int
    returns_annotated: bool
    any_params: int
    any_return: bool

    @property
    def param_coverage(self) -> float:
        if self.total_params == 0:
            return 1.0
        return self.annotated_params / self.total_params

    @property
    def fully_typed(self) -> bool:
        return (
            self.param_coverage == 1.0
            and self.returns_annotated
            and self.any_params == 0
            and not self.any_return
        )

def compute_typedness_stats(
    *,
    param_annotations: Mapping[str, object],
    return_annotation: object | None,
    param_any_flags: Mapping[str, bool],
    return_is_any: bool,
) -> TypednessStats:
    total_params = len(param_annotations)
    annotated = sum(1 for v in param_annotations.values() if v is not None)
    any_params = sum(1 for v in param_any_flags.values() if v)
    return TypednessStats(
        total_params=total_params,
        annotated_params=annotated,
        returns_annotated=return_annotation is not None,
        any_params=any_params,
        any_return=return_is_any,
    )
```

Then in `metrics.py` you do:

```python
from .typedness import compute_typedness_stats
```

and use it while building rows for `analytics.function_metrics` / `analytics.function_types`.

You can also move any bucket mapping (`TypednessBucket`, `_typedness_flags`) into this module, so there is literally one place to look for typedness semantics.

### 2.5. Step 4 – Parsing & span resolution adapters in `parsing.py`

You already have `function_parsing.py` and `span_resolver.py` in `analytics/`. Wrap them:

```python
# src/codeintel/analytics/functions/parsing.py
from __future__ import annotations

from codeintel.analytics.function_parsing import ParsedFunction, parse_functions_in_module
from codeintel.analytics.span_resolver import resolve_span

__all__ = [
    "ParsedFunction",
    "parse_functions_in_module",
    "resolve_span",
]
```

Now `metrics.py` can just depend on `from .parsing import ParsedFunction, parse_functions_in_module, resolve_span` and you have a clear “function analytics sees these parsing APIs” seam.

Later you can move `function_parsing.py` and `span_resolver.py` into this package too, but that’s optional.

### 2.6. Step 5 – Validation into `validation.py`

All the `ValidationReporter` stuff and logic that inserts function validation rows moves here.

```python
# src/codeintel/analytics/functions/validation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from codeintel.models.rows import FunctionValidationRow, function_validation_row_to_tuple
from codeintel.storage.gateway import StorageGateway

@dataclass
class ValidationCounters:
    parse_failed: int = 0
    span_not_found: int = 0
    unknown_functions: int = 0

@dataclass
class ValidationReporter:
    repo: str
    commit: str
    counters: ValidationCounters = field(default_factory=ValidationCounters)
    rows: list[FunctionValidationRow] = field(default_factory=list)

    def record(
        self,
        function_goid_h128: int | None,
        kind: str,
        message: str,
    ) -> None:
        if kind == "parse_failed":
            self.counters.parse_failed += 1
        elif kind == "span_not_found":
            self.counters.span_not_found += 1
        elif kind == "unknown_function":
            self.counters.unknown_functions += 1

        row: FunctionValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "function_goid_h128": function_goid_h128,
            "kind": kind,
            "message": message,
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        if not self.rows:
            return
        tuples = [function_validation_row_to_tuple(row) for row in self.rows]
        con = gateway.con
        con.execute(
            """
            INSERT INTO analytics.function_validation (
                repo, commit, function_goid_h128, kind, message
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()
```

### 2.7. Step 6 – Main orchestrator in `metrics.py`

Finally, the main entrypoint that glues everything together:

```python
# src/codeintel/analytics/functions/metrics.py
from __future__ import annotations

from typing import Iterable

from codeintel.storage.gateway import StorageGateway
from codeintel.analytics.context import AnalyticsContext
from codeintel.models.rows import (
    FunctionMetricsRow,
    FunctionTypesRow,
    function_metrics_row_to_tuple,
    function_types_row_to_tuple,
)
from .config import FunctionAnalyticsConfig
from .parsing import parse_functions_in_module, resolve_span
from .typedness import compute_typedness_stats
from .validation import ValidationReporter


def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    """
    Main entrypoint: populate analytics.function_metrics and analytics.function_types.
    """
    con = gateway.con
    reporter = ValidationReporter(cfg.repo, cfg.commit)

    # 1) Load module-level catalog (modules + functions)
    #    You already have helpers for this in your analytics.context or ingest outputs.
    modules = _load_modules_for_repo(con, cfg.repo, cfg.commit)

    metrics_rows: list[FunctionMetricsRow] = []
    type_rows: list[FunctionTypesRow] = []

    for module in modules:
        parsed = parse_functions_in_module(module)
        for func in parsed.functions:
            # resolve spans, map to GOID, etc.
            typedness = compute_typedness_stats(
                param_annotations=func.param_annotations,
                return_annotation=func.return_annotation,
                param_any_flags=func.param_any_flags,
                return_is_any=func.return_is_any,
            )

            # build metrics row
            metrics_rows.append(_build_metrics_row(cfg, func, typedness))
            # build types row (if enabled)
            if cfg.options.include_types:
                type_rows.extend(_build_types_rows(cfg, func))

    _write_metrics(gateway, metrics_rows)
    _write_types(gateway, type_rows)
    reporter.flush(gateway)
```

`_load_modules_for_repo`, `_build_metrics_row`, `_build_types_rows`, `_write_metrics`, `_write_types` are all previously in `functions.py`; they move here, possibly split into small helpers.

---

## 3. Splitting `analytics/subsystems.py` → `analytics/subsystems/`

### 3.1. Responsibilities today

`analytics/subsystems.py` currently does something like:

* Build a **module affinity graph** (modules as nodes, edges weighted by co-change, call/ import interactions, etc.).
* Run **clustering / label propagation** to assign modules to subsystem IDs.
* Compute **edge stats** between subsystems (internal vs external edges, densities).
* Compute **risk summaries** per subsystem (based on hotspots, function metrics, history).
* Materialize rows for `analytics.subsystems` and `analytics.subsystem_modules`.

We want to separate those concerns.

### 3.2. Step 1 – New package skeleton

```python
# src/codeintel/analytics/subsystems/__init__.py
from __future__ import annotations

"""
Subsystem analytics.

High-level entrypoint: `compute_subsystems` builds subsystem assignments,
edge stats, and risk metrics, and writes analytics.subsystems + analytics.subsystem_modules.
"""

from .affinity import build_subsystem_affinity_graph
from .materialize import compute_subsystems

__all__ = [
    "build_subsystem_affinity_graph",
    "compute_subsystems",
]
```

Shim:

```python
# src/codeintel/analytics/subsystems.py (compat wrapper)
from __future__ import annotations

from .subsystems import compute_subsystems

__all__ = ["compute_subsystems"]
```

### 3.3. Step 2 – `affinity.py`: module affinity graph + clustering

```python
# src/codeintel/analytics/subsystems/affinity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import networkx as nx

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import GraphContext
from codeintel.analytics.graphs.nx_views import (
    load_import_graph,
    load_call_graph,
)

@dataclass(frozen=True)
class SubsystemAssignment:
    module: str
    subsystem_id: int

@dataclass(frozen=True)
class SubsystemAffinityResult:
    graph: nx.DiGraph
    assignments: dict[str, int]  # module -> subsystem_id


def build_subsystem_affinity_graph(
    ctx: AnalyticsContext,
    graph_ctx: GraphContext,
) -> SubsystemAffinityResult:
    """
    Build a weighted module-affinity graph and cluster it into subsystems.

    Uses module import graph, call graph, and module metadata from AnalyticsContext.
    """
    import_graph = load_import_graph(ctx.storage_gateway.con)
    call_graph = load_call_graph(ctx.storage_gateway.con)

    # 1) Build module-level aggregated graph
    module_graph = _module_affinity_graph(import_graph, call_graph, ctx.module_map)

    # 2) Run label propagation / community detection
    assignments = _cluster_modules(module_graph, graph_ctx)

    return SubsystemAffinityResult(graph=module_graph, assignments=assignments)


def _module_affinity_graph(
    import_graph: nx.DiGraph,
    call_graph: nx.DiGraph,
    module_map: Mapping[str, object],
) -> nx.DiGraph:
    # Implementation: fold function-level graphs -> module-level weights,
    # normalize, drop tiny edges, etc.
    graph = nx.DiGraph()
    # ... your existing logic moved here ...
    return graph


def _cluster_modules(
    module_graph: nx.DiGraph,
    graph_ctx: GraphContext,
) -> dict[str, int]:
    # Use GraphService community algorithms or label propagation;
    # this was previously in subsystems.py.
    # Return a stable mapping: module -> subsystem_id.
    assignments: dict[str, int] = {}
    # ...
    return assignments
```

### 3.4. Step 3 – `edge_stats.py`: subsystem edge stats helpers

This is where things like `_subsystem_edge_stats` move:

```python
# src/codeintel/analytics/subsystems/edge_stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import networkx as nx

@dataclass(frozen=True)
class SubsystemEdgeStats:
    internal_edges: int
    external_edges: int
    internal_weight: float
    external_weight: float

def compute_edge_stats(
    module_graph: nx.DiGraph,
    module_to_subsystem: Dict[str, int],
) -> Dict[int, SubsystemEdgeStats]:
    """
    Compute edge stats per subsystem.

    Returns
    -------
    Dict[int, SubsystemEdgeStats]
        subsystem_id -> stats
    """
    # Move your current `_subsystem_edge_stats` logic into here.
    stats: Dict[int, SubsystemEdgeStats] = {}
    # ...
    return stats
```

### 3.5. Step 4 – `risk.py`: subsystem risk aggregation

All the “aggregate hotspots / risk scores / churn / coverage per subsystem” logic moves here:

```python
# src/codeintel/analytics/subsystems/risk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from codeintel.storage.gateway import StorageGateway

@dataclass(frozen=True)
class SubsystemRisk:
    subsystem_id: int
    risk_score: float
    hotspot_count: int
    churn_score: float
    test_coverage: float

def compute_subsystem_risk(
    gateway: StorageGateway,
    module_to_subsystem: Dict[str, int],
) -> Dict[int, SubsystemRisk]:
    """
    Aggregates per-module / per-function risk into subsystem-level metrics.

    Reads hotspots, history, coverage, etc. from DuckDB via gateway.
    """
    # SELECT from analytics.hotspots, function_history, coverage_* etc.
    # Aggregate into per-subsystem metrics.
    risks: Dict[int, SubsystemRisk] = {}
    # ...
    return risks
```

### 3.6. Step 5 – `materialize.py`: row building + DB writes

This is the new “main entrypoint” for subsystem analytics:

```python
# src/codeintel/analytics/subsystems/materialize.py
from __future__ import annotations

from typing import Dict

from codeintel.storage.gateway import StorageGateway
from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import GraphContext
from codeintel.models.rows import (
    SubsystemRow,
    SubsystemModuleRow,
    subsystem_row_to_tuple,
    subsystem_module_row_to_tuple,
)
from .affinity import build_subsystem_affinity_graph
from .edge_stats import compute_edge_stats
from .risk import compute_subsystem_risk


def compute_subsystems(
    gateway: StorageGateway,
    ctx: AnalyticsContext,
    graph_ctx: GraphContext,
) -> None:
    """
    End-to-end subsystem analytics:

    - Build module-affinity graph
    - Cluster modules into subsystems
    - Compute edge stats and risk summaries
    - Persist analytics.subsystems + analytics.subsystem_modules
    """
    con = gateway.con

    affinity = build_subsystem_affinity_graph(ctx, graph_ctx)
    module_graph = affinity.graph
    module_to_subsystem = affinity.assignments

    edge_stats = compute_edge_stats(module_graph, module_to_subsystem)
    risks = compute_subsystem_risk(gateway, module_to_subsystem)

    subsystem_rows: list[SubsystemRow] = []
    module_rows: list[SubsystemModuleRow] = []

    # Turn everything into rows
    for module, subsystem_id in module_to_subsystem.items():
        # module row
        module_rows.append(
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=subsystem_id,
                module=module,
                # other fields...
            )
        )

    for subsystem_id, stats in edge_stats.items():
        risk = risks.get(subsystem_id)
        subsystem_rows.append(
            SubsystemRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id=subsystem_id,
                risk_score=risk.risk_score if risk else 0.0,
                hotspot_count=risk.hotspot_count if risk else 0,
                # internal/external edges, churn, coverage…
            )
        )

    # Write to DB
    con.executemany(
        """
        INSERT INTO analytics.subsystem_modules (...)
        VALUES (...)
        """,
        [subsystem_module_row_to_tuple(row) for row in module_rows],
    )

    con.executemany(
        """
        INSERT INTO analytics.subsystems (...)
        VALUES (...)
        """,
        [subsystem_row_to_tuple(row) for row in subsystem_rows],
    )
```

Your existing `subsystems.py` functions should map almost directly into these helpers.

---

## 4. Splitting `analytics/cfg_dfg_metrics.py` → `analytics/cfg_dfg/`

### 4.1. Responsibilities today

`cfg_dfg_metrics.py` currently:

* Builds per-function **CFG** and **DFG** graphs from tables like `cfg_blocks`, `cfg_edges`, `dfg_edges`.

* Computes a mix of:

  * **Structural metrics** (blocks, loops, branching, path counts).
  * **Centralities** using GraphService (betweenness, eigenvector, etc.).
  * Possibly some risk-ish metrics (e.g. high centrality loops / blocks).

* Writes rows to `analytics.cfg_function_metrics`, `analytics.dfg_function_metrics`, etc.

We’ll separate “graph build + structural metrics” from “centralities” and “materialization”.

### 4.2. Package skeleton

```python
# src/codeintel/analytics/cfg_dfg/__init__.py
from __future__ import annotations

"""
CFG/DFG analytics per function.

Entry points:
- compute_cfg_dfg_metrics: populate analytics.cfg_* and analytics.dfg_* tables.
"""

from .materialize import compute_cfg_dfg_metrics

__all__ = ["compute_cfg_dfg_metrics"]
```

Shim:

```python
# src/codeintel/analytics/cfg_dfg_metrics.py (compat)
from __future__ import annotations

from .cfg_dfg import compute_cfg_dfg_metrics

__all__ = ["compute_cfg_dfg_metrics"]
```

### 4.3. `cfg_core.py`: build CFGs and non-centrality metrics

```python
# src/codeintel/analytics/cfg_dfg/cfg_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import networkx as nx

from codeintel.storage.gateway import StorageGateway

@dataclass(frozen=True)
class CfgGraph:
    function_goid_h128: int
    graph: nx.DiGraph

@dataclass(frozen=True)
class CfgStructuralMetrics:
    function_goid_h128: int
    block_count: int
    loop_count: int
    branching_factor: float
    max_depth: int


def load_cfg_graphs(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[CfgGraph]:
    """
    Construct per-function CFG graphs from cfg_blocks + cfg_edges tables.
    """
    con = gateway.con
    # SELECT from cfg_blocks/cfg_edges and stitch graphs
    graphs: list[CfgGraph] = []
    # ...
    return graphs


def compute_cfg_structural_metrics(
    graphs: Iterable[CfgGraph],
) -> Dict[int, CfgStructuralMetrics]:
    """
    Compute non-centrality CFG metrics for each function.
    """
    metrics: Dict[int, CfgStructuralMetrics] = {}
    for cfg in graphs:
        g = cfg.graph
        # existing logic: loop detection, branching, depth, etc.
        metrics[cfg.function_goid_h128] = CfgStructuralMetrics(
            function_goid_h128=cfg.function_goid_h128,
            block_count=g.number_of_nodes(),
            loop_count=_estimate_loop_count(g),
            branching_factor=_branching_factor(g),
            max_depth=_max_depth(g),
        )
    return metrics
```

### 4.4. `dfg_core.py`: build DFGs and non-centrality metrics

```python
# src/codeintel/analytics/cfg_dfg/dfg_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import networkx as nx

from codeintel.storage.gateway import StorageGateway

@dataclass(frozen=True)
class DfgGraph:
    function_goid_h128: int
    graph: nx.DiGraph

@dataclass(frozen=True)
class DfgStructuralMetrics:
    function_goid_h128: int
    node_count: int
    edge_count: int
    max_fan_in: int
    max_fan_out: int


def load_dfg_graphs(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[DfgGraph]:
    """
    Construct per-function DFG graphs from dfg_edges table.
    """
    con = gateway.con
    graphs: list[DfgGraph] = []
    # ...
    return graphs


def compute_dfg_structural_metrics(
    graphs: Iterable[DfgGraph],
) -> Dict[int, DfgStructuralMetrics]:
    metrics: Dict[int, DfgStructuralMetrics] = {}
    for dfg in graphs:
        g = dfg.graph
        metrics[dfg.function_goid_h128] = DfgStructuralMetrics(
            function_goid_h128=dfg.function_goid_h128,
            node_count=g.number_of_nodes(),
            edge_count=g.number_of_edges(),
            max_fan_in=max((g.in_degree(n) for n in g.nodes()), default=0),
            max_fan_out=max((g.out_degree(n) for n in g.nodes()), default=0),
        )
    return metrics
```

### 4.5. `graph_metrics.py`: centralities via GraphService

All the `centrality_directed` calls move here:

```python
# src/codeintel/analytics/cfg_dfg/graph_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from codeintel.analytics.graph_service import GraphContext, centrality_directed
from .cfg_core import CfgGraph
from .dfg_core import DfgGraph

@dataclass(frozen=True)
class CfgCentralityMetrics:
    function_goid_h128: int
    betweenness: float
    eigenvector: float

@dataclass(frozen=True)
class DfgCentralityMetrics:
    function_goid_h128: int
    betweenness: float
    eigenvector: float


def compute_cfg_centralities(
    graphs: Iterable[CfgGraph],
    graph_ctx: GraphContext,
) -> Dict[int, CfgCentralityMetrics]:
    metrics: Dict[int, CfgCentralityMetrics] = {}
    for cfg in graphs:
        g = cfg.graph
        central = centrality_directed(g, ctx=graph_ctx)
        # pick a representative node-level aggregate, or use entry block
        entry = _entry_node(g)
        metrics[cfg.function_goid_h128] = CfgCentralityMetrics(
            function_goid_h128=cfg.function_goid_h128,
            betweenness=central.betweenness.get(entry, 0.0),
            eigenvector=central.eigenvector.get(entry, 0.0)
            if central.eigenvector is not None
            else 0.0,
        )
    return metrics


def compute_dfg_centralities(
    graphs: Iterable[DfgGraph],
    graph_ctx: GraphContext,
) -> Dict[int, DfgCentralityMetrics]:
    metrics: Dict[int, DfgCentralityMetrics] = {}
    for dfg in graphs:
        g = dfg.graph
        central = centrality_directed(g, ctx=graph_ctx)
        # e.g., aggregate or use source node; copy from existing code semantics
        repr_node = _representative_node(g)
        metrics[dfg.function_goid_h128] = DfgCentralityMetrics(
            function_goid_h128=dfg.function_goid_h128,
            betweenness=central.betweenness.get(repr_node, 0.0),
            eigenvector=central.eigenvector.get(repr_node, 0.0)
            if central.eigenvector is not None
            else 0.0,
        )
    return metrics
```

### 4.6. `materialize.py`: orchestrate and write rows

```python
# src/codeintel/analytics/cfg_dfg/materialize.py
from __future__ import annotations

from codeintel.analytics.graph_service import GraphContext
from codeintel.storage.gateway import StorageGateway
from codeintel.models.rows import (
    CfgFunctionMetricsRow,
    DfgFunctionMetricsRow,
    cfg_function_metrics_row_to_tuple,
    dfg_function_metrics_row_to_tuple,
)
from .cfg_core import load_cfg_graphs, compute_cfg_structural_metrics
from .dfg_core import load_dfg_graphs, compute_dfg_structural_metrics
from .graph_metrics import compute_cfg_centralities, compute_dfg_centralities


def compute_cfg_dfg_metrics(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    graph_ctx: GraphContext,
) -> None:
    con = gateway.con

    cfg_graphs = load_cfg_graphs(gateway, repo, commit)
    dfg_graphs = load_dfg_graphs(gateway, repo, commit)

    cfg_struct = compute_cfg_structural_metrics(cfg_graphs)
    dfg_struct = compute_dfg_structural_metrics(dfg_graphs)

    cfg_cent = compute_cfg_centralities(cfg_graphs, graph_ctx)
    dfg_cent = compute_dfg_centralities(dfg_graphs, graph_ctx)

    cfg_rows: list[CfgFunctionMetricsRow] = []
    dfg_rows: list[DfgFunctionMetricsRow] = []

    for fg in cfg_graphs:
        s = cfg_struct[fg.function_goid_h128]
        c = cfg_cent[fg.function_goid_h128]
        cfg_rows.append(
            CfgFunctionMetricsRow(
                repo=repo,
                commit=commit,
                function_goid_h128=fg.function_goid_h128,
                block_count=s.block_count,
                loop_count=s.loop_count,
                branching_factor=s.branching_factor,
                max_depth=s.max_depth,
                betweenness=c.betweenness,
                eigenvector=c.eigenvector,
            )
        )

    # similar for DFG

    con.executemany(
        """
        INSERT INTO analytics.cfg_function_metrics (...)
        VALUES (...)
        """,
        [cfg_function_metrics_row_to_tuple(row) for row in cfg_rows],
    )
    # dfg insert...
```

---

## 5. Splitting `analytics/tests_analytics.py` → `analytics/tests/`

### 5.1. Responsibilities today

`tests_analytics.py` does:

* Load coverage data, test catalog.
* Map tests → functions via coverage line ranges.
* Derive “test coverage edges” and insert into `analytics.test_coverage_edges`.
* Some helper logic to flatten coverage and dedupe edges.

Additionally:

* You already have `test_profiles.py` (behavioral coverage) and `test_graph_metrics.py` (bipartite metrics).

We just group them under a subpackage.

### 5.2. Package skeleton

```python
# src/codeintel/analytics/tests/__init__.py
from __future__ import annotations

"""
Tests analytics: coverage edges, behavioral profiles, and graph metrics.
"""

from .coverage_edges import compute_test_coverage_edges
from .profiles import compute_test_profiles
from .graph_metrics import compute_test_graph_metrics

__all__ = [
    "compute_test_coverage_edges",
    "compute_test_profiles",
    "compute_test_graph_metrics",
]
```

Rename/move:

* `tests_analytics.py` → `tests/coverage_edges.py`
* `test_profiles.py` → `tests/profiles.py`
* `test_graph_metrics.py` → `tests/graph_metrics.py`

Compat shim:

```python
# src/codeintel/analytics/tests_analytics.py (compat)
from __future__ import annotations

from .tests import compute_test_coverage_edges

__all__ = ["compute_test_coverage_edges"]
```

### 5.3. Coverage edges module

Most of the old `tests_analytics.py` content moves as-is:

```python
# src/codeintel/analytics/tests/coverage_edges.py
from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.models.rows import (
    TestCoverageEdgeRow,
    test_coverage_edge_to_tuple,
)

def compute_test_coverage_edges(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> None:
    """
    Populate analytics.test_coverage_edges from coverage + test catalog + function map.
    """
    con = gateway.con
    # existing logic from tests_analytics: join coverage + function spans + tests
    rows: list[TestCoverageEdgeRow] = []
    # ...
    con.executemany(
        """
        INSERT INTO analytics.test_coverage_edges (...)
        VALUES (...)
        """,
        [test_coverage_edge_to_tuple(row) for row in rows],
    )
```

Profiles and graph_metrics modules keep their existing logic; you only adjust imports to reflect the new package path.

---

## 6. Migration & agent guidance

### 6.1. Suggested step-by-step implementation order

1. **Functions package**

   * Create `analytics/functions/` + shim `analytics/functions.py`.
   * Move configs → `config.py`, typedness helpers → `typedness.py`, validation → `validation.py`, main compute → `metrics.py`.
   * Adjust internal imports; keep external callers using `codeintel.analytics.functions`.

2. **Subsystems package**

   * Create `analytics/subsystems/` + shim.
   * Move affinity construction & clustering into `affinity.py`.
   * Move `_subsystem_edge_stats` and related logic into `edge_stats.py`.
   * Move risk aggregation into `risk.py`.
   * Create `materialize.py` with `compute_subsystems` as orchestrator and update orchestrators to call it.

3. **CFG/DFG package**

   * Create `analytics/cfg_dfg/` + shim `cfg_dfg_metrics.py`.
   * Move CFG/DFG graph construction and structural metrics into `cfg_core.py` / `dfg_core.py`.
   * Move centrality calls into `graph_metrics.py`.
   * Move DB insert code into `materialize.py`.

4. **Tests package**

   * Create `analytics/tests/` + shim `tests_analytics.py`.
   * Move / rename modules: `tests_analytics` → `coverage_edges`, `test_profiles` → `profiles`, `test_graph_metrics` → `graph_metrics`.
   * Update imports within analytics and orchestration.

Each of these can be a separate PR and will be mostly mechanical moves + import updates, which LLM agents are great at.

### 6.2. Instructions you can embed for the AI programming agent

In your implementation plan / AGENTS doc, you can literally say:

* **Do not change behavior.** Only move code and adjust imports, unless a change is necessary for the split.
* For **functions analytics**:

  * `compute_function_metrics_and_types` lives in `analytics/functions/metrics.py`.
  * Typedness lives in `analytics/functions/typedness.py`.
  * Validation lives in `analytics/functions/validation.py`.
* For **subsystems analytics**:

  * Build the module-affinity graph in `analytics/subsystems/affinity.py`.
  * Edge stats live in `analytics/subsystems/edge_stats.py`.
  * Risk aggregation lives in `analytics/subsystems/risk.py`.
  * `compute_subsystems` orchestrates in `analytics/subsystems/materialize.py`.
* For **CFG/DFG analytics**:

  * CFG/DFG construction & structural metrics in `analytics/cfg_dfg/cfg_core.py` and `dfg_core.py`.
  * Centralities in `analytics/cfg_dfg/graph_metrics.py` using `GraphService`.
  * DB writes in `analytics/cfg_dfg/materialize.py`.
* For **tests analytics**:

  * Coverage edges in `analytics/tests/coverage_edges.py`.
  * Profiles in `analytics/tests/profiles.py`.
  * Test graph metrics in `analytics/tests/graph_metrics.py`.

---

