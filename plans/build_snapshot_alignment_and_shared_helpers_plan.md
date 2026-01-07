# Build Snapshot Alignment and Shared Helpers Plan

## Goals
- Enforce strict repo/commit matching for snapshot-scoped inputs and outputs.
- Align BuildEnv snapshot with the current repo HEAD commit (dulwich) before emitting outputs.
- Consolidate duplicated graph builders, snapshot filtering, row collection, and target scaffolding.
- Introduce shared context objects that carry repo/commit/created_at consistently.

## Non-Goals
- Support multi-snapshot or historical outputs in a single run.
- Modify schema definitions, table keys, or materialization formats.
- Introduce new external dependencies beyond the existing dulwich usage.

## Current Duplication and Drift (Selected)
- Snapshot filtering with inconsistent strictness:
  - `src/codeintel/build/analytics/semantic_roles/core.py`
  - `src/codeintel/build/analytics/subsystems/cache.py`
  - `src/codeintel/build/analytics/compute/functions/goids.py`
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- Row collection logic duplicated across analytics modules:
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- Call/import graph assembly and edge weight logic duplicated:
  - `src/codeintel/build/analytics/graphs/graph_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
  - `src/codeintel/build/analytics/functions/function_effects.py`
  - `src/codeintel/build/graphs/engine/views.py`
- Symbol graph building duplicated:
  - `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- GraphRuntimeOptions from BuildEnv repeated:
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
- GraphContext resolution for extended metrics repeated:
  - `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
  - `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`
- Join spec conversion glue duplicated across Arrow + Polars paths:
  - `src/codeintel/build/tabular/arrow_ops.py`
  - `src/codeintel/build/tabular/frames.py`
- Table target scaffolding repeated:
  - `src/codeintel/build/hamilton/native/analytics/function_contracts.py`
  - `src/codeintel/build/hamilton/native/analytics/function_effects.py`

## Proposed Shared Modules and Ownership
- `src/codeintel/build/scopes/snapshot.py` (new)
  - SnapshotScope dataclass, strict matching helpers, Arrow table filtering.
  - Ownership: snapshot scoping is cross-cutting and does not belong solely to analytics or tabular.
- `src/codeintel/build/scopes/dulwich.py` (new)
  - Dulwich repo discovery + HEAD commit resolution.
  - Ownership: shared build-time repo identity, used by schema inference and build alignment.
- `src/codeintel/build/tabular/scoping.py` (new)
  - Strict row collection from tabular inputs with required column checks.
  - Ownership: tabular conversion and filtering helpers live under build/tabular.
- `src/codeintel/build/graphs/builders.py` (new)
  - Call/import/symbol graph builders and edge weight policy helpers.
  - Ownership: graph construction belongs under build/graphs.
- `src/codeintel/build/graphs/runtime/runtime.py` (existing)
  - Add `graph_runtime_options_from_env(env: BuildEnv)` to remove repeated logic.
- `src/codeintel/build/analytics/graphs/context_helpers.py` (new)
  - GraphContextFactory for extended metrics + GraphMetricsContext bridge.
  - Ownership: analytics graph computations share context derivation patterns.
- `src/codeintel/build/analytics/compute/row_builders/context.py` (new)
  - RowBuildContext (repo, commit, created_at) + row helper conventions.
  - Ownership: row builders are analytics compute concerns.
- `src/codeintel/build/tabular/join_specs.py` (new)
  - Unified JoinSpec used by Arrow joins and Polars joins (no conversion glue).
  - Ownership: join configuration belongs in tabular helpers.
- `src/codeintel/build/hamilton/native/patterns/table_target.py` (existing)
  - Add TableTargetContext + factory for `TableContractSpec` + `TableTargetSpec` with defaults.

## Strict Snapshot Alignment

### SnapshotScope (strict match)
Location: `src/codeintel/build/scopes/snapshot.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, equal_mask
from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True, slots=True)
class SnapshotScope:
    repo: str
    commit: str

    @classmethod
    def from_snapshot(cls, snapshot: SnapshotRef) -> "SnapshotScope":
        return cls(repo=snapshot.repo, commit=snapshot.commit)

    def filter_arrow_table(
        self,
        table: pa.Table,
        *,
        require_columns: bool = True,
    ) -> pa.Table:
        missing = [name for name in ("repo", "commit") if name not in table.column_names]
        if missing:
            if require_columns:
                msg = f"Missing snapshot columns: {missing}"
                raise ValueError(msg)
            return table
        mask = equal_mask(table["repo"], pa.scalar(self.repo))
        mask = and_kleene(mask, equal_mask(table["commit"], pa.scalar(self.commit)))
        return safe_filter(table, mask)

    def filter_rows(
        self,
        rows: list[dict[str, object]],
        *,
        require_keys: bool = True,
    ) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []
        for row in rows:
            if require_keys and ("repo" not in row or "commit" not in row):
                msg = "Missing snapshot keys in row"
                raise ValueError(msg)
            if "repo" in row and row.get("repo") != self.repo:
                continue
            if "commit" in row and row.get("commit") != self.commit:
                continue
            filtered.append(row)
        return filtered
```

### Strict row collection for tabular inputs
Location: `src/codeintel/build/tabular/scoping.py`

```python
from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.table_ops import select_table_columns
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import equal_mask


def collect_scoped_rows(
    value: object,
    columns: Sequence[str],
    *,
    scope: SnapshotScope,
    require_columns: bool = True,
) -> list[dict[str, object]]:
    table = tabular_to_arrow_table(value)
    if columns:
        table = select_table_columns(table, columns)
        if require_columns and len(columns) != len(table.column_names):
            msg = f"Missing columns for scoped rows: {columns}"
            raise ValueError(msg)
    scoped = scope.filter_arrow_table(table, require_columns=require_columns)
    if scoped.num_rows == 0:
        return []
    return list(iter_rows(scoped))
```

### Dulwich alignment with current repo HEAD
Location: `src/codeintel/build/scopes/dulwich.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import cast

from dulwich.repo import Repo

from codeintel.config.primitives import SnapshotRef


def resolve_head_commit(repo_root: Path) -> str | None:
    repo = Repo.discover(repo_root)
    head = repo.head()
    commit = head.decode("ascii", errors="ignore") if isinstance(head, bytes) else str(head)
    commit = commit.strip()
    return commit or None


def ensure_snapshot_matches_head(snapshot: SnapshotRef) -> SnapshotRef:
    head = resolve_head_commit(snapshot.repo_root)
    if head is None:
        msg = "Unable to resolve HEAD commit from repo_root"
        raise RuntimeError(msg)
    if head != snapshot.commit:
        msg = (
            "Snapshot commit does not match repo HEAD: "
            f"snapshot={snapshot.commit} head={head}"
        )
        raise ValueError(msg)
    return snapshot
```

### Enforcement points
- `BuildRunContext.build_env()` should call `ensure_snapshot_matches_head` when building outputs.
- Runtime/CLI entrypoints that construct `SnapshotRef` should use dulwich as the source of truth.
- Replace `_dulwich_snapshot` in `src/codeintel/build/schemas/inference_service.py` with the new helper.

## Shared Graph Builders

### Edge weight policy
Location: `src/codeintel/build/graphs/builders.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EdgeWeightPolicy:
    default: int = 1

    def next_weight(self, value: object | None) -> int:
        if value is None:
            return self.default
        if isinstance(value, bool):
            return int(value) + 1
        if isinstance(value, int):
            return value + 1
        if isinstance(value, float):
            return int(value) + 1
        if isinstance(value, str):
            try:
                return int(value) + 1
            except ValueError:
                return self.default
        return self.default
```

### Call/import graph builders
Location: `src/codeintel/build/graphs/builders.py`

```python
import networkx as nx

from codeintel.core.data_models.ids import normalize_decimal_id


def add_weighted_edge(
    graph: nx.Graph,
    source: object,
    target: object,
    *,
    policy: EdgeWeightPolicy,
) -> None:
    if graph.has_edge(source, target):
        attrs = graph[source][target]
        attrs["weight"] = policy.next_weight(attrs.get("weight"))
        return
    graph.add_edge(source, target, weight=policy.default)


def build_call_graph_from_rows(
    rows: list[dict[str, object]],
    nodes: list[dict[str, object]] | None,
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    edge_policy = policy or EdgeWeightPolicy()
    for row in rows:
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        add_weighted_edge(graph, caller, callee, policy=edge_policy)
    if nodes is None:
        return graph
    for row in nodes:
        node = normalize_decimal_id(row.get("goid_h128"))
        if node is None or node in graph:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(node, **attrs)
    return graph
```

### Symbol graph builders
Location: `src/codeintel/build/graphs/builders.py`

```python
import networkx as nx

from codeintel.core.data_models.ids import normalize_decimal_id


def build_symbol_module_graph(
    rows: list[dict[str, object]],
    module_by_path: dict[str, str],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    edge_policy = policy or EdgeWeightPolicy()
    for row in rows:
        def_path = row.get("def_path")
        use_path = row.get("use_path")
        if def_path is None or use_path is None:
            continue
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None or def_module == use_module:
            continue
        add_weighted_edge(graph, use_module, def_module, policy=edge_policy)
    return graph


def build_symbol_function_graph(
    rows: list[dict[str, object]],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    edge_policy = policy or EdgeWeightPolicy()
    for row in rows:
        def_goid = normalize_decimal_id(row.get("def_goid_h128"))
        use_goid = normalize_decimal_id(row.get("use_goid_h128"))
        if def_goid is None or use_goid is None or def_goid == use_goid:
            continue
        add_weighted_edge(graph, use_goid, def_goid, policy=edge_policy)
    return graph
```

### Integration targets
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

## GraphContext Factory for Extended Metrics

Location: `src/codeintel/build/analytics/graphs/context_helpers.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import (
    GraphContext,
    GraphContextSpec,
    resolve_graph_context,
)


@dataclass(frozen=True, slots=True)
class GraphContextFactory:
    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"

    def build(self, runtime: GraphRuntimeOptions, *, repo: str, commit: str) -> GraphContext:
        return resolve_graph_context(
            GraphContextSpec(
                repo=repo,
                commit=commit,
                use_gpu=runtime.use_gpu,
                now=datetime.now(UTC),
                betweenness_cap=self.betweenness_cap,
                eigen_cap=self.eigen_cap,
                pagerank_weight=self.pagerank_weight,
                betweenness_weight=self.betweenness_weight,
                community_detection_limit=runtime.features.community_detection_limit,
            )
        )
```

Usage pattern:

```python
factory = GraphContextFactory(
    betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
    eigen_cap=EIGEN_MAX_ITER,
    pagerank_weight="weight",
    betweenness_weight="weight",
)
ctx = factory.build(runtime_opts, repo=repo, commit=commit)
```

File targets:
- `src/codeintel/build/analytics/graphs/context_helpers.py` (new)
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`

## GraphMetricsContext

Location: `src/codeintel/build/analytics/graphs/context_helpers.py`

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContext
from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True, slots=True)
class GraphMetricsContext:
    snapshot: SnapshotRef
    runtime: GraphRuntimeOptions
    graph_context: GraphContext
    filters: GraphMetricFilters

    @classmethod
    def from_inputs(
        cls,
        *,
        snapshot: SnapshotRef,
        runtime: GraphRuntimeOptions | None,
        filters: GraphMetricFilters | None,
        context_factory: GraphContextFactory,
    ) -> "GraphMetricsContext":
        runtime_opts = runtime or GraphRuntimeOptions(snapshot=snapshot)
        graph_ctx = context_factory.build(
            runtime_opts,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        return cls(
            snapshot=snapshot,
            runtime=runtime_opts,
            graph_context=graph_ctx,
            filters=filters or GraphMetricFilters(),
        )
```

Usage pattern:

```python
context = GraphMetricsContext.from_inputs(
    snapshot=inputs.snapshot,
    runtime=runtime_options,
    filters=active_filters,
    context_factory=context_factory,
)
```

File targets:
- `src/codeintel/build/analytics/graphs/context_helpers.py` (new)
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`

## Shared RowBuildContext

Location: `src/codeintel/build/analytics/compute/row_builders/context.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True, slots=True)
class RowBuildContext:
    repo: str
    commit: str
    created_at: datetime

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotRef,
        *,
        now: datetime | None = None,
    ) -> "RowBuildContext":
        return cls(
            repo=snapshot.repo,
            commit=snapshot.commit,
            created_at=now or datetime.now(UTC),
        )
```

Usage pattern in row builders:

```python
ctx = RowBuildContext.from_snapshot(snapshot)
rows = [
    {
        "repo": ctx.repo,
        "commit": ctx.commit,
        "created_at": ctx.created_at,
        "module": module,
    }
    for module in modules
]
```

## Join Spec Unification

Location: `src/codeintel/build/tabular/join_specs.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

JoinStrategy = Literal["inner", "left", "right", "full", "semi", "anti", "cross", "outer"]
JoinValidation = Literal["m:m", "m:1", "1:m", "1:1"]


@dataclass(frozen=True, slots=True)
class JoinSpec:
    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: JoinStrategy = "left"
    validate: JoinValidation | None = None
    suffix: str = ""
    coalesce_keys: bool = True
    left_suffix: str | None = None
    right_suffix: str | None = None
```

Usage pattern (Arrow join):

```python
spec = JoinSpec(on=["repo", "commit"], how="left", validate="m:1")
joined = arrow_join_tables(left, right, spec=spec)
```

Usage pattern (Polars join):

```python
spec = JoinSpec(left_on=["left_id"], right_on=["right_id"], how="inner")
joined = join_validated(left_lazy, right_lazy, spec=spec)
```

File targets:
- `src/codeintel/build/tabular/join_specs.py` (new)
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/frames.py`
- `src/codeintel/build/tabular/__init__.py` (re-export JoinSpec)
- `src/codeintel/build/analytics/subsystems/cache.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`

## Table Target Spec Factory

Location: `src/codeintel/build/hamilton/native/patterns/table_target.py`

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.hamilton.native.patterns.table_target import (
    TableTargetSpec,
    TableTargetTableSpec,
)
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.hamilton.native.patterns.savers import DatasetSaveSpec


@dataclass(frozen=True, slots=True)
class TableTargetContext:
    domain: str
    target_name: str
    table_key: str
    base_node: str
    contract: TableContractSpec
    input_type: object


def build_single_table_target_spec(
    *,
    context: TableTargetContext,
) -> TableTargetSpec:
    return TableTargetSpec(
        domain=context.domain,
        target_name=context.target_name,
        tables=(
            TableTargetTableSpec(
                table_key=context.table_key,
                base_node=context.base_node,
                contract=context.contract,
                save_spec=DatasetSaveSpec(table_key=context.table_key),
                node_name=f"{context.target_name}__table",
                input_type=context.input_type,
            ),
        ),
        table_materializations_node=f"{context.target_name}__table_materializations",
        anchor_node_name=f"t__{context.target_name}",
    )
```

Usage pattern:

```python
context = TableTargetContext(
    domain="analytics",
    target_name="function_contracts",
    table_key=FUNCTION_CONTRACTS_TABLE_KEY,
    base_node="function_contracts__base",
    contract=FUNCTION_CONTRACTS_CONTRACT,
    input_type=pa.Table,
)
spec = build_single_table_target_spec(context=context)
attach_table_target_template(_MODULE, spec=spec)
```

File targets:
- `src/codeintel/build/hamilton/native/patterns/table_target.py`
- `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`
- `src/codeintel/build/hamilton/native/analytics/subsystems.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/data_models.py`
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
- `src/codeintel/build/hamilton/native/analytics/function_validation.py`
- `src/codeintel/build/hamilton/native/analytics/graph_validation.py`
- `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
- `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/hamilton/native/analytics/function_effects.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_agreement.py`
- `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`
- `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
- `src/codeintel/build/hamilton/native/analytics/function_types.py`
- `src/codeintel/build/hamilton/native/analytics/function_contracts.py`
- `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`
- `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

## Implementation Steps
1. Add `scopes` package and move dulwich snapshot logic out of
   `src/codeintel/build/schemas/inference_service.py`.
2. Implement strict SnapshotScope filtering and `collect_scoped_rows`.
3. Add GraphContextFactory + GraphMetricsContext helpers for analytics graphs.
4. Update snapshot filtering call sites to use SnapshotScope strict matching:
   - `src/codeintel/build/analytics/semantic_roles/core.py`
   - `src/codeintel/build/analytics/subsystems/cache.py`
   - `src/codeintel/build/analytics/compute/functions/goids.py`
   - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
   - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
   - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
   - `src/codeintel/build/analytics/functions/function_effects.py`
5. Add graph builders and switch graph assembly call sites to use them.
6. Add `graph_runtime_options_from_env` in `graphs/runtime/runtime.py` and
   update analytics modules to use it.
7. Add RowBuildContext and update row builder call sites to use it
   for consistent repo/commit/created_at injection.
8. Unify join specs in `src/codeintel/build/tabular/join_specs.py` and update
   `arrow_ops.py` + `frames.py`.
9. Add TableTargetContext + TableTargetSpec factory and update at least two
   analytics targets as a template (function_contracts, function_effects).
10. Remove or deprecate duplicated helpers where appropriate.

## Testing Plan
- SnapshotScope filtering:
  - strict match includes only exact repo/commit values
  - missing repo/commit columns raise when `require_columns=True`
- Dulwich alignment:
  - HEAD commit resolved from a temp repo
  - mismatch raises with a clear error
- Graph builders:
  - repeated edges increment weight deterministically
  - node attribute merging preserved
- GraphContextFactory:
  - caps/weights applied consistently across function/module contexts
  - runtime.use_gpu propagates to GraphContextSpec
- RowBuildContext:
  - created_at is deterministic when `now` supplied
- JoinSpec:
  - Arrow joins and Polars joins accept the same JoinSpec instance
- Regression checks:
  - graph_metrics and subsystem_metrics end-to-end still materialize rows

## Risks and Mitigations
- Strict filtering may drop rows when inputs are missing repo/commit.
  - Mitigation: fail fast with explicit error messages to fix upstream.
- Dulwich HEAD mismatch could block runs if snapshots are stale.
  - Mitigation: update snapshot resolution to use dulwich HEAD when building
    ExecutionContext; allow explicit override only for tests.

## Suggested Rollout
1. Introduce helpers without call site changes.
2. Migrate analytics and Hamilton native call sites.
3. Remove duplicated local helpers once migrations are complete.
