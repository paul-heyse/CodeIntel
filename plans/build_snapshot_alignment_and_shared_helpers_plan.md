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

## Completed Scope (Jan 2026)
- Implemented strict snapshot scoping:
  - `src/codeintel/build/scopes/snapshot.py`
  - `src/codeintel/build/tabular/scoping.py`
- Implemented dulwich-based snapshot alignment helpers:
  - `src/codeintel/build/scopes/dulwich.py`
- Enforced HEAD alignment in BuildEnv construction:
  - `src/codeintel/build/run_context.py`
- Replaced schema inference snapshot discovery with shared helper:
  - `src/codeintel/build/schemas/inference_service.py`
- Replaced snapshot filters + row collectors with SnapshotScope/collect_scoped_rows:
  - `src/codeintel/build/analytics/semantic_roles/core.py`
  - `src/codeintel/build/analytics/subsystems/cache.py`
  - `src/codeintel/build/analytics/compute/functions/goids.py`
  - `src/codeintel/build/analytics/functions/function_effects.py`
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- Added shared graph builders + edge-weight policy and migrated call/import/symbol graph assembly:
  - `src/codeintel/build/graphs/builders.py`
  - `src/codeintel/build/analytics/graphs/graph_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
  - `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
  - `src/codeintel/build/graphs/engine/views.py`
- Introduced shared GraphContext helpers (GraphContextFactory + GraphContextOverrides) and
  GraphMetricsContext:
  - `src/codeintel/build/analytics/graphs/context_helpers.py`
  - `src/codeintel/build/analytics/graphs/graph_metrics.py`
  - `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
  - `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`
- Added RowBuildContext and updated row builders to use it:
  - `src/codeintel/build/analytics/compute/row_builders/context.py`
  - `src/codeintel/build/analytics/compute/row_builders/graph_metrics.py`
  - `src/codeintel/build/analytics/compute/row_builders/graph_metrics_ext.py`
  - `src/codeintel/build/analytics/compute/row_builders/symbol_metrics.py`
  - `src/codeintel/build/analytics/graphs/graph_metrics.py`
  - `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
  - `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`
  - `src/codeintel/build/analytics/graphs/symbol_orchestrator.py`
- Centralized GraphRuntimeOptions construction from BuildEnv:
  - `src/codeintel/build/graphs/runtime/runtime.py`
  - `src/codeintel/build/graphs/runtime/__init__.py`
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- Added TableTargetContext + single-table spec factory and migrated initial targets:
  - `src/codeintel/build/hamilton/native/patterns/table_target.py`
  - `src/codeintel/build/hamilton/native/patterns/__init__.py`
  - `src/codeintel/build/hamilton/native/analytics/function_contracts.py`
  - `src/codeintel/build/hamilton/native/analytics/function_effects.py`
- Standardized ArrowJoinSpec normalization in cpg2 planes + syntax augment joins:
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/flow.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/goids.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg2/planes/syntax.py`
  - `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- Expanded TableTargetContext adoption for single-table targets:
  - `src/codeintel/build/hamilton/native/analytics/data_models.py` (data_model_usage)
  - `src/codeintel/build/hamilton/native/analytics/function_ast_features.py`
  - `src/codeintel/build/hamilton/native/analytics/function_types.py`
  - `src/codeintel/build/hamilton/native/analytics/function_validation.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py` (graph_stats)
  - `src/codeintel/build/hamilton/native/analytics/graph_validation.py`
  - `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_agreement.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_cache.py`
  - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
  - `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
  - `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
  - `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`
- Migrated remaining multi-table targets to the shared helper and exported table context:
  - `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
  - `src/codeintel/build/hamilton/native/analytics/data_models.py` (data_models)
  - `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
  - `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
  - `src/codeintel/build/hamilton/native/graphs/goids.py`
  - `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
  - `src/codeintel/build/hamilton/native/patterns/__init__.py` (export TableTargetTableContext)

## Remaining Duplication and Drift (Selected)
- Table target scaffolding consolidation complete; remaining TableTargetSpec usage is
  centralized in `src/codeintel/build/hamilton/native/patterns/table_target.py`.

## Proposed Shared Modules and Ownership
- `src/codeintel/build/scopes/snapshot.py` (new)
  - SnapshotScope dataclass, strict matching helpers, Arrow table filtering.
  - Ownership: snapshot scoping is cross-cutting and does not belong solely to analytics or tabular.
  - Status: implemented.
- `src/codeintel/build/scopes/dulwich.py` (new)
  - Dulwich repo discovery + HEAD commit resolution.
  - Ownership: shared build-time repo identity, used by schema inference and build alignment.
  - Status: implemented.
- `src/codeintel/build/tabular/scoping.py` (new)
  - Strict row collection from tabular inputs with required column checks.
  - Ownership: tabular conversion and filtering helpers live under build/tabular.
  - Status: implemented.
- `src/codeintel/build/graphs/builders.py` (new)
  - Call/import/symbol graph builders and edge weight policy helpers.
  - Ownership: graph construction belongs under build/graphs.
  - Status: implemented and adopted.
- `src/codeintel/build/graphs/runtime/runtime.py` (existing)
  - Add `graph_runtime_options_from_env(env: BuildEnv)` to remove repeated logic.
  - Status: implemented and adopted.
- `src/codeintel/build/analytics/graphs/context_helpers.py` (new)
  - GraphContextFactory + GraphContextOverrides for extended metrics + GraphMetricsContext bridge.
  - Ownership: analytics graph computations share context derivation patterns.
  - Status: implemented and adopted.
- `src/codeintel/build/analytics/compute/row_builders/context.py` (new)
  - RowBuildContext (repo, commit, created_at) + row helper conventions.
  - Ownership: row builders are analytics compute concerns.
  - Status: implemented and adopted.
- `src/codeintel/build/tabular/arrow_ops.py` (existing)
  - ArrowJoinOptions + JoinFilterClause + normalize_table_for_join helpers for join execution.
  - Ownership: Arrow join configuration belongs in tabular helpers.
- `src/codeintel/build/tabular/compute_masks.py` (existing)
  - Expression helpers (e.g., is_valid_expr) used by join filter clauses.
- `src/codeintel/build/hamilton/native/patterns/table_target.py` (existing)
  - Add TableTargetContext + factory for `TableContractSpec` + `TableTargetSpec` with defaults.
  - Status: implemented; initial targets migrated.

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
        repo_mask = equal_mask(table["repo"], pa.scalar(self.repo))
        commit_mask = equal_mask(table["commit"], pa.scalar(self.commit))
        return safe_filter(table, and_kleene(repo_mask, commit_mask))

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

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput


def collect_scoped_rows(
    value: InferableTabularInput,
    columns: Sequence[str],
    *,
    scope: SnapshotScope,
    require_scope_columns: bool = True,
) -> list[dict[str, object]]:
    table = tabular_to_arrow_table(value)
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        msg = f"Missing columns for scoped rows: {missing}"
        raise ValueError(msg)
    scoped = scope.filter_arrow_table(table, require_columns=require_scope_columns)
    if columns:
        scoped = scoped.select(list(columns))
    if scoped.num_rows == 0:
        return []
    return list(iter_rows(scoped))
```

### Dulwich alignment with current repo HEAD
Location: `src/codeintel/build/scopes/dulwich.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    from dulwich.repo import Repo as _DulwichRepo
except ImportError:
    _DulwichRepo = None

from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from dulwich.repo import Repo


def _discover_repo(start_path: Path) -> Repo | None:
    if _DulwichRepo is None:
        return None
    try:
        return _DulwichRepo.discover(start_path)
    except (OSError, ValueError):
        return None


def resolve_head_commit(repo_root: Path) -> str | None:
    repo = _discover_repo(repo_root)
    if repo is None:
        return None
    head = repo.head()
    commit = head.decode("ascii", errors="ignore") if isinstance(head, bytes) else str(head)
    commit = commit.strip()
    return commit or None


def snapshot_from_dulwich(start_path: Path | None = None) -> SnapshotRef | None:
    root = start_path or Path.cwd()
    repo = _discover_repo(root)
    if repo is None:
        return None
    repo_root = Path(repo.path).resolve()
    head = resolve_head_commit(repo_root)
    if head is None:
        return None
    repo_name = repo_root.name or "repo"
    return SnapshotRef.from_args(
        repo=repo_name,
        commit=head,
        repo_root=repo_root,
    )


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
- `BuildRunContext.build_env()` calls `ensure_snapshot_matches_head` when building outputs.
- Runtime/CLI entrypoints that construct `SnapshotRef` should use dulwich as the source of truth.
- `_dulwich_snapshot` replaced in `src/codeintel/build/schemas/inference_service.py`.

## Shared Graph Builders

### Edge weight policy
Location: `src/codeintel/build/graphs/builders.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EdgeWeightPolicy:
    default: int = 1

    def next_weight(self, value: object | None) -> int:
        parsed = _coerce_edge_weight(value)
        if parsed is None:
            return self.default
        return parsed + 1
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
    policy: EdgeWeightPolicy | None = None,
) -> None:
    resolved = policy or EdgeWeightPolicy()
    if graph.has_edge(source, target):
        attrs = graph[source][target]
        attrs["weight"] = resolved.next_weight(attrs.get("weight"))
        return
    graph.add_edge(source, target, weight=resolved.default)


def build_call_graph_from_rows(
    rows: list[dict[str, object]],
    nodes: list[dict[str, object]] | None,
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    add_call_graph_edges(graph, rows, policy=policy)
    if nodes is not None:
        add_call_graph_nodes(graph, nodes)
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
    for row in rows:
        def_path = row.get("def_path")
        use_path = row.get("use_path")
        if def_path is None or use_path is None:
            continue
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None or def_module == use_module:
            continue
        add_weighted_edge(graph, use_module, def_module, policy=policy)
    return graph


def build_symbol_function_graph(
    rows: list[dict[str, object]],
    *,
    policy: EdgeWeightPolicy | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    for row in rows:
        def_goid = normalize_decimal_id(row.get("def_goid_h128"))
        use_goid = normalize_decimal_id(row.get("use_goid_h128"))
        if def_goid is None or use_goid is None or def_goid == use_goid:
            continue
        add_weighted_edge(graph, use_goid, def_goid, policy=policy)
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

from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
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

    def build(
        self,
        runtime: GraphRuntimeOptions,
        *,
        repo: str,
        commit: str,
        overrides: GraphContextOverrides | None = None,
    ) -> GraphContext:
        resolved = overrides or GraphContextOverrides()
        return resolve_graph_context(
            GraphContextSpec(
                repo=repo,
                commit=commit,
                use_gpu=runtime.use_gpu if resolved.use_gpu is None else resolved.use_gpu,
                options=resolved.options,
                now=datetime.now(UTC),
                betweenness_cap=self.betweenness_cap,
                eigen_cap=self.eigen_cap,
                pagerank_weight=self.pagerank_weight,
                betweenness_weight=self.betweenness_weight,
                community_detection_limit=(
                    runtime.features.community_detection_limit
                    if resolved.community_detection_limit is None
                    else resolved.community_detection_limit
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class GraphContextOverrides:
    options: GraphMetricsOptions | None = None
    use_gpu: bool | None = None
    community_detection_limit: int | None = None
```

Usage pattern:

```python
factory = GraphContextFactory(
    betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
    eigen_cap=EIGEN_MAX_ITER,
    pagerank_weight="weight",
    betweenness_weight="weight",
)
ctx = factory.build(
    runtime_opts,
    repo=repo,
    commit=commit,
    overrides=GraphContextOverrides(options=options, use_gpu=use_gpu),
)
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
        filters: GraphMetricFilters,
        context_factory: GraphContextFactory,
        overrides: GraphContextOverrides | None = None,
    ) -> "GraphMetricsContext":
        runtime_opts = runtime or GraphRuntimeOptions(snapshot=snapshot)
        graph_ctx = context_factory.build(
            runtime_opts,
            repo=snapshot.repo,
            commit=snapshot.commit,
            overrides=overrides,
        )
        return cls(
            snapshot=snapshot,
            runtime=runtime_opts,
            graph_context=graph_ctx,
            filters=filters,
        )
```

Usage pattern:

```python
context = GraphMetricsContext.from_inputs(
    snapshot=inputs.snapshot,
    runtime=runtime_options,
    filters=active_filters,
    context_factory=context_factory,
    overrides=GraphContextOverrides(options=options, use_gpu=use_gpu),
)
```

File targets:
- `src/codeintel/build/analytics/graphs/context_helpers.py` (new)
- `src/codeintel/build/analytics/graphs/graph_metrics.py`

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
        created_at: datetime | None = None,
    ) -> "RowBuildContext":
        return cls.from_repo_commit(snapshot.repo, snapshot.commit, created_at=created_at)

    @classmethod
    def from_repo_commit(
        cls,
        repo: str,
        commit: str,
        *,
        created_at: datetime | None = None,
    ) -> "RowBuildContext":
        return cls(repo=repo, commit=commit, created_at=created_at or datetime.now(UTC))
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

## Arrow Join Options + Filters

Location: `src/codeintel/build/tabular/arrow_ops.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pyarrow.compute as pc


@dataclass(frozen=True, slots=True)
class ArrowJoinOptions:
    filter_expression: pc.Expression | None = None
    use_threads: bool | None = True
    normalize_inputs: bool = True


JoinFilterSide = Literal["left", "right", "either"]


@dataclass(frozen=True, slots=True)
class JoinFilterClause:
    field: str
    predicate: Callable[[str], pc.Expression]
    side: JoinFilterSide = "either"
```

Usage pattern (basic Arrow join with options):

```python
join_spec = ArrowJoinSpec(on=["repo", "commit"], how="left", validate="m:1")
join_options = build_join_options(left, right)
joined = arrow_join_tables(left, right, spec=join_spec, options=join_options)
```

`build_join_options` uses a row-count heuristic to set `use_threads` when omitted.

Usage pattern (residual join filter with expressions):

```python
join_spec = ArrowJoinSpec(on=["goid_h128"], how="left")
filter_expr = join_filter_expr(
    left=goids,
    right=anchors,
    spec=join_spec,
    clause=JoinFilterClause(
        field="cpg_node_id",
        predicate=is_valid_expr,
        side="right",
    ),
)
join_options = build_join_options(goids, anchors, filter_expression=filter_expr)
joined = arrow_join_tables(goids, anchors, spec=join_spec, options=join_options)
```

Usage pattern (explicit join normalization):

```python
left = normalize_table_for_join(left)
right = normalize_table_for_join(right)
join_options = build_join_options(left, right, normalize_inputs=False)
joined = arrow_join_tables(left, right, spec=join_spec, options=join_options)
```

File targets:
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/compute_masks.py`
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

### Multi-table TableTargetSpec helper (new scope)

Location: `src/codeintel/build/hamilton/native/patterns/table_target.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pyarrow as pa

from codeintel.build.hamilton.native.patterns.savers import DatasetSaveSpec
from codeintel.build.hamilton.native.patterns.table_target import (
    TableTargetSpec,
    TableTargetTableSpec,
)
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec


@dataclass(frozen=True, slots=True)
class MultiTableTargetContext:
    domain: str
    target_name: str
    tables: tuple[TableTargetTableSpec, ...]
    table_materializations_node: str | None = None
    anchor_node_name: str | None = None
    attach_anchor: bool = True

    @staticmethod
    def build_table_spec(
        *,
        table_key: str,
        base_node: str,
        contract: TableContractSpec,
        input_type: type[pa.Table] = pa.Table,
        save_spec: DatasetSaveSpec | None = None,
        node_name: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> TableTargetTableSpec:
        return TableTargetTableSpec(
            table_key=table_key,
            base_node=base_node,
            contract=contract,
            save_spec=save_spec or DatasetSaveSpec(table_key=table_key),
            node_name=node_name or f"{table_key}__table",
            input_type=input_type,
            extra_tags=extra_tags,
        )


def build_multi_table_target_spec(
    *,
    context: MultiTableTargetContext,
) -> TableTargetSpec:
    return TableTargetSpec(
        domain=context.domain,
        target_name=context.target_name,
        tables=context.tables,
        table_materializations_node=(
            context.table_materializations_node
            or f"{context.target_name}__table_materializations"
        ),
        anchor_node_name=(
            f"t__{context.target_name}" if context.attach_anchor else None
        )
        if context.anchor_node_name is None
        else context.anchor_node_name,
    )
```

Usage pattern:

```python
tables = (
    MultiTableTargetContext.build_table_spec(
        table_key=CALL_GRAPH_TABLE_KEY,
        base_node="call_graph__base",
        contract=CALL_GRAPH_CONTRACT,
        node_name="call_graph__table",
    ),
    MultiTableTargetContext.build_table_spec(
        table_key=IMPORT_GRAPH_TABLE_KEY,
        base_node="import_graph__base",
        contract=IMPORT_GRAPH_CONTRACT,
        node_name="import_graph__table",
    ),
)
context = MultiTableTargetContext(
    domain="analytics",
    target_name="config_graphs",
    tables=tables,
)
spec = build_multi_table_target_spec(context=context)
attach_table_target_template(_MODULE, spec=spec)
```

File targets (remaining multi-table migrations):
- `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`
- `src/codeintel/build/hamilton/native/analytics/subsystems.py`
- `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

Implementation checklist:
1. Add `MultiTableTargetContext` and `build_multi_table_target_spec` to
   `src/codeintel/build/hamilton/native/patterns/table_target.py` and export them in
   `src/codeintel/build/hamilton/native/patterns/__init__.py`.
2. Identify every multi-table `TableTargetSpec` construction and replace with the helper:
   - Build `tables` via `MultiTableTargetContext.build_table_spec`.
   - Standardize `table_materializations_node` and `anchor_node_name` defaults.
3. Ensure all `TableTargetTableSpec` fields are preserved (contract, input_type, save_spec,
   node_name, extra_tags).
4. Remove any local helper functions that only existed to construct multi-table specs.
5. Validate that anchor nodes are still attached where required (set `attach_anchor=False`
   when not needed).
6. Confirm table names remain stable (node/table keys unchanged) to avoid materialization drift.

### Multi-table TableTarget refinements (additional scope)

Location: `src/codeintel/build/hamilton/native/patterns/table_target.py`

#### 1) Context-level defaults for save specs and input types
Allow `MultiTableTargetContext` to carry defaults for save specs and input types so
relation-heavy targets can avoid repetitive `RelationTableSaveSpec` and input types.

```python
from __future__ import annotations

from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class MultiTableTargetContext:
    domain: str
    target_name: str
    tables: tuple[TableTargetTableSpec, ...]
    spec: TargetSpecDescriptor | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None
    table_materializations_node: str | None = None
    anchor_node_name: str | None = None
    attach_anchor: bool = True
    save_spec_factory: Callable[[str], DatasetSaveSpec | RelationTableSaveSpec] | None = None
    default_input_type: object | None = None

    @staticmethod
    def build_table_spec(
        *,
        context: TableTargetTableContext | None = None,
        table_key: str | None = None,
        base_node: str | None = None,
        contract: TableContractSpec | None = None,
        save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None,
        node_name: str | None = None,
        input_type: object | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
        save_spec_factory: Callable[[str], DatasetSaveSpec | RelationTableSaveSpec] | None = None,
        default_input_type: object | None = None,
    ) -> TableTargetTableSpec:
        if context is None:
            if table_key is None or base_node is None:
                msg = "table_key and base_node are required when context is not provided"
                raise ValueError(msg)
            context = TableTargetTableContext(
                table_key=table_key,
                base_node=base_node,
                contract=contract,
                save_spec=save_spec,
                node_name=node_name,
                input_type=input_type,
                extra_tags=extra_tags,
            )
        resolved_save_spec = context.save_spec
        if resolved_save_spec is None and save_spec_factory is not None:
            resolved_save_spec = save_spec_factory(context.table_key)
        if resolved_save_spec is None:
            resolved_save_spec = DatasetSaveSpec(table_key=context.table_key)
        resolved_input_type = context.input_type or default_input_type
        return TableTargetTableSpec(
            table_key=context.table_key,
            base_node=context.base_node,
            contract=context.contract,
            save_spec=resolved_save_spec,
            node_name=context.node_name,
            input_type=resolved_input_type,
            extra_tags=context.extra_tags,
        )
```

Targets:
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

#### 2) Multi-table builder from contexts
Provide a convenience factory to accept `TableTargetTableContext` entries directly.

```python
def build_multi_table_target_spec_from_contexts(
    *,
    domain: str,
    target_name: str,
    table_contexts: Sequence[TableTargetTableContext],
    spec: TargetSpecDescriptor | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
    table_materializations_node: str | None = None,
    anchor_node_name: str | None = None,
    attach_anchor: bool = True,
    save_spec_factory: Callable[[str], DatasetSaveSpec | RelationTableSaveSpec] | None = None,
    default_input_type: object | None = None,
) -> TableTargetSpec:
    tables = tuple(
        MultiTableTargetContext.build_table_spec(
            context=ctx,
            save_spec_factory=save_spec_factory,
            default_input_type=default_input_type,
        )
        for ctx in table_contexts
    )
    return build_multi_table_target_spec(
        context=MultiTableTargetContext(
            domain=domain,
            target_name=target_name,
            tables=tables,
            spec=spec,
            extra_tags=extra_tags,
            table_materializations_node=table_materializations_node,
            anchor_node_name=anchor_node_name,
            attach_anchor=attach_anchor,
            save_spec_factory=save_spec_factory,
            default_input_type=default_input_type,
        )
    )
```

Targets:
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

#### 3) Contract-derived table contexts
Add a helper to derive table contexts from `TableContractSpec` so `input_name` remains
the base-node source of truth.

```python
@dataclass(frozen=True, slots=True)
class TableTargetTableContext:
    table_key: str
    base_node: str
    contract: TableContractSpec | None = None
    save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None
    node_name: str | None = None
    input_type: object | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None

    @staticmethod
    def from_contract(
        contract: TableContractSpec,
        *,
        base_node: str | None = None,
        node_name: str | None = None,
        save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None,
        input_type: object | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> "TableTargetTableContext":
        return TableTargetTableContext(
            table_key=contract.table_key,
            base_node=base_node or contract.input_name,
            contract=contract,
            save_spec=save_spec,
            node_name=node_name,
            input_type=input_type,
            extra_tags=extra_tags,
        )
```

Targets:
- `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`

#### 4) Save-spec helper builders
Provide explicit helpers for common save-spec patterns (dataset vs relation tables).

```python
@staticmethod
def build_dataset_table_spec(
    *,
    table_key: str,
    base_node: str,
    contract: TableContractSpec | None = None,
    partition_columns: tuple[str, ...] = (),
    collect_group: str | None = None,
    validation_profile: ValidationProfile | None = None,
    output_name: str | None = None,
    node_name: str | None = None,
    input_type: object | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> TableTargetTableSpec:
    return MultiTableTargetContext.build_table_spec(
        table_key=table_key,
        base_node=base_node,
        contract=contract,
        save_spec=DatasetSaveSpec(
            table_key=table_key,
            partition_columns=partition_columns,
            collect_group=collect_group,
            validation_profile=validation_profile,
            output_name=output_name,
        ),
        node_name=node_name,
        input_type=input_type,
        extra_tags=extra_tags,
    )


@staticmethod
def build_relation_table_spec(
    *,
    table_key: str,
    base_node: str,
    contract: TableContractSpec | None = None,
    validation_profile: ValidationProfile | None = None,
    output_name: str | None = None,
    node_name: str | None = None,
    input_type: object | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> TableTargetTableSpec:
    return MultiTableTargetContext.build_table_spec(
        table_key=table_key,
        base_node=base_node,
        contract=contract,
        save_spec=RelationTableSaveSpec(
            table_key=table_key,
            validation_profile=validation_profile,
            output_name=output_name,
        ),
        node_name=node_name,
        input_type=input_type,
        extra_tags=extra_tags,
    )
```

Targets:
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`

#### 5) Optional naming policy helper
Centralize non-standard table node naming patterns to avoid drifting strings.

```python
def table_node_name_for_target(*, target_name: str, suffix: str) -> str:
    return f"{target_name}__{suffix}_table"
```

Targets:
- `src/codeintel/build/hamilton/native/graphs/graph_targets.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`

Implementation checklist (refinements):
1. Extend `MultiTableTargetContext` with `save_spec_factory` and `default_input_type`,
   and thread defaults through `build_table_spec`.
2. Add `build_multi_table_target_spec_from_contexts` convenience builder.
3. Add `TableTargetTableContext.from_contract` for contract-driven base nodes.
4. Add `build_dataset_table_spec` / `build_relation_table_spec` helpers and migrate
   repetitive Dataset/Relation save-spec blocks.
5. Optional: add a naming helper for consistent node names where targets diverge from
   `__base` conventions.

## Implementation Steps
1. Add `scopes` package and move dulwich snapshot logic out of
   `src/codeintel/build/schemas/inference_service.py`. (done)
2. Implement strict SnapshotScope filtering and `collect_scoped_rows`. (done)
3. Add GraphContextFactory + GraphContextOverrides + GraphMetricsContext helpers for
   analytics graphs. (done)
4. Update snapshot filtering call sites to use SnapshotScope strict matching: (done)
   - `src/codeintel/build/analytics/semantic_roles/core.py`
   - `src/codeintel/build/analytics/subsystems/cache.py`
   - `src/codeintel/build/analytics/compute/functions/goids.py`
   - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
   - `src/codeintel/build/hamilton/native/analytics/subsystem_metrics.py`
   - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
   - `src/codeintel/build/analytics/functions/function_effects.py`
5. Add graph builders and switch graph assembly call sites to use them. (done)
6. Add `graph_runtime_options_from_env` in `graphs/runtime/runtime.py` and
   update analytics modules to use it. (done)
7. Add RowBuildContext and update row builder call sites to use it
   for consistent repo/commit/created_at injection. (done)
8. Adopt ArrowJoinOptions + join_filter_expr + normalize_table_for_join across
   Arrow join call sites in build pipelines. (done for cpg2 planes + syntax_augment)
9. Add TableTargetContext + TableTargetSpec factory and update at least two
   analytics targets as a template (function_contracts, function_effects). (done)
10. Decide whether to add a multi-table TableTarget builder (parallel to TableTargetContext).
    Adopted; migrated multi-table targets across analytics, ingestion, and graphs. (done)

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
  - created_at is deterministic when `created_at` supplied
- ArrowJoinOptions + JoinFilterClause:
  - join_filter_expr resolves fields correctly with suffix/coalesce behavior
  - Arrow joins honor filter_expression and fall back when unsupported
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
