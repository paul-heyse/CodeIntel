# Hamilton Modularization Implementation Plan

This document lays out a detailed implementation plan for modularizing the
Hamilton integration under `src/codeintel/build/hamilton`. Each scope item
includes a design sketch and a step-by-step plan with code patterns to follow.

## 1. Execution profile factory (graph adapters + dynamic executors)

Goal:
- Centralize execution configuration (graph adapters + dynamic executors) to
  avoid drift between `executor.py`, `adapters/parallel.py`, and
  `execution_options.py`.

Status: completed.
Implemented in:
- `src/codeintel/build/hamilton/execution_profiles.py`
- `src/codeintel/build/hamilton/executor.py`

Design sketch:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.hamilton.env import BuildEnv

if TYPE_CHECKING:
    from hamilton.lifecycle import ResultBuilder


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    parallel_backend: str
    max_workers: int | None
    thread_name_prefix: str
    dynamic_enabled: bool
    dynamic_local_executor: str | None
    dynamic_remote_executor: str | None
    dynamic_remote_max_tasks: int | None


@dataclass(frozen=True, slots=True)
class DynamicExecutionConfig:
    enabled: bool
    local_executor: object | None
    remote_executor: object | None


def build_execution_profile(
    *,
    env: BuildEnv,
    options: BuildExecutionOptions,
    max_workers: int | None,
    thread_name_prefix: str,
) -> ExecutionProfile:
    remote_max_tasks = env.execution_settings.dynamic_remote_max_tasks
    if remote_max_tasks is None:
        remote_max_tasks = options.max_workers
    if remote_max_tasks is None:
        remote_max_tasks = env.execution_settings.max_workers
    return ExecutionProfile(
        parallel_backend=options.parallel_backend,
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
        dynamic_enabled=bool(env.execution_settings.dynamic_execution),
        dynamic_local_executor=env.execution_settings.dynamic_local_executor,
        dynamic_remote_executor=env.execution_settings.dynamic_remote_executor,
        dynamic_remote_max_tasks=remote_max_tasks,
    )


def build_parallel_adapter(
    profile: ExecutionProfile,
    *,
    result_builder: ResultBuilder | None,
    dynamic_enabled: bool,
) -> lifecycle_base.LifecycleAdapter | None:
    if dynamic_enabled:
        return None
    return create_parallel_adapter(
        profile.parallel_backend,
        max_workers=profile.max_workers,
        thread_name_prefix=profile.thread_name_prefix,
        result_builder=result_builder,
    )


def apply_dynamic_execution_config(
    *,
    config: dict[str, object],
    profile: ExecutionProfile,
) -> DynamicExecutionConfig:
    dynamic_config = resolve_dynamic_execution_config(profile)
    config["ci.dynamic_execution"] = dynamic_config.enabled
    config["ci_dynamic_module_records"] = dynamic_config.enabled
    if dynamic_config.enabled:
        if dynamic_config.local_executor is not None:
            config["ci.dynamic_execution.local_executor"] = dynamic_config.local_executor
        if dynamic_config.remote_executor is not None:
            config["ci.dynamic_execution.remote_executor"] = dynamic_config.remote_executor
    return dynamic_config
```

Implementation steps (completed):
1. Added `src/codeintel/build/hamilton/execution_profiles.py` with
   `ExecutionProfile`, `DynamicExecutionConfig`, `build_execution_profile`,
   `build_parallel_adapter`, and `apply_dynamic_execution_config`.
2. Moved dynamic executor resolution from
   `src/codeintel/build/hamilton/executor.py` into the new module.
3. Replaced direct calls to `create_parallel_adapter` in
   `src/codeintel/build/hamilton/executor.py` with `build_parallel_adapter`.
4. Routed dynamic executor wiring through `apply_dynamic_execution_config`.

Touchpoints:
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/adapters/parallel.py`
- `src/codeintel/build/hamilton/execution_options.py`
- New: `src/codeintel/build/hamilton/execution_profiles.py`

## 2. Input bundle collector factory

Goal:
- Replace repeated `*_inputs` and `*_frames` collectors with a shared factory
  that preserves type signatures and tags.

Design sketch:
```python
from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tag_spec import TagSpec
from codeintel.build.hamilton.tagging import tag_helper

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class BundleSpec:
    name: str
    fields: Mapping[str, object]
    return_type: object
    tag_spec: TagSpec


def make_bundle(spec: BundleSpec) -> Callable[..., object]:
    def bundle(**kwargs: object) -> object:
        data = {key: kwargs[key] for key in spec.fields}
        return cast("object", spec.return_type(**data))

    signature = inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            )
            for name, annotation in spec.fields.items()
        ],
        return_annotation=spec.return_type,
    )
    bundle = set_signature(bundle, signature)
    bundle.__name__ = spec.name
    return tag_helper(domain=spec.tag_spec.domain, target=spec.tag_spec.target)(bundle)
```

Implementation steps:
1. Add `src/codeintel/build/hamilton/nodes/bundles.py` with `BundleSpec` and
   `make_bundle`.
2. Replace collectors in
   `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py` with `make_bundle`.
3. Replace collectors in
   `src/codeintel/build/hamilton/native/analytics/entrypoints.py` with
   `make_bundle`.
4. Use `TagSpec.for_helper` to tag bundle nodes for catalog visibility.

Touchpoints:
- `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py`
- `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
- New: `src/codeintel/build/hamilton/nodes/bundles.py`

## 3. Parameterized subdag for graph pipelines

Goal:
- Collapse repeated load/compute/save patterns for graph pipelines using
  `@parameterized_subdag` and shared specs.

Design sketch:
```python
from __future__ import annotations

from hamilton.function_modifiers import parameterized_subdag, source, value

from codeintel.build.hamilton.native.graphs import call_graph, cfg_dfg


@parameterized_subdag(
    load_from=[call_graph, cfg_dfg],
    call_graph={"inputs": {"graph_backend": value("compute")}},
    cfg_dfg={"inputs": {"graph_backend": value("existing")}},
)
def graph_pipeline(graph_output: object) -> object:
    return graph_output
```

Implementation steps:
1. Add `src/codeintel/build/hamilton/native/graphs/pipelines.py` with shared
   pipeline specs and parameterized subdags.
2. Replace per-graph wiring in
   `src/codeintel/build/hamilton/native/graphs/call_graph.py`,
   `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`, and
   `src/codeintel/build/hamilton/native/graphs/variants.py` with imports from
   the new pipeline module.
3. Use `@resolve_from_config` at the subdag boundary for graph backend choice.

Touchpoints:
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/variants.py`
- New: `src/codeintel/build/hamilton/native/graphs/pipelines.py`

## 4. Central cache behavior policy

Goal:
- Centralize cache behaviors and add a safe "cache salt" mechanism to prevent
  stale hits when helper functions change.

Status: completed.
Implemented in:
- `src/codeintel/build/hamilton/cache_policy.py`
- `src/codeintel/build/hamilton/cache_adapter.py`
- `src/codeintel/build/hamilton/cache_key_resolver.py`
- `src/codeintel/runtime/compose.py`
- `src/codeintel/build/hamilton/native/planning/plan_nodes.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/tag_spec.py`
- `src/codeintel/build/hamilton/tagging.py`

Design sketch:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.caching.adapter import CachingBehavior

if TYPE_CHECKING:
    from hamilton.node import Node


@dataclass(frozen=True, slots=True)
class CachePolicy:
    default: CachingBehavior
    by_node_type: dict[str, CachingBehavior]


def resolve_behavior(node: Node, policy: CachePolicy) -> CachingBehavior:
    tags = node.tags if isinstance(node.tags, dict) else {}
    node_type = tags.get("node_type")
    return policy.by_node_type.get(str(node_type), policy.default)


def cache_salt(runtime_fingerprint: str) -> str:
    return f"codeintel:{runtime_fingerprint}"
```

Implementation steps (completed):
1. Added `src/codeintel/build/hamilton/cache_policy.py` with
   `CachePolicy`, `resolve_behavior`, `is_salt_sensitive`, and `cache_salt`.
2. Updated `ManifestBackedCacheAdapter.resolve_behaviors` in
   `src/codeintel/build/hamilton/cache_adapter.py` to call `resolve_behavior`.
3. Threaded cache salt handling through
   `src/codeintel/build/hamilton/cache_key_resolver.py` and
   `src/codeintel/runtime/compose.py`.
4. Replaced ad-hoc `@cache(behavior="ignore")` usage in
   `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py` and
   `src/codeintel/build/hamilton/native/planning/plan_nodes.py` with
   centralized cache behavior tags.

Touchpoints:
- `src/codeintel/build/hamilton/cache_adapter.py`
- `src/codeintel/build/hamilton/cache_key_resolver.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/planning/plan_nodes.py`
- New: `src/codeintel/build/hamilton/cache_policy.py`

## 5. Materializer base/mixin for consistent context + errors

Goal:
- Extract shared materializer logic for context resolution, timing, and
  error mapping across savers.

Design sketch:
```python
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv


@dataclass(frozen=True, slots=True)
class MaterializerMixin:
    env: BuildEnv
    catalog: DagCatalog
    target_name: str

    def resolve_context(self) -> tuple[object, float] | MaterializationContextError:
        start = perf_counter()
        ctx = resolve_materialization_context(
            env=self.env,
            catalog=self.catalog,
            target_name=self.target_name,
        )
        if isinstance(ctx, MaterializationContextError):
            return ctx
        return ctx, duration_ms(start)
```

Implementation steps:
1. Add a `MaterializerMixin` (or helper functions) to
   `src/codeintel/build/hamilton/materializers/base.py`.
2. Update `src/codeintel/build/hamilton/materializers/artifact_saver.py` to use
   the mixin for context and error mapping.
3. Update `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
   to use the same mixin and unify metrics metadata.

Touchpoints:
- `src/codeintel/build/hamilton/materializers/base.py`
- `src/codeintel/build/hamilton/materializers/artifact_saver.py`
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

## 6. Expand with_columns backends + column ops registry

Goal:
- Make column subDAGs first-class by supporting more backends and a clear
  feature ops registry.

Design sketch:
```python
from __future__ import annotations

from types import ModuleType

_FEATURE_MODULES: dict[str, ModuleType] = {}


def register_feature_module(table_key: str, module: ModuleType) -> None:
    _FEATURE_MODULES[table_key] = module


def feature_module(table_key: str) -> ModuleType | None:
    return _FEATURE_MODULES.get(table_key)
```

Implementation steps:
1. Add `src/codeintel/build/hamilton/column_ops/registry.py` with registry
   helpers and validation for allowed op names.
2. Update `src/codeintel/build/hamilton/column_ops/__init__.py` to re-export
   registry helpers.
3. Extend `select_with_columns` in
   `src/codeintel/build/hamilton/transforms/with_columns_backend.py` to support
   `polars` and `pandas` backends via lazy imports with clear error messages.
4. Update `with_features` in
   `src/codeintel/build/hamilton/transforms/decorators.py` to use the registry.

Touchpoints:
- `src/codeintel/build/hamilton/column_ops/__init__.py`
- `src/codeintel/build/hamilton/transforms/with_columns_backend.py`
- `src/codeintel/build/hamilton/transforms/decorators.py`
- New: `src/codeintel/build/hamilton/column_ops/registry.py`

## 7. Consolidate tabular pipelines into a single builder

Goal:
- Provide one config-driven pipeline builder for cleaning, feature ops,
  alignment, and canonicalization.

Design sketch:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.function_modifiers import pipe_input, pipe_output, resolve_from_config, step, value

if TYPE_CHECKING:
    from hamilton.function_modifiers.base import NodeTransformLifecycle


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    input_name: str
    clean_namespace: str
    output_namespace: str


def build_tabular_pipeline(spec: PipelineSpec) -> NodeTransformLifecycle:
    def _factory(*, clean_mode: str = "lenient") -> NodeTransformLifecycle:
        steps = [
            step(drop_bad_rows, required_cols=value(("loc", "cyclo"))).when(
                clean_mode="strict"
            ),
            step(normalize_nulls, policy=value("preserve")).named(
                "nulls",
                namespace=spec.clean_namespace,
            ),
        ]
        return pipe_input(*steps, on_input=spec.input_name, namespace=spec.clean_namespace)

    return resolve_from_config(decorate_with=_factory)
```

Implementation steps:
1. Add `src/codeintel/build/hamilton/transforms/pipelines.py` with
   `PipelineSpec` and `build_tabular_pipeline`.
2. Replace `pipe_clean_df`, `pipe_contract_output`, and `with_features` wiring
   in `src/codeintel/build/hamilton/transforms/decorators.py` with the builder.
3. Update `src/codeintel/build/hamilton/transforms/ingestion_normalize.py` to
   call the new pipeline for ingestion normalization.
4. Update `src/codeintel/build/hamilton/transforms/table_contract.py` to use
   the builder for a single contract pipeline entry point.

Touchpoints:
- `src/codeintel/build/hamilton/transforms/decorators.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`
- `src/codeintel/build/hamilton/transforms/table_contract.py`
- New: `src/codeintel/build/hamilton/transforms/pipelines.py`

## 8. Parameterize analytics nodes from specs

Goal:
- Generate repeated analytics nodes via `@parameterize` or
  `@parameterize_frame` for consistency and reduced boilerplate.

Design sketch:
```python
from __future__ import annotations

from hamilton.function_modifiers import parameterize, source, value


_METRIC_SPECS = {
    "graph_metrics__calls": {"frame": source("q__graph__call_graph_edges"), "metric": value("calls")},
    "graph_metrics__imports": {
        "frame": source("q__graph__import_graph_edges"),
        "metric": value("imports"),
    },
}


@parameterize(**_METRIC_SPECS)
def graph_metric(frame: object, metric: str) -> object:
    return compute_metric(frame, metric)
```

Implementation steps:
1. Identify repeated compute blocks in
   `src/codeintel/build/hamilton/native/analytics/graph_metrics.py` and
   `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`.
2. Introduce `*_SPECS` dicts or a DataFrame spec and generate nodes with
   `@parameterize` or `@parameterize_frame`.
3. Ensure generated node names follow the existing naming scheme used by
   `DagCatalog` and output tags.

Touchpoints:
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/cfg_dfg_metrics.py`

## 9. Tag nodes created by pipe and with_columns

Goal:
- Ensure generated nodes always carry catalog tags for validation and
  observability.

Design sketch:
```python
from __future__ import annotations

from collections.abc import Sequence

from hamilton.function_modifiers import tag as h_tag


def tag_generated_nodes(
    *,
    domain: str,
    target: str,
    node_names: Sequence[str],
    node_type: str,
) -> object:
    return h_tag(
        target_=list(node_names),
        domain=domain,
        target=target,
        node_type=node_type,
    )
```

Implementation steps:
1. Add `tag_generated_nodes` to `src/codeintel/build/hamilton/tagging.py`.
2. In pipeline builders, use stable step names with `step(...).named(...)`.
3. Apply `tag_generated_nodes` to named steps in
   `src/codeintel/build/hamilton/transforms/decorators.py` and
   `src/codeintel/build/hamilton/transforms/pipelines.py`.
4. For `with_columns`, tag outputs produced by selected column ops in
   `src/codeintel/build/hamilton/transforms/with_columns_backend.py`.

Touchpoints:
- `src/codeintel/build/hamilton/tagging.py`
- `src/codeintel/build/hamilton/transforms/decorators.py`
- `src/codeintel/build/hamilton/transforms/with_columns_backend.py`
- `src/codeintel/build/hamilton/transforms/pipelines.py`

## 10. Dynamic execution wiring helper

Goal:
- Automatically enable dynamic execution for DAGs that use
  `Parallelizable` and `Collect`.

Design sketch:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

from hamilton.htypes import Collect, Parallelizable

if TYPE_CHECKING:
    from hamilton.driver import Builder
    from hamilton.graph import FunctionGraph


def requires_dynamic_execution(graph: FunctionGraph) -> bool:
    for node in graph.get_nodes():
        output_type = getattr(node, "type", None)
        if output_type in {Parallelizable, Collect}:
            return True
    return False


def apply_dynamic_execution_if_needed(
    *,
    builder: Builder,
    graph: FunctionGraph,
    profile: ExecutionProfile,
) -> Builder:
    if not requires_dynamic_execution(graph):
        return builder
    return apply_dynamic_execution(builder, profile)
```

Implementation steps:
1. Add `requires_dynamic_execution` and
   `apply_dynamic_execution_if_needed` to
   `src/codeintel/build/hamilton/execution_profiles.py`.
2. Call the helper in `src/codeintel/build/hamilton/executor.py` after driver
   graph creation (or after runtime composition if needed).
3. Remove custom dynamic executor wiring from ingestion targets and rely on
   the centralized helper.

Touchpoints:
- `src/codeintel/build/hamilton/executor.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/execution_profiles.py`

## Suggested rollout order

1. Execution profile factory (completed).
2. Dynamic execution helper (uses new profile).
3. Cache policy centralization (completed).
4. Input bundle collector factory (easy extraction and high reuse).
5. Materializer base/mixin (small refactor surface, improves consistency).
6. Tabular pipeline builder (touches multiple decorators).
7. Tagging of generated nodes (needs stable names from pipeline builder).
8. with_columns backend + registry (depends on tagging conventions).
9. Parameterized subdags for graph pipelines.
10. Analytics parameterization (use stable patterns from the other steps).

## Validation gates (apply after each batch)

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted `pytest` for impacted areas, followed by segmented directory runs.
