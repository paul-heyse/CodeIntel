
# Functional requirements this implementation plan intends to achieve #

---

## 1. Single entrypoint for running the system

**FR1 – Unified pipeline entrypoint**

* Provide one programmatic API (`run_pipeline(...)`) that can run:

  * **Ingest only**
  * **Graphs only**
  * **Analytics only**
  * **Full pipeline (ingest → graphs → analytics)**
* Selection is controlled by a `PipelineSpec` (e.g. `FULL_PIPELINE`, `INGEST_ONLY`, etc.).

---

## 2. Declarative pipeline description

**FR2 – Declarative pipeline specs**

* Allow pipelines to be described as **data** via:

  * `PipelineSpec` (id, description, ordered `PipelineStage`s).
  * `PipelineStage` (engine module + stage name + required/optional).
* Include a small library of canonical specs:

  * `FULL_PIPELINE`, `INGEST_ONLY`, `GRAPHS_ONLY`, `ANALYTICS_ONLY`.
* Make it easy to define new specs (e.g. “risk-only analytics”, “metrics-only graphs”) without changing executor logic.

---

## 3. Use existing engines, don’t re-implement them

**FR3 – Delegate to ingestion recipes**

* For ingestion stages:

  * Resolve stage name → `IngestRecipe` (e.g. `"builtin.default"` → `FULL_RECIPE`).
  * Execute via `execute_recipe_for_context(...)`.
* All incremental/diffing behavior, AST/CST extraction, coverage, etc. remains inside the ingestion engine.

**FR4 – Delegate to graph runtime / recipes**

* For graph stages:

  * Build a graph plan (`GraphPluginExecutionPlan`) based on stage name.
  * Execute via `run_graph_plugins(plan, context)`.
* Reuse existing graph runtime logic for:

  * plugin selection,
  * caching / skip-on-unchanged (policy),
  * manifest handling and validation.

**FR5 – Delegate to analytics pipeline bridge**

* For analytics stages:

  * Build `AnalyticsPlanRequest` with a plugin set (e.g. `"builtin.full"` → “full” plugin bundle).
  * Plan via `plan_analytics_plugin_run(...)`.
  * Execute via `run_analytics_plugins_for_context(...)`.
* All detailed metrics, risk, profiles, etc. remain inside the analytics engine.

---

## 4. Single run identity & consistent context

**FR6 – Single `RunContext` per pipeline**

* Each pipeline run creates exactly one `RunContext` (via `new_run_context`), shared across all stages.
* `RunContext.kind` is inferred from the `PipelineSpec`:

  * `ingest`, `graphs`, `analytics`, or `full`.
* Engines receive the same `run_id` and can correlate their steps to the same run.

**FR7 – Operate over `StorageGateway`**

* The orchestrator never manages raw DuckDB connections directly.
* It always accepts and passes around a `StorageGateway`:

  * For dataset access,
  * For `gateway.runs` (run tracking),
  * For core/graphs/analytics helpers.

---

## 5. Run tracking & observability

**FR8 – Start and complete pipeline runs**

* For each call to `run_pipeline`:

  * Create a pipeline run via `runs.start_run(run_context, pipeline_name=spec.id, status="running")`.
  * On completion, call `runs.complete_run(run_id, status=..., error_summary=...)`.
* Return the final `PipelineRunRecord`.

**FR9 – Stage-level step records**

* For each stage in `PipelineSpec.stages`:

  * Insert a “stage-level” step into `pipeline_steps` with:

    * `module` (ingestion/graphs/analytics),
    * `stage="orchestrator"`,
    * `name` (stage name),
    * `status` (`running`/`succeeded`/`failed`),
    * timestamps and optional error summary.
* Engines continue to record **plugin-level steps** as they already do.

---

## 6. Failure handling and control flow

**FR10 – Required vs optional stages**

* If a **required** stage fails:

  * Mark the stage step as `failed`.
  * Mark the overall run as `failed`.
  * **Stop** executing any further stages (fail-fast).
* If an **optional** stage fails:

  * Mark the stage step as `failed`.
  * Keep the overall run `succeeded` (or `succeeded_with_warnings` if you choose to add that later).
  * Continue (if there are later stages).

**FR11 – Mode correctness**

* `INGEST_ONLY` must *not* require graphs or analytics to be runnable.
* `GRAPHS_ONLY` and `ANALYTICS_ONLY`:

  * Either rely on pre-existing ingestion/graphs,
  * Or (in a later enhancement) perform light precondition checks and fail fast with clear errors if prerequisites are missing.

---

## 7. Integration & extensibility

**FR12 – CLI integration**

* Provide a simple CLI integration point (e.g. `codeintel pipeline run-unified --mode full|ingest|graphs|analytics`) that:

  * Parses mode → chooses a `PipelineSpec`.
  * Builds `StorageGateway`, `SnapshotRef`, `BuildPaths`, `ToolsConfig`, `ConfigRegistry`.
  * Calls `run_pipeline(...)` and exits non-zero on failure.

**FR13 – Extensible stage naming**

* Stage `name` values are interpreted via pluggable mapping functions on the planner side:

  * Ingestion: `builtin.default`, `builtin.incremental`, custom recipe names.
  * Graphs: `builtin.full`, future plugin bundles (`metrics_only`, `builders_only`, etc.).
  * Analytics: `builtin.full`, future subsets like `risk_only`, `tests_only`.
* Adding new flavors **must not** require changing the executor core; only the planner mappings or new specs.

**FR14 – Backwards compatibility**

* Existing step-based orchestration (individual `steps_ingestion`, `steps_graphs`, `steps_analytics`) remains usable for:

  * Prefect flows,
  * Fine-grained pipelines.
* Unified orchestrator sits *above* engines and *alongside* step-level orchestration, not replacing it wholesale.

---

# Detailed implementation plan with code snippets #

* Use **StorageGateway + PipelineRunTracking** (not raw DuckDB connections).
* Use **`runtime.RunContext`** as the unified run identity.
* Call **existing engines**:

  * ingestion: `execute_recipe_for_context`
  * graphs: `plan_graph_plugin_run` + `run_graph_plugins`
  * analytics: `plan_analytics_plugin_run` + `run_analytics_plugins_for_context`

---

## 1. `spec.py` – declarative pipeline spec model

### 1.1 Types and canonical specs

Create `codeintel/pipeline/spec.py`:

```python
# src/codeintel/pipeline/spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Shares the same vocabulary as storage.run_tracking.ModuleKind
StageModule = Literal["ingestion", "graphs", "analytics"]


@dataclass(frozen=True)
class PipelineStage:
    """
    A single logical stage in a pipeline spec.

    Attributes
    ----------
    module
        Which engine should execute this stage:
        - "ingestion"
        - "graphs"
        - "analytics"
    name
        Stage flavor / plan identifier, interpreted by the planner.
        For example:
        - "builtin.default"      -> default ingestion recipe
        - "builtin.full"         -> all graphs + analytics
        - "builtin.metrics_only" -> analytics metrics-only plan
    description
        Human-readable description.
    required
        If True, a failure aborts the pipeline.
        If False, failures are recorded but execution continues.
    """

    module: StageModule
    name: str
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class PipelineSpec:
    """
    Declarative pipeline specification.

    Attributes
    ----------
    id
        Identifier used in run tracking and CLI.
    description
        Human-readable description.
    stages
        Ordered stages to execute.
    """

    id: str
    description: str
    stages: tuple[PipelineStage, ...]
```

Now add canonical specs your high-level plan mentioned, aligned with the current engines:

```python
# src/codeintel/pipeline/spec.py (continued)

FULL_PIPELINE = PipelineSpec(
    id="full",
    description="Ingest + graphs + analytics",
    stages=(
        PipelineStage(
            module="ingestion",
            name="builtin.default",  # maps to FULL_RECIPE
            description="Full ingestion (AST/CST, coverage, config, profiles, etc.)",
        ),
        PipelineStage(
            module="graphs",
            name="builtin.full",  # maps to 'full' graph recipe / runtime set
            description="All graph builders + metrics + validation",
        ),
        PipelineStage(
            module="analytics",
            name="builtin.full",  # maps to 'full' analytics plugin set
            description="Full analytics plugin bundle (functions, risk, tests, etc.)",
        ),
    ),
)

INGEST_ONLY = PipelineSpec(
    id="ingest",
    description="Ingestion only",
    stages=(
        PipelineStage(
            module="ingestion",
            name="builtin.default",
            description="Full ingestion only",
        ),
    ),
)

GRAPHS_ONLY = PipelineSpec(
    id="graphs",
    description="Graphs only (assumes ingestion already run)",
    stages=(
        PipelineStage(
            module="graphs",
            name="builtin.full",
            description="All graph builders + metrics + validation",
        ),
    ),
)

ANALYTICS_ONLY = PipelineSpec(
    id="analytics",
    description="Analytics only (assumes graphs already built)",
    stages=(
        PipelineStage(
            module="analytics",
            name="builtin.full",
            description="All analytics plugins",
        ),
    ),
)

PIPELINE_SPECS: dict[str, PipelineSpec] = {
    spec.id: spec
    for spec in (
        FULL_PIPELINE,
        INGEST_ONLY,
        GRAPHS_ONLY,
        ANALYTICS_ONLY,
    )
}


def get_pipeline_spec(spec_id: str) -> PipelineSpec:
    """
    Look up a pipeline spec by ID.

    Raises
    ------
    KeyError
        If no spec is registered for the given ID.
    """
    return PIPELINE_SPECS[spec_id]
```

This keeps the *declarative* part very small: a spec is just `(module, name)` stages in a sequence; the heavy translation into recipes/plugin sets happens in `planner.py`.

---

## 2. `planner.py` – translate specs into engine-level plans

The planner is responsible for turning a `PipelineSpec` into concrete per-engine plans:

* For **ingestion**: resolve which `IngestRecipe` + `RecipeExecutorContext` to use.
* For **graphs**: resolve which graph plugins to run and build a `GraphPluginExecutionPlan` + `GraphExecutorContext`.
* For **analytics**: choose analytics plugin names and build `AnalyticsPlanRequest` + `AnalyticsPluginExecutionPlan`.

### 2.1 Plan dataclasses

Create `codeintel/pipeline/planner.py`:

```python
# src/codeintel/pipeline/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from codeintel.pipeline.spec import PipelineSpec, PipelineStage, StageModule
from codeintel.runtime import RunContext, RunKind
from codeintel.runtime.orchestrator import new_run_context

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef, BuildPaths
    from codeintel.config.models import ToolsConfig
    from codeintel.core.config_registry import ConfigRegistry
    from codeintel.storage.gateway import StorageGateway
    from codeintel.graphs.runtime.executor import GraphExecutorContext, GraphRunReport
    from codeintel.graphs.runtime.planning import GraphPluginExecutionPlan
    from codeintel.analytics.core.pipeline_bridge import (
        AnalyticsPlanRequest,
        AnalyticsPluginExecutionPlan,
        AnalyticsRunContext,
    )
    from codeintel.ingestion.recipes.dsl import IngestRecipe, RecipeOptions
    from codeintel.ingestion.recipes.executor import RecipeExecutorContext
    from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
```

```python
@dataclass(frozen=True)
class IngestionStagePlan:
    stage: PipelineStage
    recipe: "IngestRecipe"
    options: "RecipeOptions"
    context: "RecipeExecutorContext"


@dataclass(frozen=True)
class GraphsStagePlan:
    stage: PipelineStage
    plan: "GraphPluginExecutionPlan"
    context: "GraphExecutorContext"


@dataclass(frozen=True)
class AnalyticsStagePlan:
    stage: PipelineStage
    request: "AnalyticsPlanRequest"
    plan: "AnalyticsPluginExecutionPlan"
    context: "AnalyticsRunContext"


@dataclass(frozen=True)
class PipelinePlan:
    """
    Fully resolved execution plan for a PipelineSpec.

    This binds a spec to:
    - a concrete RunContext (shared across stages),
    - per-engine execution plans and contexts.
    """

    spec: PipelineSpec
    run_context: RunContext
    ingestion: IngestionStagePlan | None
    graphs: GraphsStagePlan | None
    analytics: AnalyticsStagePlan | None
```

### 2.2 Deriving the RunKind from the spec

We want a single `RunContext` shared across stages; its `kind` is inferred from which modules appear in the spec:

```python
def _infer_run_kind(spec: PipelineSpec) -> RunKind:
    modules = {stage.module for stage in spec.stages}
    if modules == {"ingestion"}:
        return "ingest"
    if modules == {"graphs"}:
        return "graphs"
    if modules == {"analytics"}:
        return "analytics"
    # Mixed -> full pipeline
    return "full"
```

### 2.3 Stage-specific plan helpers

#### 2.3.1 Ingestion stage planning

Use the new ingestion recipes (`ingestion.recipes.builtin`) and the recipe executor context you’re already using in `pipeline.cli.main`.

```python
from codeintel.ingestion.recipes.builtin import FULL_RECIPE, INCREMENTAL_RECIPE, get_builtin_recipe
from codeintel.ingestion.recipes.dsl import IngestRecipe, RecipeOptions
from codeintel.ingestion.recipes.executor import RecipeExecutorContext
from codeintel.ingestion.tool_service import ToolService
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.pipeline.orchestration.core import PipelineContext
from codeintel.core.config_registry import ConfigRegistry


def _resolve_ingest_recipe(stage: PipelineStage) -> IngestRecipe:
    """
    Resolve the ingestion recipe for a stage name.

    Convention:
      - "builtin.default"  -> FULL_RECIPE
      - "builtin.incremental" -> INCREMENTAL_RECIPE
      - "builtin.<name>"   -> get_builtin_recipe(<name>)
      - otherwise          -> get_builtin_recipe(stage.name)
    """
    name = stage.name
    if name == "builtin.default":
        return FULL_RECIPE
    if name == "builtin.incremental":
        return INCREMENTAL_RECIPE
    if name.startswith("builtin."):
        return get_builtin_recipe(name.split(".", 1)[1])
    return get_builtin_recipe(name)
```

Planning function:

```python
def _plan_ingestion_stage(
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    config_registry: ConfigRegistry,
) -> IngestionStagePlan:
    recipe = _resolve_ingest_recipe(stage)
    # Build profiles exactly like CLI
    code_profile = profile_from_env(default_code_profile(snapshot.repo_root))
    config_profile = profile_from_env(default_config_profile(snapshot.repo_root))

    # Optional: derive tool runner/service the same way CLI does
    tool_runner = ToolRunner(
        repo_root=snapshot.repo_root,
        tool_binaries=config_registry.tool_binaries,
    )
    tool_service = ToolService(config_registry=config_registry)

    ctx = RecipeExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
        code_profile=code_profile,
        config_profile=config_profile,
        tool_runner=tool_runner,
        tool_service=tool_service,
        run_context=run_ctx,
    )

    # Options are intentionally lightweight for first pass
    options = RecipeOptions()  # you can thread overrides later

    return IngestionStagePlan(stage=stage, recipe=recipe, options=options, context=ctx)
```

#### 2.3.2 Graphs stage planning (via `graphs.runtime`)

Rather than using the older step wrappers, we use the new runtime layer: `GraphPluginPolicy`, `GraphRunScope`, `GraphPlanContext`, and `plan_graph_plugin_run`.

```python
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.graphs.runtime.manifest import load_prior_manifest
from codeintel.graphs.runtime.planning import GraphPlanContext, plan_graph_plugin_run
from codeintel.graphs.runtime.executor import GraphExecutorContext
from codeintel.graphs.engine_factory import ensure_graph_engine
from codeintel.graphs.catalog import FunctionCatalogProvider
from codeintel.pipeline.orchestration.core import _graph_runtime, _function_catalog
```

We can mimic the defaults in `pipeline/orchestration/steps_graphs.py`:

```python
def _default_graph_policy() -> GraphPluginPolicy:
    # Conservative: fail_fast, no skip_on_unchanged for now.
    return GraphPluginPolicy(
        fail_fast=True,
        skip_on_unchanged=False,
    )


def _plan_graphs_stage(
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    ctx: PipelineContext,
) -> GraphsStagePlan:
    """
    Plan graphs execution using the new runtime layer.

    stage.name
    ----------
    - "builtin.full": builders + metrics + validation (default plugin set)
    - future: other names could map to plugin subsets.
    """
    policy = _default_graph_policy()
    scope = GraphRunScope()

    # For now, treat "builtin.full" as "all registered plugins"
    plugin_names: Sequence[str] | None = None

    manifest_path = ctx.build_dir / "manifests" / "graphs.json"
    prior_manifest = load_prior_manifest(manifest_path)

    plan_ctx = GraphPlanContext(
        cfg=ctx.config_builder().graph_metrics(),
        runtime_snapshot=snapshot,
        target=(snapshot.repo, snapshot.commit),
        policy=policy,
        run_options={},
        prior_manifest=prior_manifest,
        # Graph runtime handles manifest hashing etc.
    )

    plan = plan_graph_plugin_run(
        plugin_names=plugin_names,
        context=plan_ctx,
    )

    # Build executor context
    graph_runtime = _graph_runtime(ctx)
    catalog = _function_catalog(ctx)

    exec_ctx = GraphExecutorContext(
        gateway=ctx.gateway,
        snapshot=snapshot,
        engine=graph_runtime.engine,
        catalog_provider=catalog,
        run_context=run_ctx,
    )

    return GraphsStagePlan(stage=stage, plan=plan, context=exec_ctx)
```

(You can refine which plugin names are chosen for other stage names later by looking at plugin groups or recipes.)

#### 2.3.3 Analytics stage planning (via `analytics.core.pipeline_bridge`)

Use the same plugin bundles as `pipeline/orchestration/steps_analytics.py`, but drive them through a single “full” set by default.

```python
from collections.abc import Sequence

from codeintel.analytics.core.pipeline_bridge import (
    AnalyticsPlanRequest,
    AnalyticsRunContext,
    plan_analytics_plugin_run,
)
from codeintel.analytics.core.plugins import (
    BEHAVIORAL_COVERAGE_PLUGIN,
    CONFIG_DATA_FLOW_PLUGIN,
    COVERAGE_FUNCTIONS_PLUGIN,
    COVERAGE_TEST_EDGES_PLUGIN,
    DATA_MODEL_USAGE_PLUGIN,
    DATA_MODELS_PLUGIN,
    ENTRYPOINTS_PLUGIN,
    EXTERNAL_DEPS_PLUGIN,
    FUNCTION_AST_FEATURES_PLUGIN,
    FUNCTION_CONTRACTS_PLUGIN,
    FUNCTION_EFFECTS_PLUGIN,
    FUNCTION_HISTORY_PLUGIN,
    FUNCTION_METRICS_PLUGIN,
    HISTORY_TIMESERIES_PLUGIN,
    HOTSPOTS_PLUGIN,
    PROFILES_PLUGIN,
    RISK_FACTORS_PLUGIN,
    SEMANTIC_ROLES_PLUGIN,
    SUBSYSTEMS_PLUGIN,
    TEST_PROFILE_PLUGIN,
)
from codeintel.analytics.graphs.runtime import load_prior_manifest
from codeintel.analytics.runtime_manifest import encode_manifest
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
```

Define the default full plugin set:

```python
DEFAULT_ANALYTICS_FULL_PLUGINS: tuple[str, ...] = (
    FUNCTION_METRICS_PLUGIN.metadata.name,
    FUNCTION_EFFECTS_PLUGIN.metadata.name,
    FUNCTION_AST_FEATURES_PLUGIN.metadata.name,
    FUNCTION_CONTRACTS_PLUGIN.metadata.name,
    FUNCTION_HISTORY_PLUGIN.metadata.name,
    HISTORY_TIMESERIES_PLUGIN.metadata.name,
    HOTSPOTS_PLUGIN.metadata.name,
    RISK_FACTORS_PLUGIN.metadata.name,
    PROFILES_PLUGIN.metadata.name,
    TEST_PROFILE_PLUGIN.metadata.name,
    COVERAGE_FUNCTIONS_PLUGIN.metadata.name,
    COVERAGE_TEST_EDGES_PLUGIN.metadata.name,
    BEHAVIORAL_COVERAGE_PLUGIN.metadata.name,
    DATA_MODELS_PLUGIN.metadata.name,
    DATA_MODEL_USAGE_PLUGIN.metadata.name,
    SUBSYSTEMS_PLUGIN.metadata.name,
    ENTRYPOINTS_PLUGIN.metadata.name,
    EXTERNAL_DEPS_PLUGIN.metadata.name,
    SEMANTIC_ROLES_PLUGIN.metadata.name,
    CONFIG_DATA_FLOW_PLUGIN.metadata.name,
)
```

Planner:

```python
def _analytics_plugins_for_stage(stage: PipelineStage) -> Sequence[str]:
    name = stage.name
    if name == "builtin.full":
        return DEFAULT_ANALYTICS_FULL_PLUGINS
    # Future: "builtin.tests_only", "builtin.risk_only", etc.
    return DEFAULT_ANALYTICS_FULL_PLUGINS


def _plan_analytics_stage(
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    ctx: PipelineContext,
) -> AnalyticsStagePlan:
    plugin_names = _analytics_plugins_for_stage(stage)
    policy = GraphPluginPolicy()
    scope = GraphRunScope()

    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    manifest_path = ctx.build_dir / "manifests" / "analytics.json"
    prior_manifest = load_prior_manifest(manifest_path)

    request = AnalyticsPlanRequest(
        plugin_names=plugin_names,
        policy=policy,
        repo=snapshot.repo,
        commit=snapshot.commit,
        scope=scope,
        prior_manifest=prior_manifest,
        cfg_options={"graph_metrics": cfg.__dict__},
        runtime_options={},
        run_id=run_ctx.run_id,
    )

    plan = plan_analytics_plugin_run(request)

    run_context = AnalyticsRunContext(
        gateway=ctx.gateway,
        snapshot=snapshot,
        graph_runtime=_graph_runtime(ctx),
        cfgs={"graph_metrics": cfg},
        extra={"tool_runner": ctx.tool_runner},
        catalog_provider=_function_catalog(ctx),
    )

    return AnalyticsStagePlan(stage=stage, request=request, plan=plan, context=run_context)
```

### 2.4 Top-level `build_pipeline_plan`

Finally, tie it all together:

```python
from codeintel.pipeline.orchestration.core import PipelineContext
from codeintel.core.config_registry import ConfigRegistry


def build_pipeline_plan(
    *,
    spec: PipelineSpec,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    config_registry: ConfigRegistry,
    trigger: str = "cli",
) -> PipelinePlan:
    """
    Build a concrete pipeline plan for a spec + snapshot.

    This is the main entrypoint used by the executor layer.
    """
    run_kind = _infer_run_kind(spec)
    run_ctx = new_run_context(
        snapshot=snapshot,
        kind=run_kind,
        trigger=trigger,
    )

    # Build a PipelineContext so we can reuse existing config builders, graph runtime, etc.
    pipeline_ctx = PipelineContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        code_profile_cfg=None,   # resolved lazily in _resolve_code_profile
        config_profile_cfg=None,
        config_registry=config_registry,
        tools=tools,
        run_id=run_ctx.run_id,
        run_context=run_ctx,
    )

    ingestion_plan: IngestionStagePlan | None = None
    graphs_plan: GraphsStagePlan | None = None
    analytics_plan: AnalyticsStagePlan | None = None

    for stage in spec.stages:
        if stage.module == "ingestion":
            ingestion_plan = _plan_ingestion_stage(
                stage=stage,
                run_ctx=run_ctx,
                snapshot=snapshot,
                paths=paths,
                gateway=gateway,
                tools=tools,
                config_registry=config_registry,
            )
        elif stage.module == "graphs":
            graphs_plan = _plan_graphs_stage(
                stage=stage,
                run_ctx=run_ctx,
                snapshot=snapshot,
                ctx=pipeline_ctx,
            )
        elif stage.module == "analytics":
            analytics_plan = _plan_analytics_stage(
                stage=stage,
                run_ctx=run_ctx,
                snapshot=snapshot,
                ctx=pipeline_ctx,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported stage module: {stage.module}")

    return PipelinePlan(
        spec=spec,
        run_context=run_ctx,
        ingestion=ingestion_plan,
        graphs=graphs_plan,
        analytics=analytics_plan,
    )
```

---

## 3. `executor.py` – run a plan and drive run-tracking

This is where we:

* Start the pipeline run in **PipelineRunTracking**.
* Optionally record **stage-level step records** (top-down).
* Delegate to the engines (which record plugin-level steps).
* Mark the run as succeeded / failed and return a `PipelineRunRecord`.

### 3.1 Signature and imports

```python
# src/codeintel/pipeline/executor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.pipeline.planner import PipelinePlan
from codeintel.pipeline.spec import PipelineSpec, PipelineStage
from codeintel.pipeline.run_registry import (
    ModuleKind,
    PipelineRunRecord,
    PipelineRunTracking,
    PipelineStatus,
    PipelineStepRecord,
    StepStatus,
)
from codeintel.ingestion.recipes.executor import execute_recipe_for_context
from codeintel.graphs.runtime.executor import run_graph_plugins
from codeintel.analytics.core.pipeline_bridge import run_analytics_plugins_for_context

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef, BuildPaths
    from codeintel.config.models import ToolsConfig
    from codeintel.core.config_registry import ConfigRegistry
    from codeintel.storage.gateway import StorageGateway
```

### 3.2 Stage-level record helpers

```python
def _now() -> datetime:
    return datetime.now(tz=UTC)


def _start_stage_step(
    runs: PipelineRunTracking,
    run_id: str,
    stage: PipelineStage,
) -> None:
    """Insert a 'running' stage-level step record."""
    runs.record_step(
        PipelineStepRecord(
            run_id=run_id,
            module=stage.module,
            stage="orchestrator",
            name=stage.name,
            status="running",
            started_at=_now(),
            completed_at=None,
            row_counts=None,
            extra=None,
        )
    )


def _complete_stage_step(
    runs: PipelineRunTracking,
    run_id: str,
    stage: PipelineStage,
    status: StepStatus,
    *,
    error: str | None = None,
) -> None:
    extra = {"error": error} if error else None
    runs.record_step(
        PipelineStepRecord(
            run_id=run_id,
            module=stage.module,
            stage="orchestrator",
            name=stage.name,
            status=status,
            started_at=_now(),   # we don't persist the original start_t; acceptable approximation
            completed_at=_now(),
            row_counts=None,
            extra=extra,
        )
    )
```

### 3.3 Main `run_pipeline` function

```python
from codeintel.pipeline.planner import build_pipeline_plan


def run_pipeline(
    *,
    spec: PipelineSpec,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    config_registry: ConfigRegistry,
) -> PipelineRunRecord:
    """
    Execute a unified pipeline over ingestion, graphs, and analytics.

    Parameters
    ----------
    spec
        Declarative pipeline spec (e.g. FULL_PIPELINE).
    snapshot
        Repository snapshot to operate on.
    paths
        Build paths for this run.
    gateway
        Storage gateway for DuckDB and datasets.
    tools
        Tools configuration (used by ingestion and analytics).
    config_registry
        Config registry with profiles, tool binaries, etc.

    Returns
    -------
    PipelineRunRecord
        Final run record from the run tracking table.

    Raises
    ------
    RuntimeError
        If a required stage fails.
    """
    runs: PipelineRunTracking = gateway.runs

    plan = build_pipeline_plan(
        spec=spec,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        config_registry=config_registry,
        trigger="cli",
    )
    run_ctx = plan.run_context
    run_id = run_ctx.run_id

    # Mark run as started
    runs.start_run(
        run_ctx,
        pipeline_name=spec.id,
        status="running",
    )

    overall_status: PipelineStatus = "succeeded"
    last_error: str | None = None

    # 1) Ingestion
    if plan.ingestion:
        stage = plan.ingestion.stage
        _start_stage_step(runs, run_id, stage)
        try:
            result = execute_recipe_for_context(
                recipe=plan.ingestion.recipe,
                run_context=run_ctx,
                context=plan.ingestion.context,
                config=None,  # using defaults; extend later if needed
            )
            # ingestion executor already records plugin-level steps via gateway.runs
            _complete_stage_step(runs, run_id, stage, status="succeeded")
        except Exception as exc:  # noqa: BLE001
            msg = f"Ingestion stage {stage.name!r} failed: {exc}"
            last_error = msg
            _complete_stage_step(runs, run_id, stage, status="failed", error=str(exc))
            if stage.required:
                overall_status = "failed"
                # Fail fast: stop executing further stages
                runs.complete_run(run_id, status=overall_status, error_summary=msg)
                # Return the run record to caller
                return runs.fetch_run(run_id)  # type: ignore[return-value]

    # 2) Graphs
    if plan.graphs and overall_status == "succeeded":
        stage = plan.graphs.stage
        _start_stage_step(runs, run_id, stage)
        try:
            report = run_graph_plugins(
                plan=plan.graphs.plan,
                context=plan.graphs.context,
            )
            # graph runtime already records plugin-level steps via gateway.runs
            _complete_stage_step(runs, run_id, stage, status="succeeded")
        except Exception as exc:  # noqa: BLE001
            msg = f"Graphs stage {stage.name!r} failed: {exc}"
            last_error = msg
            _complete_stage_step(runs, run_id, stage, status="failed", error=str(exc))
            if stage.required:
                overall_status = "failed"
                runs.complete_run(run_id, status=overall_status, error_summary=msg)
                return runs.fetch_run(run_id)  # type: ignore[return-value]

    # 3) Analytics
    if plan.analytics and overall_status == "succeeded":
        stage = plan.analytics.stage
        _start_stage_step(runs, run_id, stage)
        try:
            report = run_analytics_plugins_for_context(
                unified_run_context=run_ctx,
                plan=plan.analytics.plan,
                run_context=plan.analytics.context,
            )
            # analytics pipeline_bridge already records plugin-level steps
            _complete_stage_step(runs, run_id, stage, status="succeeded")
        except Exception as exc:  # noqa: BLE001
            msg = f"Analytics stage {stage.name!r} failed: {exc}"
            last_error = msg
            _complete_stage_step(runs, run_id, stage, status="failed", error=str(exc))
            if stage.required:
                overall_status = "failed"
                runs.complete_run(run_id, status=overall_status, error_summary=msg)
                return runs.fetch_run(run_id)  # type: ignore[return-value]

    # If we reach here, no required stage failed
    runs.complete_run(run_id, status=overall_status, error_summary=last_error)
    run = runs.fetch_run(run_id)
    if run is None:  # pragma: no cover
        raise RuntimeError(f"Failed to fetch run record for run_id={run_id}")
    return run
```

This matches your original high-level flow:

* **Start run** once with `RunContext`.
* For each stage:

  * Insert stage-level `pipeline_steps` record with `status="running"`.
  * Dispatch to the appropriate engine and let it record plugin-level steps.
  * Insert final stage-level step with `status="succeeded"` or `"failed"`.
* On required stage failure, mark run failed and stop.

Note: returns `PipelineRunRecord` via `PipelineRunTracking.fetch_run`.

---

## 4. (Optional) Wiring Epic 9 into the CLI

Once spec/planner/executor exist, you can gradually route CLI commands to Epic 9 instead of ad-hoc flows. For example, add a CLI subcommand `run-unified`:

```python
# in pipeline/cli/main.py

from codeintel.pipeline.spec import FULL_PIPELINE, INGEST_ONLY, GRAPHS_ONLY, ANALYTICS_ONLY
from codeintel.pipeline.executor import run_pipeline

def handle_run_unified(args: argparse.Namespace, cfg: CodeIntelConfig) -> int:
    gateway = _open_gateway(cfg, read_only=False)
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = _build_paths_from_cli(cfg.paths)
    tools = cfg.tools

    if args.mode == "full":
        spec = FULL_PIPELINE
    elif args.mode == "ingest":
        spec = INGEST_ONLY
    elif args.mode == "graphs":
        spec = GRAPHS_ONLY
    elif args.mode == "analytics":
        spec = ANALYTICS_ONLY
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")

    run = run_pipeline(
        spec=spec,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        config_registry=cfg.config_registry,
    )

    if run.status != "succeeded":
        raise RuntimeError(f"Unified pipeline failed: {run.status} ({run.run_id})")

    return 0
```

You can keep the existing CLI entrypoints in parallel until you’re confident in the unified orchestrator.

---

## 5. Tests (Epic 9.3)

Finally, tests that actually exercise this orchestration over your **real** engines and run tracking, using the harnesses and fixtures you already have.

### 5.1 Smoke test: full unified pipeline

New file: `tests/orchestration/test_unified_pipeline_full.py`:

```python
from __future__ import annotations

from pathlib import Path

from codeintel.pipeline.spec import FULL_PIPELINE
from codeintel.pipeline.executor import run_pipeline
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.config_registry import ConfigRegistry
from codeintel.tests._helpers.repo_builder import build_tiny_repo  # if you have this helper

def test_unified_full_pipeline_smoke(tmp_path: Path) -> None:
    repo_root = build_tiny_repo(tmp_path)
    db_path = tmp_path / "codeintel.duckdb"

    gateway = open_gateway(StorageConfig(db_path=db_path))
    cfg_registry = ConfigRegistry.default_for_repo(repo_root)
    paths = BuildPaths.for_repo_root(repo_root)
    snapshot = SnapshotRef(
        repo="local/tiny",
        commit="HEAD",
        repo_root=repo_root,
    )

    run = run_pipeline(
        spec=FULL_PIPELINE,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=cfg_registry.tools,
        config_registry=cfg_registry,
    )

    assert run.status == "succeeded"
    steps = gateway.runs.fetch_steps(run.run_id)
    modules = {s.module for s in steps}
    assert {"ingestion", "graphs", "analytics"}.issubset(modules)
```

(Replace any helper imports with the concrete ones you actually have; you can copy patterns from `tests/test_pipeline_smoke.py`.)

### 5.2 Mode-specific tests

1. **Ingest only**: ensure only ingestion steps are present.

```python
from codeintel.pipeline.spec import INGEST_ONLY

def test_unified_ingest_only(tmp_path: Path, repo_root: Path, gateway, cfg_registry) -> None:
    # Use fixtures to get repo_root, gateway, cfg_registry
    paths = BuildPaths.for_repo_root(repo_root)
    snapshot = SnapshotRef(repo="local/test", commit="HEAD", repo_root=repo_root)

    run = run_pipeline(
        spec=INGEST_ONLY,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=cfg_registry.tools,
        config_registry=cfg_registry,
    )

    assert run.status == "succeeded"
    steps = gateway.runs.fetch_steps(run.run_id)
    assert {s.module for s in steps} == {"ingestion"}
```

2. **Fail-fast behavior**: configure a spec whose ingestion stage uses a deliberately broken recipe (for test only), assert that:

* run status is `"failed"`,
* graphs / analytics stages didn’t execute (no `module == "graphs"`, `"analytics"` steps),
* there is an orchestrator-level step record with `status="failed"` and `stage="orchestrator"`.

3. **Non-required stage**: spec with `required=False` on analytics, force analytics failure (e.g. by misconfiguring plugin names), and verify:

* run status remains `"succeeded"`,
* analytics stage step is recorded with `status="failed"` but earlier stages succeeded.

You can add a small helper spec just for tests:

```python
from codeintel.pipeline.spec import PipelineSpec, PipelineStage

OPTIONAL_ANALYTICS = PipelineSpec(
    id="opt_analytics",
    description="Ingestion + graphs, analytics optional",
    stages=(
        PipelineStage("ingestion", "builtin.default", required=True),
        PipelineStage("graphs", "builtin.full", required=True),
        PipelineStage("analytics", "builtin.full", required=False),
    ),
)
```

---

## 6. What this gives you

Putting it all together, Epic 9 ends up as:

* A **tiny spec layer** (`spec.py`) describing what should run.
* A **planner** (`planner.py`) that converts that spec into concrete **engine-specific plans** using:

  * ingestion recipes (`ingestion.recipes`),
  * graph runtime (`graphs.runtime`),
  * analytics pipeline bridge (`analytics.core.pipeline_bridge`).
* An **executor** (`executor.py`) that:

  * creates a single `RunContext` via `runtime.new_run_context`,
  * starts a pipeline run in `gateway.runs`,
  * runs each engine in sequence,
  * records orchestrator-level stage steps in `pipeline_steps`,
  * relies on the engines to record plugin-level steps,
  * and returns a `PipelineRunRecord`.

