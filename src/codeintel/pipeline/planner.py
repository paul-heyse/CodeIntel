"""Pipeline planning for unified orchestration.

This module translates declarative PipelineSpec definitions into concrete
execution plans with engine-specific contexts and configurations. The planner
handles recipe resolution, plugin selection, and context construction for each
stage module.

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.pipeline.spec import PipelineSpec, PipelineStage
from codeintel.runtime import RunKind, TriggerKind, new_run_context

if TYPE_CHECKING:
    from codeintel.analytics.core.pipeline_bridge import (
        AnalyticsPlanRequest,
        AnalyticsPluginExecutionPlan,
        AnalyticsRunContext,
    )
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.graphs.runtime.executor import GraphExecutorContext
    from codeintel.graphs.runtime.planning import GraphPluginExecutionPlan
    from codeintel.ingestion.recipes.dsl import IngestRecipe, RecipeOptions
    from codeintel.ingestion.recipes.executor import RecipeExecutorContext
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Stage Plan Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionStagePlan:
    """Resolved plan for an ingestion stage.

    Attributes
    ----------
    stage
        The pipeline stage this plan corresponds to.
    recipe
        Resolved ingestion recipe to execute.
    options
        Recipe execution options.
    context
        Executor context with all dependencies.
    """

    stage: PipelineStage
    recipe: IngestRecipe
    options: RecipeOptions
    context: RecipeExecutorContext


@dataclass(frozen=True)
class GraphsStagePlan:
    """Resolved plan for a graphs stage.

    Attributes
    ----------
    stage
        The pipeline stage this plan corresponds to.
    plan
        Graph plugin execution plan with plugin ordering.
    context
        Executor context with gateway, snapshot, and engine.
    """

    stage: PipelineStage
    plan: GraphPluginExecutionPlan
    context: GraphExecutorContext


@dataclass(frozen=True)
class AnalyticsStagePlan:
    """Resolved plan for an analytics stage.

    Attributes
    ----------
    stage
        The pipeline stage this plan corresponds to.
    request
        Analytics plan request parameters.
    plan
        Analytics plugin execution plan.
    context
        Analytics run context with runtime dependencies.
    """

    stage: PipelineStage
    request: AnalyticsPlanRequest
    plan: AnalyticsPluginExecutionPlan
    context: AnalyticsRunContext


@dataclass(frozen=True)
class PipelinePlan:
    """Fully resolved execution plan for a PipelineSpec.

    Bundles the spec with a unified RunContext and per-engine plans.
    The executor uses this to dispatch to appropriate engine entrypoints.

    Attributes
    ----------
    spec
        Original pipeline specification.
    run_context
        Unified run context shared across all stages.
    ingestion
        Ingestion stage plan, if spec includes ingestion.
    graphs
        Graphs stage plan, if spec includes graphs.
    analytics
        Analytics stage plan, if spec includes analytics.
    """

    spec: PipelineSpec
    run_context: RunContext
    ingestion: IngestionStagePlan | None
    graphs: GraphsStagePlan | None
    analytics: AnalyticsStagePlan | None


# -----------------------------------------------------------------------------
# RunKind Inference
# -----------------------------------------------------------------------------


def _infer_run_kind(spec: PipelineSpec) -> RunKind:
    """Infer the RunKind from the modules present in a pipeline spec.

    Parameters
    ----------
    spec
        Pipeline specification to analyze.

    Returns
    -------
    RunKind
        Appropriate run kind based on stage modules:
        - ``ingest`` if only ingestion stages
        - ``graphs`` if only graph stages
        - ``analytics`` if only analytics stages
        - ``full`` for any combination
    """
    modules = {stage.module for stage in spec.stages}
    if modules == {"ingestion"}:
        return "ingest"
    if modules == {"graphs"}:
        return "graphs"
    if modules == {"analytics"}:
        return "analytics"
    return "full"


# -----------------------------------------------------------------------------
# Ingestion Planning
# -----------------------------------------------------------------------------


def _resolve_ingest_recipe(stage: PipelineStage) -> IngestRecipe:
    """Resolve the ingestion recipe for a stage name.

    Parameters
    ----------
    stage
        Pipeline stage with name specifying recipe flavor.

    Returns
    -------
    IngestRecipe
        Resolved recipe for execution.

    Raises
    ------
    ValueError
        If the recipe cannot be resolved.
    """
    from codeintel.ingestion.recipes.builtin import (
        FULL_PYTHON_RECIPE,
        INCREMENTAL_RECIPE,
        get_builtin_recipe,
    )

    name = stage.name
    if name == "builtin.default":
        return FULL_PYTHON_RECIPE
    if name == "builtin.incremental":
        return INCREMENTAL_RECIPE
    if name.startswith("builtin."):
        recipe_name = name.split(".", 1)[1]
        recipe = get_builtin_recipe(recipe_name)
        if recipe is None:
            message = f"Unknown builtin ingestion recipe: {recipe_name}"
            raise ValueError(message)
        return recipe

    # Try as direct recipe name
    recipe = get_builtin_recipe(name)
    if recipe is None:
        message = f"Unknown ingestion recipe: {name}"
        raise ValueError(message)
    return recipe


def _plan_ingestion_stage(  # noqa: PLR0913
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
) -> IngestionStagePlan:
    """Plan an ingestion stage execution.

    Parameters
    ----------
    stage
        Ingestion stage to plan.
    run_ctx
        Unified run context.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration.
    gateway
        Storage gateway for database access.
    tools
        Tools configuration.

    Returns
    -------
    IngestionStagePlan
        Fully resolved ingestion plan.
    """
    from codeintel.ingestion.infrastructure_utilities.source_scanner import (
        default_code_profile,
        default_config_profile,
    )
    from codeintel.ingestion.recipes.dsl import RecipeOptions
    from codeintel.ingestion.recipes.executor import RecipeExecutorContext

    recipe = _resolve_ingest_recipe(stage)

    code_profile = default_code_profile(snapshot.repo_root)
    config_profile = default_config_profile(snapshot.repo_root)

    context = RecipeExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
        code_profile=code_profile,
        config_profile=config_profile,
        tool_runner=None,
        tool_service=None,
        change_tracker=None,
        ingest_run_sink=None,
        run_context=run_ctx,
    )

    options = RecipeOptions()

    return IngestionStagePlan(
        stage=stage,
        recipe=recipe,
        options=options,
        context=context,
    )


# -----------------------------------------------------------------------------
# Graphs Planning
# -----------------------------------------------------------------------------


def _plan_graphs_stage(
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> GraphsStagePlan:
    """Plan a graphs stage execution.

    Parameters
    ----------
    stage
        Graphs stage to plan.
    run_ctx
        Unified run context.
    snapshot
        Repository snapshot reference.
    gateway
        Storage gateway for database access.

    Returns
    -------
    GraphsStagePlan
        Fully resolved graphs plan.
    """
    from codeintel.config.steps_graphs import GraphPluginPolicy
    from codeintel.graphs.runtime.executor import GraphExecutorContext
    from codeintel.graphs.runtime.planning import GraphPlanContext, plan_graph_plugin_run

    policy = GraphPluginPolicy(fail_fast=True, skip_on_unchanged=False)

    plan_ctx = GraphPlanContext(
        runtime_snapshot=snapshot,
        target=(snapshot.repo, snapshot.commit),
        policy=policy,
        prior_manifest=None,
    )

    # For builtin.full, use all default plugins (None means all)
    plugin_names: list[str] | None = None

    plan = plan_graph_plugin_run(
        plugin_names=plugin_names,
        context=plan_ctx,
    )

    exec_ctx = GraphExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        engine=None,
        catalog_provider=None,
        run_context=run_ctx,
    )

    return GraphsStagePlan(
        stage=stage,
        plan=plan,
        context=exec_ctx,
    )


# -----------------------------------------------------------------------------
# Analytics Planning
# -----------------------------------------------------------------------------


def _get_analytics_plugin_names() -> tuple[str, ...]:
    """Get the full list of analytics plugin names.

    Returns
    -------
    tuple[str, ...]
        Plugin names for the full analytics bundle.
    """
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

    return (
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


def _plan_analytics_stage(
    *,
    stage: PipelineStage,
    run_ctx: RunContext,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> AnalyticsStagePlan:
    """Plan an analytics stage execution.

    Parameters
    ----------
    stage
        Analytics stage to plan.
    run_ctx
        Unified run context.
    snapshot
        Repository snapshot reference.
    gateway
        Storage gateway for database access.

    Returns
    -------
    AnalyticsStagePlan
        Fully resolved analytics plan.
    """
    from codeintel.analytics.core.pipeline_bridge import (
        AnalyticsPlanRequest,
        AnalyticsRunContext,
        plan_analytics_plugin_run,
    )
    from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope

    plugin_names = _get_analytics_plugin_names()
    policy = GraphPluginPolicy(fail_fast=True, skip_on_unchanged=False)
    scope = GraphRunScope()

    request = AnalyticsPlanRequest(
        plugin_names=plugin_names,
        policy=policy,
        repo=snapshot.repo,
        commit=snapshot.commit,
        scope=scope,
        prior_manifest=None,
        cfg_options=None,
        runtime_options=None,
        run_id=run_ctx.run_id,
    )

    plan = plan_analytics_plugin_run(request)

    context = AnalyticsRunContext(
        gateway=gateway,
        graph_runtime=None,
        cfgs={},
        extra={},
        catalog_provider=None,
        snapshot=snapshot,
    )

    return AnalyticsStagePlan(
        stage=stage,
        request=request,
        plan=plan,
        context=context,
    )


# -----------------------------------------------------------------------------
# Main Planning Function
# -----------------------------------------------------------------------------


def build_pipeline_plan(  # noqa: PLR0913
    *,
    spec: PipelineSpec,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    trigger: TriggerKind = "cli",
    run_kind_override: RunKind | None = None,
) -> PipelinePlan:
    """Build a concrete execution plan for a pipeline specification.

    Translates a declarative PipelineSpec into a PipelinePlan with resolved
    recipes, plugin plans, and execution contexts for each stage.

    Parameters
    ----------
    spec
        Declarative pipeline specification.
    snapshot
        Repository snapshot to operate on.
    paths
        Build paths configuration.
    gateway
        Storage gateway for database access.
    tools
        Tools configuration.
    trigger
        How the run was triggered.
    run_kind_override
        If provided, use this RunKind instead of inferring from spec stages.
        Useful for operation prerequisite runs where the kind should be
        ``"op_prereqs"`` regardless of the spec.

    Returns
    -------
    PipelinePlan
        Fully resolved execution plan.

    Raises
    ------
    ValueError
        If a stage module is not recognized or recipe cannot be resolved.
    """
    run_kind = run_kind_override if run_kind_override is not None else _infer_run_kind(spec)
    run_ctx = new_run_context(
        snapshot=snapshot,
        kind=run_kind,
        trigger=trigger,
    )

    log.info(
        "pipeline.plan.build spec=%s run_id=%s kind=%s stages=%d",
        spec.id,
        run_ctx.run_id,
        run_kind,
        len(spec.stages),
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
            )
        elif stage.module == "graphs":
            graphs_plan = _plan_graphs_stage(
                stage=stage,
                run_ctx=run_ctx,
                snapshot=snapshot,
                gateway=gateway,
            )
        elif stage.module == "analytics":
            analytics_plan = _plan_analytics_stage(
                stage=stage,
                run_ctx=run_ctx,
                snapshot=snapshot,
                gateway=gateway,
            )
        else:
            message = f"Unknown stage module: {stage.module}"
            raise ValueError(message)

    return PipelinePlan(
        spec=spec,
        run_context=run_ctx,
        ingestion=ingestion_plan,
        graphs=graphs_plan,
        analytics=analytics_plan,
    )


__all__ = [
    "AnalyticsStagePlan",
    "GraphsStagePlan",
    "IngestionStagePlan",
    "PipelinePlan",
    "build_pipeline_plan",
]
