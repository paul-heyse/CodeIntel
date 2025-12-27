"""Hamilton build planner wrapper for DAG-native plan execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.planning.plan_targets import CI_PLAN_TARGET_NAME
from codeintel.build.planning.model import BuildPlan, PlanRequest
from codeintel.runtime.compose import compose_runtime, set_execution_active
from codeintel.runtime.inputs import ExecutionInputs
from codeintel.runtime.runtime_bundle import RuntimeBundle


def compute_plan(
    *,
    env: BuildEnv,
    plan_request: PlanRequest,
    runtime: RuntimeBundle | None = None,
    config: Mapping[str, Any] | None = None,
    materialize: bool = True,
) -> BuildPlan:
    """Execute the planning DAG and return the BuildPlan object.

    Parameters
    ----------
    env
        Build environment to inject into the planning DAG.
    plan_request
        Plan request parameters for the planning DAG.
    runtime
        Optional pre-composed runtime bundle to reuse.
    config
        Optional Hamilton config when composing a runtime.
    materialize
        When True, execute the ci_plan target to emit plan artifacts.

    Returns
    -------
    BuildPlan
        DAG-native build plan output.
    """
    resolved_runtime = runtime or _compose_planning_runtime(env=env, config=config)
    final_vars = _plan_final_vars(runtime=resolved_runtime, materialize=materialize)
    inputs = ExecutionInputs(
        env=env,
        catalog=resolved_runtime.catalog,
        tag_query=resolved_runtime.tag_query,
        cache_index=resolved_runtime.cache_index,
        cache_key_resolver=resolved_runtime.cache_key_resolver,
        schema_index=resolved_runtime.schema_index,
        semantic_registry=resolved_runtime.semantic_registry,
        runtime_fingerprint=resolved_runtime.fingerprint,
        plan_request=plan_request,
    )
    outputs = _execute_plan(runtime=resolved_runtime, inputs=inputs, final_vars=final_vars)
    plan = outputs.get("plan")
    if not isinstance(plan, BuildPlan):
        msg = "Planning DAG did not return a BuildPlan"
        raise TypeError(msg)
    return plan


def _compose_planning_runtime(
    *,
    env: BuildEnv,
    config: Mapping[str, Any] | None,
) -> RuntimeBundle:
    resolved_config = _planning_config(env=env, config=config)
    return compose_runtime(env=env, config=resolved_config).bundle


def _planning_config(
    *,
    env: BuildEnv,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved = dict(config or {})
    if env.profile and "profile" not in resolved:
        resolved["profile"] = env.profile
    resolved.update(env.variants.as_hamilton_config())
    resolved["variant_fingerprint"] = env.variants.variant_fingerprint
    return resolved


def _plan_final_vars(*, runtime: RuntimeBundle, materialize: bool) -> list[str]:
    final_vars = ["plan"]
    if materialize:
        target_node = runtime.catalog.target_nodes.get(CI_PLAN_TARGET_NAME)
        if target_node is None:
            msg = "Planning target node is missing from the catalog"
            raise ValueError(msg)
        final_vars.append(target_node)
    return final_vars


def _execute_plan(
    *,
    runtime: RuntimeBundle,
    inputs: ExecutionInputs,
    final_vars: Sequence[str],
) -> dict[str, object]:
    input_mapping = _execution_input_mapping(inputs)
    set_execution_active(active=True)
    try:
        return runtime.driver.execute(list(final_vars), inputs=input_mapping)
    finally:
        set_execution_active(active=False)


def _execution_input_mapping(inputs: ExecutionInputs) -> dict[str, object]:
    mapping: dict[str, object] = {
        "env": inputs.env,
        "catalog": inputs.catalog,
    }
    optional: dict[str, object | None] = {
        "tag_query": inputs.tag_query,
        "cache_index": inputs.cache_index,
        "cache_key_resolver": inputs.cache_key_resolver,
        "schema_index": inputs.schema_index,
        "semantic_registry": inputs.semantic_registry,
        "runtime_fingerprint": inputs.runtime_fingerprint,
        "plan_request": inputs.plan_request,
    }
    mapping.update({key: value for key, value in optional.items() if value is not None})
    return mapping


__all__ = ["compute_plan"]
