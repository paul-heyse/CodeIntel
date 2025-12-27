"""Preflight planning nodes for blocked targets."""

from __future__ import annotations

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.planning.preflight import (
    PreflightIssue,
    classify_missing_inputs,
    missing_input_issues,
    optional_inputs_for_targets,
)
from codeintel.storage.helpers.table_key import split_table_key


def preflight_issues(
    env: BuildEnv,
    catalog: DagCatalog,
    plan_target_closure: tuple[str, ...],
) -> tuple[PreflightIssue, ...]:
    """Return preflight issues for the requested closure.

    Returns
    -------
    tuple[PreflightIssue, ...]
        Preflight issues for missing inputs.
    """
    produced_table_keys = {
        output.key
        for target in plan_target_closure
        for output in catalog.table_outputs_by_target.get(target, ())
    }
    issues: list[PreflightIssue] = []

    for target in plan_target_closure:
        surface = catalog.io_surfaces.get(target)
        if surface is None:
            continue
        missing_inputs: set[str] = set()
        optional_inputs = optional_inputs_for_targets(target)

        for read in surface.reads:
            table_key = read.table_key
            if table_key in produced_table_keys:
                continue
            if not _table_key_exists(env, table_key):
                missing_inputs.add(table_key)

        if not missing_inputs:
            continue

        missing_required, missing_optional = classify_missing_inputs(
            optional_inputs=optional_inputs,
            missing=missing_inputs,
        )
        issues.extend(
            missing_input_issues(
                missing_required=missing_required,
                missing_optional=missing_optional,
                target=target,
            )
        )

    return tuple(issues)


def preflight_block_map(
    catalog: DagCatalog,
    plan_target_closure: tuple[str, ...],
    preflight_issues: tuple[PreflightIssue, ...],
) -> dict[str, tuple[str, ...]]:
    """Map blocked targets to preflight reasons.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of blocked targets to reason strings.
    """
    issues_by_target: dict[str, list[PreflightIssue]] = {}
    for issue in preflight_issues:
        if issue.target is None:
            continue
        issues_by_target.setdefault(issue.target, []).append(issue)

    roots = set(issues_by_target)
    blocked_targets = _blocked_targets(catalog, roots)
    block_map: dict[str, tuple[str, ...]] = {}

    for target in plan_target_closure:
        if target not in blocked_targets:
            continue
        issues = issues_by_target.get(target)
        if issues:
            reasons = tuple(issue.to_block_reason() for issue in issues)
        else:
            roots_list = ", ".join(sorted(roots))
            reasons = (f"missing_prerequisites:{roots_list}",)
        block_map[target] = reasons

    return block_map


def _blocked_targets(catalog: DagCatalog, roots: set[str]) -> set[str]:
    blocked = set(roots)
    queue = list(roots)
    while queue:
        current = queue.pop()
        for dependent in catalog.dependents_of(current):
            if dependent in blocked:
                continue
            blocked.add(dependent)
            queue.append(dependent)
    return blocked


def _table_key_exists(env: BuildEnv, table_key: str) -> bool:
    schema, table = split_table_key(table_key)
    return env.gateway.policy.table_exists(schema=schema, table=table)


__all__ = ["preflight_block_map", "preflight_issues"]
