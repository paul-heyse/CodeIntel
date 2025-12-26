"""Validate that build target contracts match the Hamilton DAG shape.

This module is executed by `tools.quality_report` to ensure that:

1) Every target in the canonical catalog has exactly one materialize node.
2) The Hamilton DAG exposes dataset/artifact outputs that match each target's contract.

The intent is to prevent "silent drift" where build metadata and the executable DAG diverge.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from codeintel.build.hamilton.introspect import (
    DerivedTargetOutputs,
    derive_target_outputs_from_savers,
)
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.hamilton.tags import NODE_TYPE_MATERIALIZE, TAG_NODE_TYPE, TAG_TARGET

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import TargetGraph


class TargetContractsError(RuntimeError):
    """Raised when Hamilton target contracts are invalid."""

    def __init__(self, *, issues: Iterable[str]) -> None:
        lines = "\n".join(f"- {issue}" for issue in issues)
        super().__init__(f"Target contract check failed:\n{lines}")


def _derive_materialize_targets(runtime: HamiltonRuntime) -> tuple[dict[str, str], list[str]]:
    """Return mapping of target_name -> node_name for materialize nodes.

    Parameters
    ----------
    runtime
        Hamilton runtime returned by :func:`codeintel.build.hamilton.driver_factory.build_driver`.

    Returns
    -------
    tuple[dict[str, str], list[str]]
        (target_to_node, issues) where issues includes duplicate-target materialize nodes.
    """
    nodes = getattr(getattr(runtime, "dr", None), "graph", None)
    nodes = getattr(nodes, "nodes", None)
    if not isinstance(nodes, dict):
        return {}, ["Hamilton runtime does not expose dr.graph.nodes dict"]

    issues: list[str] = []
    target_to_node: dict[str, str] = {}
    for node_name, node in nodes.items():
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get(TAG_NODE_TYPE) != NODE_TYPE_MATERIALIZE:
            continue
        target = tags.get(TAG_TARGET)
        if not isinstance(target, str) or not target:
            continue
        existing = target_to_node.get(target)
        if existing is not None:
            issues.append(
                f"Duplicate materialize nodes for target {target}: {existing}, {node_name}"
            )
            continue
        target_to_node[target] = node_name
    return target_to_node, issues


def _check_catalog_completeness(runtime: HamiltonRuntime) -> list[str]:
    issues: list[str] = []
    target_to_node, node_issues = _derive_materialize_targets(runtime)
    issues.extend(node_issues)

    graph = getattr(runtime, "graph", None)
    all_targets = getattr(graph, "all_targets", None)
    if not isinstance(all_targets, tuple):
        issues.append("Hamilton runtime does not expose graph.all_targets tuple")
        return issues

    catalog_targets = {t.name for t in all_targets if hasattr(t, "name")}
    dag_targets = set(target_to_node)

    missing = sorted(catalog_targets - dag_targets)
    extra = sorted(dag_targets - catalog_targets)

    if missing:
        issues.append(f"Missing materialize nodes for targets: {missing}")
    if extra:
        issues.append(f"Materialize nodes exist for unknown targets: {extra}")

    return issues


def _check_contract_outputs(graph: TargetGraph, outputs: DerivedTargetOutputs) -> list[str]:
    issues: list[str] = []
    all_targets = graph.all_targets
    datasets_by_target = outputs.datasets_by_target
    artifacts_by_target = outputs.artifacts_by_target
    templates_by_target = outputs.artifact_templates_by_target

    for target in all_targets:
        name = getattr(target, "name", None)
        contract = getattr(target, "contract", None)
        if not isinstance(name, str) or contract is None:
            continue

        expected_tables = tuple(sorted(getattr(contract, "table_keys", ())))
        expected_artifacts = tuple(sorted(getattr(contract, "artifact_names", ())))
        expected_templates = {
            artifact.name: artifact.path_template for artifact in getattr(contract, "artifacts", ())
        }

        observed_tables = tuple(sorted(datasets_by_target.get(name, ())))
        observed_artifacts = tuple(sorted(artifacts_by_target.get(name, ())))
        observed_templates = templates_by_target.get(name, {})

        if expected_tables != observed_tables:
            issues.append(
                "Target contract table_keys differ from DAG outputs "
                f"for {name}: expected={expected_tables} observed={observed_tables}"
            )
        if expected_artifacts != observed_artifacts:
            issues.append(
                "Target contract artifact_names differ from DAG outputs "
                f"for {name}: expected={expected_artifacts} observed={observed_artifacts}"
            )
        if expected_templates != observed_templates:
            issues.append(
                "Target contract artifact templates differ from DAG outputs "
                f"for {name}: expected={expected_templates} observed={observed_templates}"
            )

    return issues


def main() -> int:
    """Run target contract checks.

    Returns
    -------
    int
        Process exit code (0 = success, 1 = failure).
    """
    service = get_target_metadata_service()
    runtime = service.system.runtime
    graph = service.system.graph

    issues: list[str] = []
    issues.extend(_check_catalog_completeness(runtime))
    derived = derive_target_outputs_from_savers(service.system.runtime)
    issues.extend(_check_contract_outputs(graph, derived))

    if issues:
        err = TargetContractsError(issues=issues)
        sys.stderr.write(f"{err}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
