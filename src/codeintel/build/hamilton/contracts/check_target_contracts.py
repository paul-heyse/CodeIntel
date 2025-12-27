"""Validate that catalog outputs match the Hamilton DAG shape.

This module is executed by `tools.quality_report` to ensure that:

1) Every target in the canonical catalog has exactly one materialize node.
2) The DAG-derived output inventory is internally consistent.

The intent is to prevent "silent drift" where build metadata and the executable DAG diverge.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.hamilton.tags import NODE_TYPE_MATERIALIZE, TAG_NODE_TYPE, TAG_TARGET

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime


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

    catalog_targets = set(runtime.catalog.targets)
    dag_targets = set(target_to_node)

    missing = sorted(catalog_targets - dag_targets)
    extra = sorted(dag_targets - catalog_targets)

    if missing:
        issues.append(f"Missing materialize nodes for targets: {missing}")
    if extra:
        issues.append(f"Materialize nodes exist for unknown targets: {extra}")

    return issues


def _check_catalog_outputs(catalog: DagCatalog) -> list[str]:
    issues: list[str] = []
    for table_key, output in catalog.table_outputs.items():
        if output.producer_target not in catalog.targets:
            issues.append(
                "Catalog table output has unknown producer target: "
                f"table_key={table_key} target={output.producer_target}"
            )
            continue
        if output not in catalog.table_outputs_by_target.get(output.producer_target, ()):
            issues.append(
                "Catalog table output missing from per-target index: "
                f"table_key={table_key} target={output.producer_target}"
            )
    for artifact_name, output in catalog.artifact_outputs.items():
        if output.producer_target not in catalog.targets:
            issues.append(
                "Catalog artifact output has unknown producer target: "
                f"artifact_name={artifact_name} target={output.producer_target}"
            )
            continue
        if output not in catalog.artifact_outputs_by_target.get(output.producer_target, ()):
            issues.append(
                "Catalog artifact output missing from per-target index: "
                f"artifact_name={artifact_name} target={output.producer_target}"
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

    issues: list[str] = []
    issues.extend(_check_catalog_completeness(runtime))
    issues.extend(_check_catalog_outputs(runtime.catalog))

    if issues:
        err = TargetContractsError(issues=issues)
        sys.stderr.write(f"{err}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
