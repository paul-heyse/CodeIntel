"""Expected outputs utility for native Hamilton targets.

This module provides helper functions for generating expected DatasetRef and
ArtifactRef objects based on the DAG-derived output inventory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.materializers.path_templates import format_path_template
from codeintel.build.target_inventory import get_output_inventory

if TYPE_CHECKING:
    from codeintel.build.output_inventory import OutputInventory
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef


def expected_datasets(
    target: OutputTarget,
    snapshot: SnapshotRef,
    *,
    output_inventory: OutputInventory | None = None,
) -> tuple[DatasetRef, ...]:
    """Generate expected DatasetRef objects for a target's output tables.

    Parameters
    ----------
    target
        Output target with contract defining table_keys.
    snapshot
        Snapshot identity (repo + commit) for lineage.
    output_inventory
        Optional output inventory providing canonical table keys per target.

    Returns
    -------
    tuple[DatasetRef, ...]
        Tuple of DatasetRef objects, one per table_key in the contract.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from codeintel.build.target_catalog import load_target_specs
    >>> target = next(t for t in load_target_specs() if t.name == "function_metrics")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_datasets(target, snapshot)
    >>> len(refs)
    2
    >>> refs[0].table_key
    'analytics.function_metrics'
    """
    resolved_inventory = output_inventory or get_output_inventory()
    table_keys = resolved_inventory.datasets_for(target.name)
    if not table_keys:
        return ()

    return tuple(
        DatasetRef(
            table_key=table_key,
            repo=snapshot.repo,
            commit=snapshot.commit,
            row_count=None,
        )
        for table_key in table_keys
    )


def expected_artifacts(
    target: OutputTarget,
    snapshot: SnapshotRef,
    *,
    output_inventory: OutputInventory | None = None,
    path_formatter: dict[str, str] | None = None,
) -> tuple[ArtifactRef, ...]:
    """Generate expected ArtifactRef objects for a target's output artifacts.

    Parameters
    ----------
    target
        Output target used for identity only (artifacts are DAG-derived).
    snapshot
        Snapshot identity (repo + commit) for lineage.
    output_inventory
        Optional output inventory providing canonical artifact names per target.
    path_formatter
        Optional dict for formatting path templates (e.g., {"build_dir": "/tmp/build"}).

    Returns
    -------
    tuple[ArtifactRef, ...]
        Tuple of ArtifactRef objects, one per DAG-derived artifact.

    Raises
    ------
    ValueError
        If required artifact templates are missing from the output inventory.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from codeintel.build.target_catalog import load_target_specs
    >>> target = next(t for t in load_target_specs() if t.name == "scip")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_artifacts(target, snapshot)
    >>> len(refs) > 0
    True
    """
    resolved_inventory = output_inventory or get_output_inventory()
    allowed = set(resolved_inventory.artifacts_for(target.name))
    if not allowed:
        return ()
    templates = resolved_inventory.artifact_templates_for(target.name)
    if not templates:
        msg = f"Missing artifact templates for target: {target.name}"
        raise ValueError(msg)

    formatter = path_formatter or {}

    refs: list[ArtifactRef] = []
    for artifact_name in sorted(allowed):
        template = templates.get(artifact_name)
        if not template:
            msg = f"Missing path template for artifact {target.name}.{artifact_name}"
            raise ValueError(msg)
        path = None
        if template and formatter:
            path = format_path_template(template, formatter=formatter)

        refs.append(
            ArtifactRef(
                name=artifact_name,
                artifact_type="file",
                repo=snapshot.repo,
                commit=snapshot.commit,
                path=path,
                metadata={"description": ""},
            )
        )

    return tuple(refs)


__all__ = [
    "expected_artifacts",
    "expected_datasets",
]
