"""Expected outputs utility for native Hamilton targets.

This module provides helper functions for generating expected DatasetRef and
ArtifactRef objects based on the DAG-derived output inventory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.materializers.path_templates import format_path_template

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import TargetDescriptor
    from codeintel.config.primitives import SnapshotRef


def expected_table_keys_for_target(
    target_name: str,
    *,
    outputs: DagCatalog,
) -> tuple[str, ...]:
    """Return expected table keys for a target from DAG saver tags.

    Parameters
    ----------
    target_name
        Target name to resolve.
    outputs
        Optional DAG-derived outputs mapping.

    Returns
    -------
    tuple[str, ...]
        Table keys expected to be written by the target.
    """
    return tuple(output.key for output in outputs.table_outputs_by_target.get(target_name, ()))


def expected_artifact_names_for_target(
    target_name: str,
    *,
    outputs: DagCatalog,
) -> tuple[str, ...]:
    """Return expected artifact names for a target from DAG saver tags.

    Parameters
    ----------
    target_name
        Target name to resolve.
    outputs
        Optional DAG-derived outputs mapping.

    Returns
    -------
    tuple[str, ...]
        Artifact names expected to be written by the target.
    """
    return tuple(output.key for output in outputs.artifact_outputs_by_target.get(target_name, ()))


def artifact_templates_for_target(
    target_name: str,
    *,
    outputs: DagCatalog,
) -> dict[str, str]:
    """Return artifact path templates for a target from DAG saver tags.

    Parameters
    ----------
    target_name
        Target name to resolve.
    outputs
        Optional DAG-derived outputs mapping.

    Returns
    -------
    dict[str, str]
        Mapping of artifact name to path template.
    """
    templates = outputs.artifact_outputs_by_target.get(target_name, ())
    return {
        output.key: output.artifact_path_template
        for output in templates
        if output.artifact_path_template is not None
    }


def expected_datasets(
    target: TargetDescriptor,
    snapshot: SnapshotRef,
    *,
    outputs: DagCatalog,
) -> tuple[DatasetRef, ...]:
    """Generate expected DatasetRef objects for a target's output tables.

    Parameters
    ----------
    target
        Target descriptor used for identity; table keys are DAG-derived.
    snapshot
        Snapshot identity (repo + commit) for lineage.
    outputs
        Optional DAG-derived outputs mapping.

    Returns
    -------
    tuple[DatasetRef, ...]
        Tuple of DatasetRef objects, one per DAG-declared table key.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> catalog = runtime.catalog
    >>> target = catalog.get("function_types")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_datasets(target, snapshot, outputs=catalog)
    >>> len(refs) > 0
    True
    >>> refs[0].table_key
    'analytics.function_types'
    """
    table_keys = expected_table_keys_for_target(target.name, outputs=outputs)
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
    target: TargetDescriptor,
    snapshot: SnapshotRef,
    *,
    outputs: DagCatalog,
    path_formatter: dict[str, str] | None = None,
) -> tuple[ArtifactRef, ...]:
    """Generate expected ArtifactRef objects for a target's output artifacts.

    Parameters
    ----------
    target
        Target descriptor used for identity only (artifacts are DAG-derived).
    snapshot
        Snapshot identity (repo + commit) for lineage.
    outputs
        Optional DAG-derived outputs mapping.
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
    >>> catalog = runtime.catalog
    >>> target = catalog.get("scip")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_artifacts(target, snapshot, outputs=catalog)
    >>> len(refs) > 0
    True
    """
    allowed = set(expected_artifact_names_for_target(target.name, outputs=outputs))
    if not allowed:
        return ()
    templates = artifact_templates_for_target(target.name, outputs=outputs)
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
    "artifact_templates_for_target",
    "expected_artifact_names_for_target",
    "expected_artifacts",
    "expected_datasets",
    "expected_table_keys_for_target",
]
