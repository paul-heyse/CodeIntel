"""Expected outputs utility for native Hamilton targets.

This module provides helper functions for generating expected DatasetRef and
ArtifactRef objects based on a target's OutputContract. This enables native
targets to declare their outputs upfront for downstream consumption.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from codeintel.build.hamilton.artifact_ref import ArtifactRef
from codeintel.build.hamilton.dataset_ref import DatasetRef

if TYPE_CHECKING:
    from codeintel.build.snapshot import SnapshotRef
    from codeintel.build.targets import OutputTarget


def expected_datasets(
    target: OutputTarget,
    snapshot: SnapshotRef,
) -> tuple[DatasetRef, ...]:
    """Generate expected DatasetRef objects for a target's output tables.

    Parameters
    ----------
    target
        Output target with contract defining table_keys.
    snapshot
        Snapshot identity (repo + commit) for lineage.

    Returns
    -------
    tuple[DatasetRef, ...]
        Tuple of DatasetRef objects, one per table_key in the contract.

    Examples
    --------
    >>> from codeintel.build.snapshot import SnapshotRef
    >>> from codeintel.build.registry import get_target_graph
    >>> graph = get_target_graph()
    >>> target = graph.get("function_metrics")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_datasets(target, snapshot)
    >>> len(refs)
    2
    >>> refs[0].table_key
    'analytics.function_metrics'
    """
    if not target.contract or not target.contract.table_keys:
        return ()

    return tuple(
        DatasetRef(
            table_key=table_key,
            repo=snapshot.repo,
            commit=snapshot.commit,
            row_count=None,  # Not known until materialized
        )
        for table_key in target.contract.table_keys
    )


def expected_artifacts(
    target: OutputTarget,
    snapshot: SnapshotRef,
    *,
    path_formatter: dict[str, str] | None = None,
) -> tuple[ArtifactRef, ...]:
    """Generate expected ArtifactRef objects for a target's output artifacts.

    Parameters
    ----------
    target
        Output target with contract defining artifacts.
    snapshot
        Snapshot identity (repo + commit) for lineage.
    path_formatter
        Optional dict for formatting path templates (e.g., {"build_dir": "/tmp/build"}).

    Returns
    -------
    tuple[ArtifactRef, ...]
        Tuple of ArtifactRef objects, one per artifact in the contract.

    Examples
    --------
    >>> from codeintel.build.snapshot import SnapshotRef
    >>> from codeintel.build.registry import get_target_graph
    >>> graph = get_target_graph()
    >>> target = graph.get("scip")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_artifacts(target, snapshot)
    >>> len(refs) > 0
    True
    """
    if not target.contract or not target.contract.artifacts:
        return ()

    formatter = path_formatter or {}

    refs: list[ArtifactRef] = []
    for artifact_spec in target.contract.artifacts:
        # Format path template if formatter provided
        path = None
        if artifact_spec.path_template and formatter:
            with suppress(KeyError):
                path = artifact_spec.path_template.format(**formatter)

        refs.append(
            ArtifactRef(
                name=artifact_spec.name,
                artifact_type="file",
                repo=snapshot.repo,
                commit=snapshot.commit,
                path=path,
                metadata={"description": artifact_spec.description or ""},
            )
        )

    return tuple(refs)


__all__ = [
    "expected_artifacts",
    "expected_datasets",
]
