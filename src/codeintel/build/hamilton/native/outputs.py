"""Expected outputs utility for native Hamilton targets.

This module provides helper functions for generating expected DatasetRef and
ArtifactRef objects based on the DAG-derived output inventory.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.introspect import derive_target_outputs_from_savers
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.materializers.path_templates import format_path_template
from codeintel.core.imports.lazy import lazy_getattr

if TYPE_CHECKING:
    from codeintel.build.hamilton.introspect import DerivedTargetOutputs
    from codeintel.build.hamilton.runtime import HamiltonRuntime
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef


@lru_cache(maxsize=1)
def _derived_outputs() -> DerivedTargetOutputs:
    build_driver = cast(
        "Callable[..., HamiltonRuntime]",
        lazy_getattr("codeintel.build.hamilton.driver_factory", "build_driver"),
    )
    runtime = build_driver()
    return derive_target_outputs_from_savers(runtime)


def _resolve_outputs(outputs: DerivedTargetOutputs | None) -> DerivedTargetOutputs:
    return outputs or _derived_outputs()


def expected_table_keys_for_target(
    target_name: str,
    *,
    outputs: DerivedTargetOutputs | None = None,
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
    resolved_outputs = _resolve_outputs(outputs)
    return tuple(resolved_outputs.datasets_by_target.get(target_name, ()))


def expected_artifact_names_for_target(
    target_name: str,
    *,
    outputs: DerivedTargetOutputs | None = None,
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
    resolved_outputs = _resolve_outputs(outputs)
    return tuple(resolved_outputs.artifacts_by_target.get(target_name, ()))


def artifact_templates_for_target(
    target_name: str,
    *,
    outputs: DerivedTargetOutputs | None = None,
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
    resolved_outputs = _resolve_outputs(outputs)
    return dict(resolved_outputs.artifact_templates_by_target.get(target_name, {}))


def expected_datasets(
    target: OutputTarget,
    snapshot: SnapshotRef,
    *,
    outputs: DerivedTargetOutputs | None = None,
) -> tuple[DatasetRef, ...]:
    """Generate expected DatasetRef objects for a target's output tables.

    Parameters
    ----------
    target
        Output target with contract defining table_keys.
    snapshot
        Snapshot identity (repo + commit) for lineage.
    outputs
        Optional DAG-derived outputs mapping.

    Returns
    -------
    tuple[DatasetRef, ...]
        Tuple of DatasetRef objects, one per table_key in the contract.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from codeintel.build.target_metadata import get_target_metadata_service
    >>> graph = get_target_metadata_service().system.graph
    >>> target = graph.get("function_metrics")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_datasets(target, snapshot)
    >>> len(refs) > 0
    True
    >>> refs[0].table_key
    'analytics.function_metrics'
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
    target: OutputTarget,
    snapshot: SnapshotRef,
    *,
    outputs: DerivedTargetOutputs | None = None,
    path_formatter: dict[str, str] | None = None,
) -> tuple[ArtifactRef, ...]:
    """Generate expected ArtifactRef objects for a target's output artifacts.

    Parameters
    ----------
    target
        Output target used for identity only (artifacts are DAG-derived).
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
    >>> from codeintel.build.target_metadata import get_target_metadata_service
    >>> graph = get_target_metadata_service().system.graph
    >>> target = graph.get("scip")
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> refs = expected_artifacts(target, snapshot)
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
