"""Support node specification derived from the DAG catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataset_node,
    path_node,
    query_node,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class SupportDatasetSpec:
    """Dataset entry used to generate support nodes."""

    table_key: str
    producer_target: str
    domain: str


@dataclass(frozen=True, slots=True)
class SupportArtifactSpec:
    """Artifact entry used to generate support nodes."""

    artifact_name: str
    producer_target: str
    domain: str


@dataclass(frozen=True, slots=True)
class SupportNodeSpec:
    """Specification for support node generation."""

    datasets: tuple[SupportDatasetSpec, ...] = ()
    artifacts: tuple[SupportArtifactSpec, ...] = ()
    include_dataset_nodes: bool = True
    include_loader_nodes: bool = True
    include_artifact_nodes: bool = True
    include_artifact_path_nodes: bool = True

    def validate(self, *, catalog: DagCatalog | None = None) -> None:
        """Validate specification invariants.

        Raises
        ------
        ValueError
            If table keys or artifact names are duplicated or invalid.
        """
        _require_unique(
            [spec.table_key for spec in self.datasets],
            label="table_key",
        )
        _require_unique(
            [spec.artifact_name for spec in self.artifacts],
            label="artifact_name",
        )

        for spec in self.datasets:
            _require_identifier(dataset_node(spec.table_key), label="dataset_node")
            _require_identifier(query_node(spec.table_key), label="query_node")
            if catalog is not None and spec.producer_target not in catalog.targets:
                msg = f"Unknown producer target in dataset spec: {spec.producer_target}"
                raise ValueError(msg)

        for spec in self.artifacts:
            _require_identifier(artifact_node(spec.artifact_name), label="artifact_node")
            _require_identifier(path_node(spec.artifact_name), label="path_node")
            if catalog is not None and spec.producer_target not in catalog.targets:
                msg = f"Unknown producer target in artifact spec: {spec.producer_target}"
                raise ValueError(msg)

    def to_hamilton_config(self) -> dict[str, object]:
        """Return flat config keys for resolve_from_config usage.

        Returns
        -------
        dict[str, object]
            Hamilton config mapping for support node expansion.
        """
        return {
            "ci_support_datasets": tuple(_dataset_dicts(self.datasets)),
            "ci_support_artifacts": tuple(_artifact_dicts(self.artifacts)),
            "ci_support_include_dataset_nodes": self.include_dataset_nodes,
            "ci_support_include_loader_nodes": self.include_loader_nodes,
            "ci_support_include_artifact_nodes": self.include_artifact_nodes,
            "ci_support_include_artifact_path_nodes": self.include_artifact_path_nodes,
        }


def support_spec_from_catalog(
    catalog: DagCatalog,
    *,
    include_dataset_nodes: bool = True,
    include_loader_nodes: bool = True,
    include_artifact_nodes: bool = True,
    include_artifact_path_nodes: bool = True,
) -> SupportNodeSpec:
    """Build a support node spec from the catalog's contract outputs.

    Returns
    -------
    SupportNodeSpec
        Support node specification derived from the catalog.

    Raises
    ------
    ValueError
        If the catalog outputs reference unknown targets.
    """
    datasets: list[SupportDatasetSpec] = []
    artifacts: list[SupportArtifactSpec] = []

    for table_key, output in sorted(catalog.table_outputs.items()):
        target = catalog.targets.get(output.producer_target)
        if target is None:
            msg = f"Unknown producer target for table output {table_key}"
            raise ValueError(msg)
        datasets.append(
            SupportDatasetSpec(
                table_key=table_key,
                producer_target=output.producer_target,
                domain=target.module,
            )
        )

    for artifact_name, output in sorted(catalog.artifact_outputs.items()):
        target = catalog.targets.get(output.producer_target)
        if target is None:
            msg = f"Unknown producer target for artifact output {artifact_name}"
            raise ValueError(msg)
        artifacts.append(
            SupportArtifactSpec(
                artifact_name=artifact_name,
                producer_target=output.producer_target,
                domain=target.module,
            )
        )

    spec = SupportNodeSpec(
        datasets=tuple(datasets),
        artifacts=tuple(artifacts),
        include_dataset_nodes=include_dataset_nodes,
        include_loader_nodes=include_loader_nodes,
        include_artifact_nodes=include_artifact_nodes,
        include_artifact_path_nodes=include_artifact_path_nodes,
    )
    spec.validate(catalog=catalog)
    return spec


def _dataset_dicts(specs: Iterable[SupportDatasetSpec]) -> Iterable[dict[str, str]]:
    for spec in specs:
        yield {
            "table_key": spec.table_key,
            "producer_target": spec.producer_target,
            "domain": spec.domain,
        }


def _artifact_dicts(specs: Iterable[SupportArtifactSpec]) -> Iterable[dict[str, str]]:
    for spec in specs:
        yield {
            "artifact_name": spec.artifact_name,
            "producer_target": spec.producer_target,
            "domain": spec.domain,
        }


def _require_unique(values: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            msg = f"Duplicate {label} in support spec: {value}"
            raise ValueError(msg)
        seen.add(value)


def _require_identifier(value: str, *, label: str) -> None:
    if not value.isidentifier():
        msg = f"{label} is not a valid identifier: {value}"
        raise ValueError(msg)


__all__ = [
    "SupportArtifactSpec",
    "SupportDatasetSpec",
    "SupportNodeSpec",
    "support_spec_from_catalog",
]
