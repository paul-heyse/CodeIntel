"""Support node specification derived from the DAG catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.external_inputs import load_external_inputs_allowlist
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataset_node,
    path_node,
    query_node,
)
from codeintel.core.hamilton import tags as ht
from codeintel.core.imports.lazy import lazy_import
from codeintel.core.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class SupportDatasetSpec:
    """Dataset entry used to generate support nodes."""

    table_key: str
    producer_target: str | None
    domain: str
    data_node: str | None
    allowlisted: bool = False


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
            if spec.data_node is not None:
                _require_identifier(spec.data_node, label="data_node")
            if spec.producer_target is None:
                continue
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


@dataclass(frozen=True, slots=True)
class SupportSpecOptions:
    """Options for building support node specifications."""

    include_dataset_nodes: bool = True
    include_loader_nodes: bool = True
    include_artifact_nodes: bool = True
    include_artifact_path_nodes: bool = True


def support_spec_from_catalog(
    catalog: DagCatalog,
    *,
    options: SupportSpecOptions | None = None,
    repo_root: Path | None = None,
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
    resolved = options or SupportSpecOptions()
    resolved_root = repo_root or Path.cwd()
    allowlist = load_external_inputs_allowlist(repo_root=resolved_root)
    allowlisted_keys = allowlist.table_keys()
    datasets: list[SupportDatasetSpec] = []
    artifacts: list[SupportArtifactSpec] = []

    for table_key, output in sorted(catalog.table_outputs.items()):
        target = catalog.targets.get(output.producer_target)
        if target is None:
            msg = f"Unknown producer target for table output {table_key}"
            raise ValueError(msg)
        data_node = catalog.table_data_nodes.get(table_key)
        if data_node is None:
            msg = f"Missing data node for {table_key}"
            raise ValueError(msg)
        datasets.append(
            SupportDatasetSpec(
                table_key=table_key,
                producer_target=output.producer_target,
                domain=target.module,
                data_node=data_node,
                allowlisted=table_key in allowlisted_keys,
            )
        )

    dataset_keys = {spec.table_key for spec in datasets}
    for table_key in _view_base_table_keys():
        if table_key in dataset_keys:
            continue
        data_node_entry = _dataset_node_for_table(catalog=catalog, table_key=table_key)
        if data_node_entry is not None:
            data_node, producer_target, domain = data_node_entry
            datasets.append(
                SupportDatasetSpec(
                    table_key=table_key,
                    producer_target=producer_target,
                    domain=domain,
                    data_node=data_node,
                    allowlisted=False,
                )
            )
            dataset_keys.add(table_key)
            continue
        allowlisted = table_key in allowlisted_keys
        if not allowlisted:
            msg = f"External input {table_key} is not allowlisted"
            raise ValueError(msg)
        datasets.append(
            SupportDatasetSpec(
                table_key=table_key,
                producer_target=None,
                domain=_domain_from_table_key(table_key),
                data_node=None,
                allowlisted=allowlisted,
            )
        )
        dataset_keys.add(table_key)

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
        include_dataset_nodes=resolved.include_dataset_nodes,
        include_loader_nodes=resolved.include_loader_nodes,
        include_artifact_nodes=resolved.include_artifact_nodes,
        include_artifact_path_nodes=resolved.include_artifact_path_nodes,
    )
    spec.validate(catalog=catalog)
    return spec


def _dataset_dicts(specs: Iterable[SupportDatasetSpec]) -> Iterable[dict[str, object]]:
    for spec in specs:
        yield {
            "table_key": spec.table_key,
            "producer_target": spec.producer_target,
            "domain": spec.domain,
            "data_node": spec.data_node,
            "allowlisted": spec.allowlisted,
        }


def _dataset_node_for_table(
    *,
    catalog: DagCatalog,
    table_key: str,
) -> tuple[str, str | None, str] | None:
    candidates = [
        node
        for node in catalog.nodes.values()
        if node.tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_DATASET
        and node.tags.get(ht.TAG_TABLE_KEY) == table_key
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        msg = f"Multiple dataset nodes found for {table_key}"
        raise ValueError(msg)
    node = candidates[0]
    target = node.tags.get(ht.TAG_TARGET)
    domain = node.tags.get(ht.TAG_DOMAIN)
    resolved_domain = (
        domain if isinstance(domain, str) and domain else _domain_from_table_key(table_key)
    )
    resolved_target = target if isinstance(target, str) and target else None
    return node.name, resolved_target, resolved_domain


def _view_base_table_keys() -> tuple[str, ...]:
    try:
        module = lazy_import("codeintel.build.hamilton.native.views.view_outputs")
    except ImportError:
        return ()
    plans = module.view_plan_map()
    if not plans:
        return ()
    view_keys = set(plans)
    base_keys = {
        dep for plan in plans.values() for dep in plan.dependencies if dep not in view_keys
    }
    return tuple(sorted(base_keys))


def _domain_from_table_key(table_key: str) -> str:
    try:
        schema, _ = split_table_key(table_key)
    except ValueError:
        return "external"
    if schema:
        return schema
    return "external"


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
    "SupportSpecOptions",
    "support_spec_from_catalog",
]
