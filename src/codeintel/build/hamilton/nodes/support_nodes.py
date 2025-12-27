"""Static Hamilton support nodes expanded from config."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import parameterize, resolve_from_config, source, value

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.io.duckdb_relation_adapter import load_dataset_relation
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataset_node,
    path_node,
    query_node,
    target_node,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tag_spec import TagKey, TagSpec, TagValue
from codeintel.storage.gateway import DuckDBRelation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hamilton.function_modifiers.base import NodeTransformLifecycle
    from hamilton.function_modifiers.dependencies import ParametrizedDependency
    from hamilton.node import Node


class _ParameterizeWithTags(parameterize):
    def __init__(
        self,
        *,
        tags_by_output: Mapping[str, Mapping[str, str]],
        **parametrization: dict[str, ParametrizedDependency]
        | tuple[dict[str, ParametrizedDependency], str],
    ) -> None:
        super().__init__(**parametrization)
        self._tags_by_output = tags_by_output

    def expand_node(
        self,
        node_: Node,
        config: dict[str, Any],
        fn: Callable[..., object],
    ) -> Collection[Node]:
        nodes = super().expand_node(node_, config, fn)
        updated: list[Node] = []
        for expanded in nodes:
            tags = self._tags_by_output.get(expanded.name)
            if not tags:
                updated.append(expanded)
                continue
            node_tags = expanded.tags.copy()
            node_tags.update(tags)
            updated.append(expanded.copy_with(tags=node_tags))
        return updated


def _normalize_tags(tags: Mapping[TagKey, TagValue]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, tag_value in tags.items():
        if isinstance(tag_value, str):
            normalized[str(key)] = tag_value
            continue
        msg = f"Support node tag {key} must be a string value"
        raise ValueError(msg)
    return normalized


def _decorate_dataset_nodes(
    ci_support_datasets: Sequence[Mapping[str, str]] | None = None,
    *,
    ci_support_include_dataset_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_dataset_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_datasets:
        return _ParameterizeWithTags(tags_by_output={})

    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in ci_support_datasets:
        table_key = spec["table_key"]
        producer_target = spec["producer_target"]
        domain = spec["domain"]
        node_name = dataset_node(table_key)
        mapping[node_name] = {
            "record": source(target_node(producer_target)),
            "table_key": value(table_key),
            "producer_target": value(producer_target),
        }
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_dataset(
                domain=domain,
                target=producer_target,
                table_key=table_key,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_dataset_nodes)
def dataset_ref(
    record: TargetRunRecord,
    table_key: str,
    producer_target: str,
) -> DatasetRef:
    """Return DatasetRef for a target output.

    Returns
    -------
    DatasetRef
        Dataset reference for the requested table.

    Raises
    ------
    ValueError
        If the dataset reference is missing for the producer target.
    """
    ds = record.get_dataset(table_key)
    if ds is None:
        msg = f"Missing DatasetRef for {table_key} from {producer_target}"
        raise ValueError(msg)
    if isinstance(ds, DatasetRef):
        return ds
    return DatasetRef(
        table_key=ds.table_key,
        repo=ds.repo,
        commit=ds.commit,
        row_count=ds.row_count,
        source_target=producer_target,
    )


def _decorate_query_nodes(
    ci_support_datasets: Sequence[Mapping[str, str]] | None = None,
    *,
    ci_support_include_loader_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_loader_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_datasets:
        return _ParameterizeWithTags(tags_by_output={})

    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in ci_support_datasets:
        table_key = spec["table_key"]
        producer_target = spec["producer_target"]
        domain = spec["domain"]
        node_name = query_node(table_key)
        mapping[node_name] = {"ref": source(dataset_node(table_key))}
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_loader_query(
                domain=domain,
                target=producer_target,
                table_key=table_key,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_query_nodes)
def load_relation(env: BuildEnv, ref: DatasetRef) -> DuckDBRelation:
    """Load a dataset as a DuckDB relation.

    Returns
    -------
    DuckDBRelation
        DuckDB relation for the dataset reference.
    """
    return load_dataset_relation(gateway=env.gateway, ref=ref)


def _decorate_artifact_nodes(
    ci_support_artifacts: Sequence[Mapping[str, str]] | None = None,
    *,
    ci_support_include_artifact_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_artifact_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_artifacts:
        return _ParameterizeWithTags(tags_by_output={})

    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in ci_support_artifacts:
        artifact_name = spec["artifact_name"]
        producer_target = spec["producer_target"]
        domain = spec["domain"]
        node_name = artifact_node(artifact_name)
        mapping[node_name] = {
            "record": source(target_node(producer_target)),
            "artifact_name": value(artifact_name),
        }
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_artifact(
                domain=domain,
                target=producer_target,
                artifact_name=artifact_name,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_artifact_nodes)
def artifact_ref(
    env: BuildEnv,
    record: TargetRunRecord,
    artifact_name: str,
) -> ArtifactRef:
    """Return ArtifactRef for a target output.

    Returns
    -------
    ArtifactRef
        Artifact reference for the requested artifact.
    """
    for art in record.artifacts:
        if art.name == artifact_name:
            if isinstance(art, ArtifactRef):
                return art
            return ArtifactRef(
                name=art.name,
                artifact_type=art.artifact_type,
                repo=art.repo,
                commit=art.commit,
                path=art.path,
            )
    return ArtifactRef(
        name=artifact_name,
        artifact_type="unknown",
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        path=None,
    )


def _decorate_path_nodes(
    ci_support_artifacts: Sequence[Mapping[str, str]] | None = None,
    *,
    ci_support_include_artifact_path_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_artifact_path_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_artifacts:
        return _ParameterizeWithTags(tags_by_output={})

    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in ci_support_artifacts:
        artifact_name = spec["artifact_name"]
        producer_target = spec["producer_target"]
        domain = spec["domain"]
        node_name = path_node(artifact_name)
        mapping[node_name] = {"ref": source(artifact_node(artifact_name))}
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_helper(
                domain=domain,
                target=producer_target,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_path_nodes)
def artifact_path(ref: ArtifactRef) -> Path | None:
    """Return filesystem path for an artifact reference.

    Returns
    -------
    Path | None
        Filesystem path when available, otherwise None.
    """
    if ref.path is None:
        return None
    return Path(ref.path)


__all__ = [
    "artifact_path",
    "artifact_ref",
    "dataset_ref",
    "load_relation",
]
