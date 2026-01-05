"""Static Hamilton support nodes expanded from config."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import parameterize, resolve_from_config, source, value

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.naming import (
    artifact_node,
    dataset_node,
    path_node,
    query_node,
    target_node,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tag_spec import TagKey, TagSpec, TagValue
from codeintel.build.tabular.types import TabularInput

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hamilton.function_modifiers.base import NodeTransformLifecycle
    from hamilton.function_modifiers.dependencies import ParametrizedDependency
    from hamilton.node import Node

log = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class _SupportDatasetConfig:
    table_key: str
    producer_target: str | None
    domain: str
    data_node: str | None
    allowlisted: bool


def _parse_support_dataset_spec(
    spec: Mapping[str, object],
) -> _SupportDatasetConfig:
    table_key_raw = spec.get("table_key")
    if not isinstance(table_key_raw, str) or not table_key_raw:
        msg = "Support dataset spec missing table_key"
        raise ValueError(msg)
    domain_raw = spec.get("domain")
    if not isinstance(domain_raw, str) or not domain_raw:
        msg = f"Support dataset spec missing domain for {table_key_raw}"
        raise ValueError(msg)
    producer_raw = spec.get("producer_target")
    producer_target = producer_raw if isinstance(producer_raw, str) and producer_raw else None
    data_node_raw = spec.get("data_node")
    data_node = data_node_raw if isinstance(data_node_raw, str) and data_node_raw else None
    allowlisted = bool(spec.get("allowlisted"))
    return _SupportDatasetConfig(
        table_key=table_key_raw,
        producer_target=producer_target,
        domain=domain_raw,
        data_node=data_node,
        allowlisted=allowlisted,
    )


def _parse_support_dataset_specs(
    ci_support_datasets: Sequence[Mapping[str, object]],
) -> list[_SupportDatasetConfig]:
    return [_parse_support_dataset_spec(spec) for spec in ci_support_datasets]


def _decorate_dataset_nodes(
    ci_support_datasets: Sequence[Mapping[str, object]] | None = None,
    *,
    ci_support_include_dataset_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_dataset_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_datasets:
        return _ParameterizeWithTags(tags_by_output={})

    parsed_specs = _parse_support_dataset_specs(ci_support_datasets)
    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in parsed_specs:
        table_key = spec.table_key
        producer_target = spec.producer_target
        domain = spec.domain
        node_name = dataset_node(table_key)
        mapping[node_name] = {
            "table_key": value(table_key),
            "producer_target": value(producer_target),
        }
        if producer_target is not None:
            mapping[node_name]["record"] = source(target_node(producer_target))
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_dataset_ref(
                domain=domain,
                target=producer_target,
                table_key=table_key,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_dataset_nodes)
def dataset_ref(
    env: BuildEnv,
    table_key: str,
    producer_target: str | None,
    record: TargetRunRecord | None = None,
) -> DatasetRef:
    """Return DatasetRef for a target output.

    Returns
    -------
    DatasetRef
        Dataset reference for the requested table.

    Raises
    ------
    ValueError
        If the dataset reference is missing for the producer target on a failed run.
    """
    if record is not None:
        ds = record.get_dataset(table_key)
        if ds is not None:
            if isinstance(ds, DatasetRef):
                return ds
            return DatasetRef(
                table_key=ds.table_key,
                repo=ds.repo,
                commit=ds.commit,
                row_count=ds.row_count,
                source_target=producer_target,
            )
        row_count = record.row_counts.get(table_key) if record.row_counts else None
        if record.success or record.skipped:
            return DatasetRef(
                table_key=table_key,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                row_count=row_count,
                source_target=producer_target,
            )
        log.warning(
            "Missing DatasetRef for %s from %s (status=%s); returning empty reference",
            table_key,
            producer_target,
            record.status,
        )
        return DatasetRef(
            table_key=table_key,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            row_count=row_count,
            source_target=producer_target,
        )

    resolved_repo = env.snapshot.repo
    resolved_commit = env.snapshot.commit
    if not resolved_repo or not resolved_commit:
        msg = f"DatasetRef missing repo/commit for {table_key}"
        raise ValueError(msg)
    return DatasetRef(
        table_key=table_key,
        repo=resolved_repo,
        commit=resolved_commit,
        source_target=producer_target,
    )


def _requires_scip_gate(table_key: str) -> bool:
    return table_key.startswith("core.scip_")


def scip_ready(t__scip: TargetRunRecord) -> TargetRunRecord:
    """Ensure SCIP completed before dependent nodes run.

    Returns
    -------
    TargetRunRecord
        SCIP target run record for the current execution.

    Raises
    ------
    RuntimeError
        If the SCIP target was skipped or failed.
    """
    if t__scip.skipped:
        msg = "SCIP target skipped; SCIP-backed data is required for this run"
        raise RuntimeError(msg)
    if not t__scip.success:
        msg = t__scip.error or "SCIP target failed"
        raise RuntimeError(msg)
    return t__scip


def _decorate_query_nodes(
    ci_support_datasets: Sequence[Mapping[str, object]] | None = None,
    *,
    ci_support_include_loader_nodes: bool = True,
) -> NodeTransformLifecycle:
    if not ci_support_include_loader_nodes:
        return _ParameterizeWithTags(tags_by_output={})
    if not ci_support_datasets:
        return _ParameterizeWithTags(tags_by_output={})

    mapping: dict[str, dict[str, ParametrizedDependency]] = {}
    tags_by_output: dict[str, dict[str, str]] = {}
    for spec in _parse_support_dataset_specs(ci_support_datasets):
        table_key = spec.table_key
        producer_target = spec.producer_target
        domain = spec.domain
        node_name = query_node(table_key)
        params: dict[str, ParametrizedDependency] = {}
        if spec.data_node is not None:
            params["data"] = source(spec.data_node)
        else:
            if not spec.allowlisted:
                msg = f"External input {table_key} is not allowlisted"
                raise ValueError(msg)
            params["ref"] = source(dataset_node(table_key))
        if _requires_scip_gate(table_key):
            params["scip_ready"] = source("scip_ready")
        mapping[node_name] = params
        tags_by_output[node_name] = _normalize_tags(
            TagSpec.for_loader_query(
                domain=domain,
                target=producer_target,
                table_key=table_key,
            ).to_tags()
        )

    return _ParameterizeWithTags(tags_by_output=tags_by_output, **mapping)


@resolve_from_config(decorate_with=_decorate_query_nodes)
def load_relation(
    env: BuildEnv,
    ref: DatasetRef | None = None,
    data: TabularInput | None = None,
    scip_ready: TargetRunRecord | None = None,
) -> TabularInput:
    """Load a dataset as an inferable tabular input.

    Returns
    -------
    TabularInput
        Tabular input for the dataset reference.

    Raises
    ------
    ValueError
        If the snapshot_id cannot be resolved for the dataset reference.
    """
    _ = scip_ready
    if data is not None:
        return data
    if ref is None:
        msg = "Missing DatasetRef for external input"
        raise ValueError(msg)
    snapshot_id = ref.commit or env.commit
    if not snapshot_id:
        msg = f"Missing snapshot_id for {ref.table_key}"
        raise ValueError(msg)
    return load_snapshot_tabular(
        env=env,
        table_key=ref.table_key,
        snapshot_id=snapshot_id,
    )


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
    "scip_ready",
]
