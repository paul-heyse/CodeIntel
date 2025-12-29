"""Stable save-to decorator helpers for plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ParamSpec, TypeVar, cast

from codeintel.build.hamilton.native.patterns.savers import (
    ArtifactSaveSpec,
    DatasetSaveSpec,
    RelationTableSaveSpec,
    SaverContext,
    save_artifact,
    save_dataset,
    save_relation_table,
)
from codeintel.build.hamilton.native.patterns.specs import OutputRole
from codeintel.build.hamilton.tagging import TagKey, TagValue

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


@dataclass(frozen=True, slots=True)
class TableSaveOptions:
    """Options for materializing a table output."""

    table_key: str
    partition_columns: Sequence[str] = ()
    collect_group: str | None = None
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSaveOptions:
    """Options for materializing an artifact output."""

    artifact_name: str
    path_template: str
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class RelationTableSaveOptions:
    """Options for materializing a relation table output."""

    table_key: str
    output_role: OutputRole | None = None
    output_name: str | None = None


def save_to_table(
    *,
    domain: str,
    target: str,
    options: TableSaveOptions,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Return a decorator that materializes a table output.

    Parameters
    ----------
    domain
        Target domain name.
    target
        Target identifier.
    options
        Table materialization options.
    extra_tags
        Additional tags to apply to the target materialization nodes.

    Returns
    -------
    Decorator[P, R]
        Decorator that materializes the table output.
    """
    context = SaverContext(domain=domain, target=target, extra_tags=extra_tags)
    spec = DatasetSaveSpec(
        table_key=options.table_key,
        partition_columns=tuple(options.partition_columns),
        collect_group=options.collect_group,
        output_role=options.output_role,
        output_name=options.output_name,
    )
    return cast("Decorator[P, R]", save_dataset(context=context, spec=spec))


def save_to_artifact(
    *,
    domain: str,
    target: str,
    options: ArtifactSaveOptions,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Return a decorator that materializes a file artifact output.

    Parameters
    ----------
    domain
        Target domain name.
    target
        Target identifier.
    options
        Artifact materialization options.
    extra_tags
        Additional tags to apply to the target materialization nodes.

    Returns
    -------
    Decorator[P, R]
        Decorator that materializes the artifact output.
    """
    context = SaverContext(domain=domain, target=target, extra_tags=extra_tags)
    spec = ArtifactSaveSpec(
        artifact_name=options.artifact_name,
        path_template=options.path_template,
        output_role=options.output_role,
        output_name=options.output_name,
    )
    return cast("Decorator[P, R]", save_artifact(context=context, spec=spec))


def save_to_relation_table(
    *,
    domain: str,
    target: str,
    options: RelationTableSaveOptions,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Return a decorator that materializes a DuckDB relation output.

    Parameters
    ----------
    domain
        Target domain name.
    target
        Target identifier.
    options
        Relation table materialization options.
    extra_tags
        Additional tags to apply to the target materialization nodes.

    Returns
    -------
    Decorator[P, R]
        Decorator that materializes the relation table output.
    """
    context = SaverContext(domain=domain, target=target, extra_tags=extra_tags)
    spec = RelationTableSaveSpec(
        table_key=options.table_key,
        output_role=options.output_role,
        output_name=options.output_name,
    )
    return cast("Decorator[P, R]", save_relation_table(context=context, spec=spec))


__all__ = [
    "ArtifactSaveOptions",
    "RelationTableSaveOptions",
    "TableSaveOptions",
    "save_to_artifact",
    "save_to_relation_table",
    "save_to_table",
]
