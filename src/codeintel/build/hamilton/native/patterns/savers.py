"""Save-to decorator helpers for native Hamilton targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.materializers import FileArtifactSaver, IcebergDatasetSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.patterns.specs import OutputRole
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import TagKey, TagValue, tag_compute, tag_dataset

if TYPE_CHECKING:
    from hamilton.function_modifiers.dependencies import ParametrizedDependency

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class SaverContext:
    """Context for save-to decorator helpers."""

    domain: str
    target: str
    extra_tags: Mapping[TagKey, TagValue] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSaveSpec:
    """Specification for saving a file artifact."""

    artifact_name: str
    path_template: str
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSaveSpec:
    """Specification for saving a table as an Iceberg dataset."""

    table_key: str
    partition_columns: tuple[str, ...] = ()
    collect_group: str | None = None
    validation_profile: str | None = "lenient"
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class RelationTableSaveSpec:
    """Specification for saving a relation table to Iceberg."""

    table_key: str
    validation_profile: str | None = "lenient"
    output_role: OutputRole | None = None
    output_name: str | None = None


def _dep(value: object) -> ParametrizedDependency:
    return cast("ParametrizedDependency", value)


def save_artifact(
    *,
    context: SaverContext,
    spec: ArtifactSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes a file artifact output.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Artifact output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the artifact output.
    """
    decorator = SaveToObjectMetadataDecorator(
        [FileArtifactSaver],
        output_name_=spec.output_name or materialize_node(f"artifact.{spec.artifact_name}"),
        env=_dep(source("env")),
        catalog=_dep(source("catalog")),
        target_name=_dep(value(context.target)),
        artifact_name=_dep(value(spec.artifact_name)),
        path_template=_dep(value(spec.path_template)),
        output_role=_dep(value(spec.output_role)),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(
            domain=context.domain,
            target=context.target,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_dataset(
    *,
    context: SaverContext,
    spec: DatasetSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes Iceberg dataset outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Dataset output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the dataset output.
    """
    decorator = SaveToObjectMetadataDecorator(
        [IcebergDatasetSaver],
        output_name_=spec.output_name or materialize_node(spec.table_key),
        env=_dep(source("env")),
        catalog=_dep(source("catalog")),
        target_name=_dep(value(context.target)),
        table_key=_dep(value(spec.table_key)),
        partition_columns=_dep(value(spec.partition_columns)),
        collect_group=_dep(value(spec.collect_group)),
        validation_profile=_dep(value(spec.validation_profile)),
        output_role=_dep(value(spec.output_role)),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_dataset(
            domain=context.domain,
            target=context.target,
            table_key=spec.table_key,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_relation_table(
    *,
    context: SaverContext,
    spec: RelationTableSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes relation table outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Relation table output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the relation table output.
    """
    decorator = SaveToObjectMetadataDecorator(
        [IcebergDatasetSaver],
        output_name_=spec.output_name or materialize_node(spec.table_key),
        env=_dep(source("env")),
        catalog=_dep(source("catalog")),
        target_name=_dep(value(context.target)),
        table_key=_dep(value(spec.table_key)),
        validation_profile=_dep(value(spec.validation_profile)),
        output_role=_dep(value(spec.output_role)),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_dataset(
            domain=context.domain,
            target=context.target,
            table_key=spec.table_key,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_artifact_internal(
    *,
    context: SaverContext,
    spec: ArtifactSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator for internal file artifact outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Artifact output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the internal artifact output.
    """
    return save_artifact(
        context=context,
        spec=replace(spec, output_role="internal"),
    )


__all__ = [
    "ArtifactSaveSpec",
    "DatasetSaveSpec",
    "RelationTableSaveSpec",
    "SaverContext",
    "save_artifact",
    "save_artifact_internal",
    "save_dataset",
    "save_relation_table",
]
