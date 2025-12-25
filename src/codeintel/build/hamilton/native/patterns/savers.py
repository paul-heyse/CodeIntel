"""Save-to decorator helpers for native Hamilton targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.materializers import (
    DuckDBIbisTableSaver,
    DuckDBRowsSaver,
    FileArtifactSaver,
)
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.patterns.specs import OutputRole
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import TagKey, TagValue, tag_compute
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.schemas.column_resolution import DeferredColumns

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hamilton.function_modifiers.dependencies import ParametrizedDependency

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class SaverContext:
    """Context for save-to decorator helpers."""

    domain: str
    target: str
    hash_options_node: str | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSaveSpec:
    """Specification for saving a file artifact."""

    artifact_name: str
    path_template: str
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class TableSaveSpec:
    """Specification for saving a table from row output."""

    table_key: str
    columns: tuple[str, ...] | DeferredColumns | None = None
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class IbisTableSaveSpec:
    """Specification for saving an Ibis table."""

    table_key: str
    output_role: OutputRole | None = None
    output_name: str | None = None


def _dep(value: object) -> ParametrizedDependency:
    return cast("ParametrizedDependency", value)


def _hash_options_dep(context: SaverContext) -> ParametrizedDependency:
    if context.hash_options_node is None:
        return _dep(value(None))
    return _dep(source(context.hash_options_node))


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
        graph=_dep(source("graph")),
        target_name=_dep(value(context.target)),
        artifact_name=_dep(value(spec.artifact_name)),
        path_template=_dep(value(spec.path_template)),
        output_role=_dep(value(spec.output_role)),
        hash_options=_hash_options_dep(context),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(
            domain=context.domain,
            target=context.target,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_rows(
    *,
    context: SaverContext,
    spec: TableSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes row-based table outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Table output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the row output.
    """
    resolved_columns = (
        spec.columns if spec.columns is not None else deferred_columns_for_table_key(spec.table_key)
    )
    decorator = SaveToObjectMetadataDecorator(
        [DuckDBRowsSaver],
        output_name_=spec.output_name or materialize_node(spec.table_key),
        env=_dep(source("env")),
        graph=_dep(source("graph")),
        target_name=_dep(value(context.target)),
        table_key=_dep(value(spec.table_key)),
        columns=_dep(value(resolved_columns)),
        output_role=_dep(value(spec.output_role)),
        hash_options=_hash_options_dep(context),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(
            domain=context.domain,
            target=context.target,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_ibis_table(
    *,
    context: SaverContext,
    spec: IbisTableSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes Ibis table outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Ibis table output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the Ibis table output.
    """
    decorator = SaveToObjectMetadataDecorator(
        [DuckDBIbisTableSaver],
        output_name_=spec.output_name or materialize_node(spec.table_key),
        env=_dep(source("env")),
        graph=_dep(source("graph")),
        target_name=_dep(value(context.target)),
        table_key=_dep(value(spec.table_key)),
        output_role=_dep(value(spec.output_role)),
        hash_options=_hash_options_dep(context),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(
            domain=context.domain,
            target=context.target,
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


def save_rows_internal(
    *,
    context: SaverContext,
    spec: TableSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator for internal table outputs.

    Parameters
    ----------
    context
        Shared context for tagging and saver metadata.
    spec
        Table output specification.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that tags and materializes the internal table output.
    """
    return save_rows(
        context=context,
        spec=replace(spec, output_role="internal"),
    )


__all__ = [
    "ArtifactSaveSpec",
    "IbisTableSaveSpec",
    "SaverContext",
    "TableSaveSpec",
    "save_artifact",
    "save_artifact_internal",
    "save_ibis_table",
    "save_rows",
    "save_rows_internal",
]
