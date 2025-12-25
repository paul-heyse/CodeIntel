"""Save-to decorator helpers for native Hamilton targets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar

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


def _build_common_kwargs(
    *,
    target: str,
    output_role: OutputRole | None,
    hash_options_node: str | None,
) -> dict[str, ParametrizedDependency]:
    kwargs: dict[str, ParametrizedDependency] = {
        "env": source("env"),
        "graph": source("graph"),
        "target_name": value(target),
    }
    if output_role is not None:
        kwargs["output_role"] = value(output_role)
    if hash_options_node is not None:
        kwargs["hash_options"] = source(hash_options_node)
    return kwargs


def save_artifact(
    *,
    domain: str,
    target: str,
    artifact_name: str,
    path_template: str,
    output_role: OutputRole | None = None,
    output_name: str | None = None,
    hash_options_node: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes a file artifact output."""
    decorator = SaveToObjectMetadataDecorator(
        [FileArtifactSaver],
        output_name_=output_name or materialize_node(f"artifact.{artifact_name}"),
        artifact_name=value(artifact_name),
        path_template=value(path_template),
        **_build_common_kwargs(
            target=target,
            output_role=output_role,
            hash_options_node=hash_options_node,
        ),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(domain=domain, target=target, extra_tags=extra_tags)(fn)
        return decorator(tagged)

    return apply


def save_rows(
    *,
    domain: str,
    target: str,
    table_key: str,
    columns: tuple[str, ...] | DeferredColumns | None = None,
    output_role: OutputRole | None = None,
    output_name: str | None = None,
    hash_options_node: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes row-based table outputs."""
    resolved_columns = (
        columns if columns is not None else deferred_columns_for_table_key(table_key)
    )
    decorator = SaveToObjectMetadataDecorator(
        [DuckDBRowsSaver],
        output_name_=output_name or materialize_node(table_key),
        table_key=value(table_key),
        columns=value(resolved_columns),
        **_build_common_kwargs(
            target=target,
            output_role=output_role,
            hash_options_node=hash_options_node,
        ),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(domain=domain, target=target, extra_tags=extra_tags)(fn)
        return decorator(tagged)

    return apply


def save_ibis_table(
    *,
    domain: str,
    target: str,
    table_key: str,
    output_role: OutputRole | None = None,
    output_name: str | None = None,
    hash_options_node: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes Ibis table outputs."""
    decorator = SaveToObjectMetadataDecorator(
        [DuckDBIbisTableSaver],
        output_name_=output_name or materialize_node(table_key),
        table_key=value(table_key),
        **_build_common_kwargs(
            target=target,
            output_role=output_role,
            hash_options_node=hash_options_node,
        ),
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        tagged = tag_compute(domain=domain, target=target, extra_tags=extra_tags)(fn)
        return decorator(tagged)

    return apply


def save_artifact_internal(
    *,
    domain: str,
    target: str,
    artifact_name: str,
    path_template: str,
    output_name: str | None = None,
    hash_options_node: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator for internal file artifact outputs."""
    return save_artifact(
        domain=domain,
        target=target,
        artifact_name=artifact_name,
        path_template=path_template,
        output_role="internal",
        output_name=output_name,
        hash_options_node=hash_options_node,
        extra_tags=extra_tags,
    )


def save_rows_internal(
    *,
    domain: str,
    target: str,
    table_key: str,
    columns: tuple[str, ...] | DeferredColumns | None = None,
    output_name: str | None = None,
    hash_options_node: str | None = None,
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator for internal table outputs."""
    return save_rows(
        domain=domain,
        target=target,
        table_key=table_key,
        columns=columns,
        output_role="internal",
        output_name=output_name,
        hash_options_node=hash_options_node,
        extra_tags=extra_tags,
    )


__all__ = [
    "save_artifact",
    "save_artifact_internal",
    "save_ibis_table",
    "save_rows",
    "save_rows_internal",
]
