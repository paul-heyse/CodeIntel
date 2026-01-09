"""Save-to decorator helpers for native Hamilton targets."""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import hamilton.node as h_node
from hamilton.function_modifiers import (
    cache,
    check_output_custom,
    resolve_from_config,
    source,
    value,
)
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.hamilton.data_quality import build_table_schema_validators
from codeintel.build.hamilton.materializers import ArrowDatasetSaver, FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.patterns.specs import OutputRole
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import TagKey, TagValue, tag_compute, tag_dataset
from codeintel.build.tabular.finalize_ops import FinalizeMode
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.hamilton import tags as ht
from codeintel.core.validation.profiles import ValidationProfile, normalize_validation_profile

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
    """Specification for saving a table as an Arrow dataset."""

    table_key: str
    partition_columns: tuple[str, ...] = ()
    collect_group: str | None = None
    validation_profile: ValidationProfile | None = "lenient"
    output_role: OutputRole | None = None
    output_name: str | None = None
    manifest_extras_node: str | None = None
    ingest_finalize: bool = False
    ingest_finalize_mode: FinalizeMode | None = None


@dataclass(frozen=True, slots=True)
class RelationTableSaveSpec:
    """Specification for saving a table output."""

    table_key: str
    validation_profile: ValidationProfile | None = "lenient"
    output_role: OutputRole | None = None
    output_name: str | None = None
    manifest_extras_node: str | None = None
    ingest_finalize: bool = False
    ingest_finalize_mode: FinalizeMode | None = None


def _dep(value: object) -> ParametrizedDependency:
    return cast("ParametrizedDependency", value)


class _NoOpTransform(NodeTransformLifecycle):
    """No-op decorator used when validation is disabled."""

    @classmethod
    def get_lifecycle_name(cls) -> str:
        return "codeintel_noop_validation"

    @classmethod
    def allows_multiple(cls) -> bool:
        return True

    def validate(self, fn: Callable[..., object]) -> None:
        _ = (self, fn)

    def __call__(self, fn: Callable[..., object]) -> Callable[..., object]:
        return fn


_VALIDATOR_NODE_TAG = "hamilton.data_quality.contains_dq_results"


class _TaggedValidation(check_output_custom):
    """Validator decorator that hides data-quality nodes from UI outputs."""

    def transform_node(
        self,
        node_: h_node.Node,
        config: dict[str, Any],
        fn: Callable[..., object],
    ) -> Collection[h_node.Node]:
        nodes = super().transform_node(node_, config, fn)
        for node in nodes:
            if _VALIDATOR_NODE_TAG in node.tags or node.name.endswith("_raw"):
                node.tags[ht.TAG_MCP_VISIBLE] = "0"
        return nodes


def _resolve_validation_profile(
    *,
    default_profile: ValidationProfile | None,
    config_mode: str,
) -> ValidationProfile | None:
    if not isinstance(config_mode, str):
        return default_profile
    normalized = config_mode.strip().lower()
    if normalized in {"off", "none", ""}:
        return None
    try:
        return normalize_validation_profile(
            normalized,
            default=default_profile or "strict",
        )
    except ValueError:
        return default_profile


def _resolve_min_rows(
    *,
    table_key: str,
    base_min_rows: int,
    overrides: Mapping[str, int] | None,
) -> int:
    if not overrides or not isinstance(overrides, Mapping):
        return base_min_rows
    override = overrides.get(table_key)
    if isinstance(override, int) and override >= 0:
        return override
    return base_min_rows


def _validation_from_config(
    *,
    table_key: str,
    default_profile: ValidationProfile | None,
) -> NodeTransformLifecycle:
    def _factory(
        *,
        ci_validate_outputs: bool = False,
        ci_validation_mode: str = "lenient",
        ci_validation_min_rows: int = 0,
        ci_validation_min_rows_by_table: Mapping[str, int] | None = None,
    ) -> NodeTransformLifecycle:
        if not ci_validate_outputs:
            return _NoOpTransform()
        profile = _resolve_validation_profile(
            default_profile=default_profile,
            config_mode=ci_validation_mode,
        )
        if profile is None:
            return _NoOpTransform()
        base_min_rows = ci_validation_min_rows if isinstance(ci_validation_min_rows, int) else 0
        min_rows = _resolve_min_rows(
            table_key=table_key,
            base_min_rows=base_min_rows,
            overrides=ci_validation_min_rows_by_table,
        )
        validators = build_table_schema_validators(
            table_key=table_key,
            profile=profile,
            min_rows=min_rows,
        )
        if not validators:
            return _NoOpTransform()
        return _TaggedValidation(*validators)

    return resolve_from_config(decorate_with=_factory)


def _coerce_none_output[**P_coerce, R_coerce](
    fn: Callable[P_coerce, R_coerce],
    *,
    table_key: str,
) -> Callable[P_coerce, R_coerce]:
    @functools.wraps(fn)
    def wrapper(*args: P_coerce.args, **kwargs: P_coerce.kwargs) -> R_coerce:
        result = fn(*args, **kwargs)
        if result is None:
            return cast("R_coerce", empty_table_for_table(table_key))
        return result

    return wrapper


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
        data_node_name = fn.__name__
        tagged = tag_compute(
            domain=context.domain,
            target=context.target,
            target_=data_node_name,
            extra_tags=context.extra_tags,
        )(fn)
        return decorator(tagged)

    return apply


def save_dataset(
    *,
    context: SaverContext,
    spec: DatasetSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes Arrow dataset outputs.

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

    def _materialize() -> NodeTransformLifecycle:
        manifest_deps: dict[str, ParametrizedDependency] = {}
        if spec.manifest_extras_node is not None:
            manifest_deps["manifest_extras"] = _dep(source(spec.manifest_extras_node))
        ingest_deps: dict[str, ParametrizedDependency] = {
            "ingest_finalize": _dep(value(spec.ingest_finalize)),
            "ingest_finalize_mode": _dep(value(spec.ingest_finalize_mode)),
        }
        return SaveToObjectMetadataDecorator(
            [ArrowDatasetSaver],
            output_name_=spec.output_name or materialize_node(spec.table_key),
            env=_dep(source("env")),
            catalog=_dep(source("catalog")),
            target_name=_dep(value(context.target)),
            table_key=_dep(value(spec.table_key)),
            partition_columns=_dep(value(spec.partition_columns)),
            collect_group=_dep(value(spec.collect_group)),
            validation_profile=_dep(value(spec.validation_profile)),
            output_role=_dep(value(spec.output_role)),
            **manifest_deps,
            **ingest_deps,
        )

    materializer = resolve_from_config(decorate_with=_materialize)
    validator = _validation_from_config(
        table_key=spec.table_key,
        default_profile=spec.validation_profile,
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        data_node_name = fn.__name__
        tagged = tag_dataset(
            domain=context.domain,
            target=context.target,
            table_key=spec.table_key,
            target_=data_node_name,
            extra_tags=context.extra_tags,
        )(fn)
        coerced = _coerce_none_output(tagged, table_key=spec.table_key)
        validated = validator(coerced)
        cached = cache(behavior="default", format="parquet", target_=data_node_name)(validated)
        return materializer(cached)

    return apply


def save_relation_table(
    *,
    context: SaverContext,
    spec: RelationTableSaveSpec,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator that tags and materializes table outputs.

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

    def _materialize() -> NodeTransformLifecycle:
        manifest_deps: dict[str, ParametrizedDependency] = {}
        if spec.manifest_extras_node is not None:
            manifest_deps["manifest_extras"] = _dep(source(spec.manifest_extras_node))
        ingest_deps: dict[str, ParametrizedDependency] = {
            "ingest_finalize": _dep(value(spec.ingest_finalize)),
            "ingest_finalize_mode": _dep(value(spec.ingest_finalize_mode)),
        }
        return SaveToObjectMetadataDecorator(
            [ArrowDatasetSaver],
            output_name_=spec.output_name or materialize_node(spec.table_key),
            env=_dep(source("env")),
            catalog=_dep(source("catalog")),
            target_name=_dep(value(context.target)),
            table_key=_dep(value(spec.table_key)),
            validation_profile=_dep(value(spec.validation_profile)),
            output_role=_dep(value(spec.output_role)),
            **manifest_deps,
            **ingest_deps,
        )

    materializer = resolve_from_config(decorate_with=_materialize)
    validator = _validation_from_config(
        table_key=spec.table_key,
        default_profile=spec.validation_profile,
    )

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        data_node_name = fn.__name__
        tagged = tag_dataset(
            domain=context.domain,
            target=context.target,
            table_key=spec.table_key,
            target_=data_node_name,
            extra_tags=context.extra_tags,
        )(fn)
        coerced = _coerce_none_output(tagged, table_key=spec.table_key)
        validated = validator(coerced)
        cached = cache(behavior="default", format="parquet", target_=data_node_name)(validated)
        return materializer(cached)

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
