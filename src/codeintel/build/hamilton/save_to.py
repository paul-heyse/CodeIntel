"""Project-local Hamilton saver decorator variants.

Hamilton's built-in ``SaveToDecorator`` always types saver metadata nodes as
``typing.Dict[str, typing.Any]``. This repo uses a stricter contract: saver metadata is treated
as ``MaterializationResult`` at the DAG boundary.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeAliasType,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

import hamilton.node as h_node
from hamilton.function_modifiers.adapters import (
    AdapterFactory,
    resolve_adapter_class,
    resolve_kwargs,
)
from hamilton.function_modifiers.base import InvalidDecoratorException, SingleNodeNodeTransformer
from hamilton.node import DependencyType

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.materializers.path_templates import validate_path_template
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.provider_unified import UnifiedSchemaProvider
from codeintel.core.hamilton import tags as ht
from codeintel.core.schemas.declared import declared_schema_provider
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.resolution import ResolvedSchemaProvider, resolve_table_schema

_TAG_ONLY_KWARGS = {
    "output_role",
    "json_schema_id",
    "jsonl_filename",
    "parquet_filename",
    "dataset_owner",
    "validation_profile",
}

_METADATA_TAGS: dict[str, str] = {
    "json_schema_id": "ci.json_schema_id",
    "jsonl_filename": "ci.jsonl_filename",
    "parquet_filename": "ci.parquet_filename",
    "dataset_owner": "ci.dataset_owner",
    "validation_profile": "ci.validation_profile",
    "collect_group": "ci.collect_group",
}

_VALIDATION_PROFILES: frozenset[str] = frozenset({"strict", "lenient"})
_SCHEMA_OUTPUT_TAG = "hamilton.internal.schema_output"

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from hamilton.function_modifiers.dependencies import ParametrizedDependency
    from hamilton.io.data_adapters import AdapterCommon, DataSaver


@runtime_checkable
class _SupportsInference(Protocol):
    """Protocol for schema providers that can toggle inference."""

    def with_inference(self, *, allow_inference: bool) -> SchemaProvider:
        """Return a schema provider with inference enabled or disabled."""
        ...


class SaveToObjectMetadataDecorator(SingleNodeNodeTransformer):
    """Save-to decorator that types metadata nodes as ``MaterializationResult``."""

    def __init__(
        self,
        saver_classes_: Collection[type[DataSaver]],
        output_name_: str | None = None,
        target_: str | None = None,
        **kwargs: ParametrizedDependency,
    ) -> None:
        """Create a save-to decorator for typed saver metadata.

        Parameters
        ----------
        saver_classes_
            Candidate saver classes to use for persisting the upstream node output.
        output_name_
            Name of the saver metadata node to create.
        target_
            Optional name of the node to save (defaults to the decorated node).
        **kwargs
            Saver constructor arguments (source/value dependencies).
        """
        super().__init__()
        self.artifact_name = output_name_
        self.saver_classes: Sequence[type[AdapterCommon]] = tuple(saver_classes_)
        self.kwargs = kwargs
        self.target = target_

    def create_saver_node(
        self,
        node_: h_node.Node,
        _config: dict[str, object],
        fn: Callable[..., object],
    ) -> h_node.Node:
        """Create the saver metadata node for a decorated function node.

        Parameters
        ----------
        node_
            Node producing the value to persist.
        _config
            Hamilton configuration (unused).
        fn
            Function that produced ``node_``.

        Returns
        -------
        hamilton.node.Node
            A metadata node that executes the saver and returns a materialization result.

        """
        target_override = _normalize_target_override(target_override=self.target, fn=fn)
        inputs = _SaverNodeInputs(
            node=node_,
            fn=fn,
            output_name=self.artifact_name,
            saver_classes=self.saver_classes,
            kwargs=self.kwargs,
            target_override=target_override,
        )
        resolution = _resolve_saver_node(inputs=inputs)

        def save_data(
            __adapter_factory: AdapterFactory = resolution.adapter_factory,
            __dependencies: dict[str, str] = resolution.dependencies_inverted,
            __resolved_kwargs: dict[str, object] = resolution.resolved_kwargs,
            __data_node_name: str = resolution.node_to_save_str,
            __table_key: str | None = resolution.table_key,
            __artifact_name: str | None = resolution.artifact_name,
            /,
            **input_kwargs: object,
        ) -> MaterializationResult:
            input_args_with_fixed_dependencies = {
                __dependencies.get(key, key): value for key, value in input_kwargs.items()
            }
            merged_kwargs = {**__resolved_kwargs, **input_args_with_fixed_dependencies}
            data_to_save = merged_kwargs[__data_node_name]
            saver_kwargs = {
                k: v
                for k, v in merged_kwargs.items()
                if k != __data_node_name and k not in _TAG_ONLY_KWARGS
            }
            data_saver = __adapter_factory.create_saver(**saver_kwargs)
            metadata = data_saver.save_data(data_to_save)
            return _coerce_materialization_result(
                metadata,
                table_key=__table_key,
                artifact_name=__artifact_name,
                saver_name=type(data_saver).__name__,
            )

        input_types = _build_input_types(node_=node_, resolution=resolution)
        node_input_types = cast(
            "dict[str, type | tuple[type, DependencyType]]",
            input_types,
        )

        return h_node.Node(
            name=resolution.metadata_node_name,
            callabl=save_data,
            typ=cast("type[object]", MaterializationResult),
            input_types=node_input_types,
            namespace=resolution.metadata_namespace,
            tags=_build_saver_tags(
                node_=node_,
                context=SaverTagContext(
                    sink=resolution.sink,
                    saver_cls=resolution.saver_cls,
                    output_role=resolution.output_role,
                    target_name=resolution.target_name,
                    data_node_name=resolution.node_to_save_str,
                    table_key=resolution.table_key,
                    artifact_name=resolution.artifact_name,
                    path_template=resolution.path_template,
                ),
                metadata_tags=resolution.metadata_tags,
            ),
        )

    def transform_node(
        self, node_: h_node.Node, config: dict[str, object], fn: Callable[..., object]
    ) -> Collection[h_node.Node]:
        """Transform a node into a saver + original node pair.

        Parameters
        ----------
        node_
            Node to wrap with a saver metadata node.
        config
            Hamilton configuration passed through to the node creator.
        fn
            Function that produced ``node_``.

        Returns
        -------
        Collection[hamilton.node.Node]
            The saver metadata node followed by the original node.
        """
        return [self.create_saver_node(node_, config, fn), node_]

    def validate(self, fn: Callable[..., object]) -> None:
        """Validate decorator usage.

        Parameters
        ----------
        fn
            Function being decorated.
        """


@dataclass(frozen=True, slots=True)
class _SaverNodeInputs:
    node: h_node.Node
    fn: Callable[..., object]
    output_name: str | None
    saver_classes: Sequence[type[AdapterCommon]]
    kwargs: dict[str, ParametrizedDependency]
    target_override: str | None


@dataclass(frozen=True, slots=True)
class _SaverNodeResolution:
    adapter_factory: AdapterFactory
    dependencies: dict[str, str]
    dependencies_inverted: dict[str, str]
    resolved_kwargs: dict[str, object]
    node_to_save_str: str
    metadata_node_name: str
    metadata_namespace: tuple[str, ...]
    sink: str
    saver_cls: type[AdapterCommon]
    output_role: str | None
    target_name: str
    table_key: str | None
    artifact_name: str | None
    path_template: str | None
    metadata_tags: dict[str, object]


def _resolve_saver_node(*, inputs: _SaverNodeInputs) -> _SaverNodeResolution:
    node_to_save = inputs.node.name if inputs.target_override is None else inputs.target_override
    metadata_node_name, metadata_namespace = _resolve_metadata_node_name(
        node_to_save=node_to_save,
        output_name=inputs.output_name,
    )

    saver_cls = _resolve_saver_class(
        node_type=inputs.node.type,
        saver_classes=inputs.saver_classes,
        fn=inputs.fn,
    )
    adapter_factory, dependencies, resolved_kwargs = _resolve_saver_factory(
        saver_cls=saver_cls,
        kwargs=inputs.kwargs,
    )
    output_role = _resolve_output_role(
        fn=inputs.fn,
        kwargs=inputs.kwargs,
        resolved_kwargs=resolved_kwargs,
    )
    target_name = _resolve_target_name(fn=inputs.fn, resolved_kwargs=resolved_kwargs)
    table_key, artifact_name = _resolve_output_identity(
        fn=inputs.fn,
        resolved_kwargs=resolved_kwargs,
        output_role=output_role,
    )
    path_template = _resolve_artifact_path_template(
        fn=inputs.fn,
        resolved_kwargs=resolved_kwargs,
        artifact_name=artifact_name,
    )
    metadata_tags = _resolve_metadata_tags(fn=inputs.fn, resolved_kwargs=resolved_kwargs)
    sink = _resolve_saver_sink(saver_cls=saver_cls, fn=inputs.fn)
    dependencies_inverted = {value: key for key, value in dependencies.items()}

    return _SaverNodeResolution(
        adapter_factory=adapter_factory,
        dependencies=dependencies,
        dependencies_inverted=dependencies_inverted,
        resolved_kwargs=resolved_kwargs,
        node_to_save_str=node_to_save,
        metadata_node_name=metadata_node_name,
        metadata_namespace=metadata_namespace,
        sink=sink,
        saver_cls=saver_cls,
        output_role=output_role,
        target_name=target_name,
        table_key=table_key,
        artifact_name=artifact_name,
        path_template=path_template,
        metadata_tags=metadata_tags,
    )


def _normalize_target_override(
    *,
    target_override: object | None,
    fn: Callable[..., object],
) -> str | None:
    if target_override is None:
        return None
    if isinstance(target_override, str) and target_override:
        return target_override
    msg = f"{fn.__qualname__}: target_ must be a non-empty string"
    raise InvalidDecoratorException(msg)


def _build_input_types(
    *,
    node_: h_node.Node,
    resolution: _SaverNodeResolution,
) -> dict[str, tuple[type, DependencyType]]:
    def _input_key(key: str) -> str:
        return resolution.dependencies.get(key, key)

    input_types = {
        _input_key(key): (cast("type", type_), DependencyType.REQUIRED)
        for key, type_ in resolution.saver_cls.get_required_arguments().items()
    }
    input_types.update(
        {
            resolution.dependencies[key]: (cast("type", type_), DependencyType.OPTIONAL)
            for key, type_ in resolution.saver_cls.get_optional_arguments().items()
            if key in resolution.dependencies
        }
    )
    input_types = {
        key: value for key, value in input_types.items() if key not in resolution.resolved_kwargs
    }
    input_types[resolution.node_to_save_str] = (
        cast("type", node_.type),
        DependencyType.REQUIRED,
    )
    return input_types


__all__ = ["SaveToObjectMetadataDecorator"]


def _resolve_metadata_node_name(
    *,
    node_to_save: str,
    output_name: str | None,
) -> tuple[str, tuple[str, ...]]:
    if output_name is None:
        return node_to_save, ("save",)
    return str(output_name), ()


def _resolve_saver_class(
    *,
    node_type: object,
    saver_classes: Sequence[type[AdapterCommon]],
    fn: Callable[..., object],
) -> type[AdapterCommon]:
    normalized_type = _normalize_node_type(node_type)
    node_type_cast = cast("type[type]", normalized_type)
    saver_cls = resolve_adapter_class(node_type_cast, list(saver_classes))
    if saver_cls is None:
        msg = f"No saver class found for type: {node_type!r} (fn={fn.__qualname__})"
        raise InvalidDecoratorException(msg)
    return saver_cls


def _normalize_node_type(node_type: object) -> object:
    resolved = _resolve_type_alias(node_type)
    origin = get_origin(resolved)
    if origin in {types.UnionType, typing.Union}:
        args = _flatten_union_args(get_args(resolved))
        if not args:
            return resolved
        normalized_args = [_resolve_type_alias(arg) for arg in args]
        unique_args = _dedupe_args(normalized_args)
        if len(unique_args) == 1:
            return unique_args[0]
        unionable_args = _coerce_unionable_args(unique_args)
        if unionable_args is None:
            return resolved
        normalized = unionable_args[0]
        for arg in unionable_args[1:]:
            normalized |= arg
        return normalized
    return resolved


def _resolve_type_alias(type_: object) -> object:
    if isinstance(type_, TypeAliasType):
        return _resolve_type_alias(type_.__value__)
    return type_


def _flatten_union_args(args: tuple[object, ...]) -> list[object]:
    flattened: list[object] = []
    for arg in args:
        resolved = _resolve_type_alias(arg)
        origin = get_origin(resolved)
        if origin in {types.UnionType, typing.Union}:
            flattened.extend(get_args(resolved))
        else:
            flattened.append(resolved)
    return flattened


def _dedupe_args(args: list[object]) -> list[object]:
    seen: set[object] = set()
    unique: list[object] = []
    for arg in args:
        if arg in seen:
            continue
        unique.append(arg)
        seen.add(arg)
    return unique


type _UnionableType = type[object] | types.UnionType


def _coerce_unionable_args(args: list[object]) -> list[_UnionableType] | None:
    unionable: list[_UnionableType] = []
    for arg in args:
        if not _is_unionable_type(arg):
            return None
        unionable.append(arg)
    return unionable


def _is_unionable_type(value: object) -> TypeGuard[_UnionableType]:
    return isinstance(value, (type, types.UnionType))


def _resolve_saver_factory(
    *,
    saver_cls: type[AdapterCommon],
    kwargs: dict[str, ParametrizedDependency],
) -> tuple[AdapterFactory, dict[str, str], dict[str, object]]:
    adapter_kwargs = {key: value for key, value in kwargs.items() if key not in _TAG_ONLY_KWARGS}
    adapter_factory = AdapterFactory(saver_cls, **adapter_kwargs)
    dependencies, resolved_kwargs = resolve_kwargs(kwargs)
    resolved_kwargs_typed = cast("dict[str, object]", resolved_kwargs)
    return adapter_factory, dependencies, resolved_kwargs_typed


def _resolve_saver_sink(*, saver_cls: type[AdapterCommon], fn: Callable[..., object]) -> str:
    sink = saver_cls.name()
    if not isinstance(sink, str) or not sink:
        msg = f"{fn.__qualname__}: DataSaver.name() must return a non-empty string"
        raise InvalidDecoratorException(msg)
    return sink


def _resolve_output_role(
    *,
    fn: Callable[..., object],
    kwargs: dict[str, ParametrizedDependency],
    resolved_kwargs: dict[str, object],
) -> str | None:
    if "output_role" in kwargs and "output_role" not in resolved_kwargs:
        msg = (
            f"{fn.__qualname__}: output_role must be provided via value(...) so tags "
            "are available at DAG-build time."
        )
        raise InvalidDecoratorException(msg)

    output_role = resolved_kwargs.get("output_role")
    if output_role not in {None, "contract", "internal"}:
        msg = (
            f"{fn.__qualname__}: output_role must be 'contract' or 'internal'; got {output_role!r}"
        )
        raise InvalidDecoratorException(msg)
    return cast("str | None", output_role)


def _resolve_target_name(*, fn: Callable[..., object], resolved_kwargs: dict[str, object]) -> str:
    target_name = resolved_kwargs.get("target_name")
    if not isinstance(target_name, str) or not target_name:
        msg = (
            f"{fn.__qualname__}: SaveToObjectMetadataDecorator requires target_name=value(<str>) "
            "so saver tags can be derived at DAG-build time."
        )
        raise InvalidDecoratorException(msg)
    return target_name


def _resolve_output_identity(
    *,
    fn: Callable[..., object],
    resolved_kwargs: dict[str, object],
    output_role: str | None,
) -> tuple[str | None, str | None]:
    table_key = resolved_kwargs.get("table_key")
    artifact_name = resolved_kwargs.get("artifact_name")
    if isinstance(table_key, str) and table_key:
        table_key_str: str | None = table_key
    else:
        table_key_str = None

    if isinstance(artifact_name, str) and artifact_name:
        artifact_name_str: str | None = artifact_name
    else:
        artifact_name_str = None

    if output_role != "internal" and (table_key_str is None) == (artifact_name_str is None):
        msg = (
            f"{fn.__qualname__}: contract saver nodes must declare exactly one of "
            "table_key or artifact_name"
        )
        raise InvalidDecoratorException(msg)

    return table_key_str, artifact_name_str


@dataclass(frozen=True, slots=True)
class SaverTagContext:
    """Container for saver tag metadata."""

    sink: str
    saver_cls: type[AdapterCommon]
    output_role: str | None
    target_name: str
    data_node_name: str
    table_key: str | None
    artifact_name: str | None
    path_template: str | None


def _build_saver_tags(
    *,
    node_: h_node.Node,
    context: SaverTagContext,
    metadata_tags: Mapping[str, object],
) -> dict[str, object]:
    tags: dict[str, object] = {
        "hamilton.data_saver": True,
        "hamilton.data_saver.sink": context.sink,
        "hamilton.data_saver.classname": f"{context.saver_cls.__qualname__}",
        "output_role": "contract" if context.output_role is None else context.output_role,
        ht.TAG_TARGET: context.target_name,
    }
    tags.update(metadata_tags)
    tags["ci.data_node"] = context.data_node_name
    if context.table_key is not None:
        tags[ht.TAG_TABLE_KEY] = context.table_key
        tags[ht.TAG_OUTPUT_KIND] = ht.OUTPUT_KIND_TABLE
        _apply_schema_output_tag(tags, table_key=context.table_key)
        tags.setdefault(ht.TAG_MATERIALIZATION, context.sink)
        tags.setdefault(ht.TAG_MATERIALIZED_NAME, context.table_key)
    if context.artifact_name is not None:
        tags[ht.TAG_ARTIFACT] = context.artifact_name
        tags.setdefault(ht.TAG_MATERIALIZATION, context.sink)
        tags.setdefault(ht.TAG_MATERIALIZED_NAME, context.artifact_name)
    if context.path_template is not None:
        tags[ht.TAG_ARTIFACT_PATH_TEMPLATE] = context.path_template
    if isinstance(node_.tags, dict):
        domain = node_.tags.get(ht.TAG_DOMAIN)
        if isinstance(domain, str) and domain:
            tags[ht.TAG_DOMAIN] = domain
    return tags


def _apply_schema_output_tag(tags: dict[str, object], *, table_key: str) -> None:
    if _SCHEMA_OUTPUT_TAG in tags:
        return
    schema = _resolve_table_schema(table_key)
    if schema is None:
        return
    tags[_SCHEMA_OUTPUT_TAG] = {column.name: column.type for column in schema.columns}


def _resolve_table_schema(table_key: str) -> TableSchema | None:
    try:
        provider = get_schema_provider()
    except RuntimeError:
        provider = MappingSchemaProvider(OUTPUT_TABLE_SCHEMAS)
    provider = _disable_inference(provider)
    result = resolve_table_schema(table_key, schema_provider=provider)
    return result.table_schema


def _disable_inference(provider: SchemaProvider) -> SchemaProvider:
    if isinstance(provider, ResolvedSchemaProvider):
        fallback = _disable_inference(provider.fallback_provider)
        if fallback is provider.fallback_provider:
            return provider
        return ResolvedSchemaProvider(
            observation_provider=provider.observation_provider,
            fallback_provider=fallback,
        )
    if isinstance(provider, UnifiedSchemaProvider):
        return UnifiedSchemaProvider(
            declared=declared_schema_provider(),
            schema_index=provider.schema_index,
            allow_inference=False,
            fallback_to_override_on_error=provider.fallback_to_override_on_error,
        )
    if isinstance(provider, _SupportsInference):
        return provider.with_inference(allow_inference=False)
    return provider


def _resolve_metadata_tags(
    *,
    fn: Callable[..., object],
    resolved_kwargs: dict[str, object],
) -> dict[str, object]:
    tags: dict[str, object] = {}
    for key, tag_name in _METADATA_TAGS.items():
        value = resolved_kwargs.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            msg = f"{fn.__qualname__}: {key} must be a non-empty string"
            raise InvalidDecoratorException(msg)
        if key == "validation_profile" and value not in _VALIDATION_PROFILES:
            msg = (
                f"{fn.__qualname__}: validation_profile must be one of "
                f"{sorted(_VALIDATION_PROFILES)}"
            )
            raise InvalidDecoratorException(msg)
        tags[tag_name] = value
    return tags


def _resolve_artifact_path_template(
    *,
    fn: Callable[..., object],
    resolved_kwargs: dict[str, object],
    artifact_name: str | None,
) -> str | None:
    path_template = resolved_kwargs.get("path_template")
    if artifact_name is None:
        return None
    if not isinstance(path_template, str) or not path_template:
        msg = (
            f"{fn.__qualname__}: artifact saver nodes must provide "
            "path_template=value(<str>) so artifact paths are DAG-derived."
        )
        raise InvalidDecoratorException(msg)
    validate_path_template(path_template)
    return path_template


def _coerce_materialization_result(
    metadata: object,
    *,
    table_key: str | None,
    artifact_name: str | None,
    saver_name: str,
) -> MaterializationResult:
    if isinstance(metadata, MaterializationResult):
        return metadata
    if isinstance(metadata, Mapping):
        return MaterializationResult.from_mapping(
            cast("Mapping[str, object]", metadata),
            default_table_key=table_key,
            default_artifact_name=artifact_name,
        )
    msg = (
        "SaveToObjectMetadataDecorator expected MaterializationResult or mapping from "
        f"{saver_name}, got {type(metadata).__name__}"
    )
    raise TypeError(msg)
