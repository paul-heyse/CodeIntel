"""Project-local Hamilton saver decorator variants.

Hamilton's built-in ``SaveToDecorator`` always types saver metadata nodes as
``typing.Dict[str, typing.Any]``. This repo uses a stricter contract: saver metadata is treated
as ``MaterializationResult`` at the DAG boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
from codeintel.core.hamilton import tags as ht

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
}

_VALIDATION_PROFILES: frozenset[str] = frozenset({"strict", "lenient"})

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from hamilton.function_modifiers.dependencies import ParametrizedDependency
    from hamilton.io.data_adapters import AdapterCommon, DataSaver


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
        node_to_save_str = str(node_.name if self.target is None else self.target)
        metadata_node_name, metadata_namespace = _resolve_metadata_node_name(
            node_to_save=node_to_save_str,
            output_name=self.artifact_name,
        )
        saver_cls = _resolve_saver_class(
            node_type=node_.type,
            saver_classes=self.saver_classes,
            fn=fn,
        )
        adapter_factory, dependencies, resolved_kwargs_typed = _resolve_saver_factory(
            saver_cls=saver_cls,
            kwargs=self.kwargs,
        )
        dependencies_inverted = {v: k for k, v in dependencies.items()}
        sink = _resolve_saver_sink(saver_cls=saver_cls, fn=fn)
        output_role = _resolve_output_role(
            fn=fn,
            kwargs=self.kwargs,
            resolved_kwargs=resolved_kwargs_typed,
        )
        target_name = _resolve_target_name(fn=fn, resolved_kwargs=resolved_kwargs_typed)
        table_key, artifact_name = _resolve_output_identity(
            fn=fn,
            resolved_kwargs=resolved_kwargs_typed,
            output_role=output_role,
        )
        path_template = _resolve_artifact_path_template(
            fn=fn,
            resolved_kwargs=resolved_kwargs_typed,
            artifact_name=artifact_name,
        )
        metadata_tags = _resolve_metadata_tags(fn=fn, resolved_kwargs=resolved_kwargs_typed)

        def save_data(
            __adapter_factory: AdapterFactory = adapter_factory,
            __dependencies: dict[str, str] = dependencies_inverted,
            __resolved_kwargs: dict[str, object] = resolved_kwargs_typed,
            __data_node_name: str = node_to_save_str,
            __table_key: str | None = table_key,
            __artifact_name: str | None = artifact_name,
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

        def get_input_type_key(key: str) -> str:
            return dependencies.get(key, key)

        input_types: dict[str, type | tuple[type, DependencyType]] = {
            get_input_type_key(key): (type_, DependencyType.REQUIRED)
            for key, type_ in saver_cls.get_required_arguments().items()
        }
        input_types.update(
            {
                dependencies[key]: (type_, DependencyType.OPTIONAL)
                for key, type_ in saver_cls.get_optional_arguments().items()
                if key in dependencies
            }
        )
        input_types = {
            key: value for key, value in input_types.items() if key not in resolved_kwargs_typed
        }
        input_types[node_to_save_str] = (node_.type, DependencyType.REQUIRED)

        return h_node.Node(
            name=metadata_node_name,
            callabl=save_data,
            typ=cast("type[object]", MaterializationResult),
            input_types=input_types,
            namespace=metadata_namespace,
            tags=_build_saver_tags(
                node_=node_,
                context=SaverTagContext(
                    sink=sink,
                    saver_cls=saver_cls,
                    output_role=output_role,
                    target_name=target_name,
                    table_key=table_key,
                    artifact_name=artifact_name,
                    path_template=path_template,
                ),
                metadata_tags=metadata_tags,
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
    node_type: type[object],
    saver_classes: Sequence[type[AdapterCommon]],
    fn: Callable[..., object],
) -> type[AdapterCommon]:
    node_type_cast = cast("type[type]", node_type)
    saver_cls = resolve_adapter_class(node_type_cast, list(saver_classes))
    if saver_cls is None:
        msg = f"No saver class found for type: {node_type!r} (fn={fn.__qualname__})"
        raise InvalidDecoratorException(msg)
    return saver_cls


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
    if context.table_key is not None:
        tags[ht.TAG_TABLE_KEY] = context.table_key
    if context.artifact_name is not None:
        tags[ht.TAG_ARTIFACT] = context.artifact_name
    if context.path_template is not None:
        tags[ht.TAG_ARTIFACT_PATH_TEMPLATE] = context.path_template
    if isinstance(node_.tags, dict):
        domain = node_.tags.get(ht.TAG_DOMAIN)
        if isinstance(domain, str) and domain:
            tags[ht.TAG_DOMAIN] = domain
    return tags


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
