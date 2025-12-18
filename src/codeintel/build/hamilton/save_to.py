"""Project-local Hamilton saver decorator variants.

Hamilton's built-in ``SaveToDecorator`` always types saver metadata nodes as
``typing.Dict[str, typing.Any]``. This repo uses a stricter contract: saver metadata is treated
as ``MaterializationMetadata`` at the DAG boundary, with typed schemas (e.g.,
``FileArtifactMaterializationMetadata``) parsing the dict downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import hamilton.node as h_node
from hamilton.function_modifiers.adapters import (
    AdapterFactory,
    resolve_adapter_class,
    resolve_kwargs,
)
from hamilton.function_modifiers.base import InvalidDecoratorException, SingleNodeNodeTransformer
from hamilton.node import DependencyType

from codeintel.build.hamilton.boundary_types import MaterializationMetadata

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from hamilton.function_modifiers.dependencies import ParametrizedDependency
    from hamilton.io.data_adapters import AdapterCommon, DataSaver


class SaveToObjectMetadataDecorator(SingleNodeNodeTransformer):
    """Save-to decorator that types metadata nodes as ``MaterializationMetadata``.

    Use this when downstream nodes expect ``MaterializationMetadata`` metadata mappings and you want
    to avoid ``Any`` in the DAG type system.
    """

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
            A metadata node that executes the saver and returns a metadata mapping.

        Raises
        ------
        InvalidDecoratorException
            If no saver class can handle the upstream node's output type.
        """
        artifact_name = self.artifact_name
        artifact_namespace: tuple[str, ...] = ()
        node_to_save = node_.name if self.target is None else self.target
        node_to_save_str = str(node_to_save)

        if artifact_name is None:
            artifact_name = node_to_save_str
            artifact_namespace = ("save",)
        artifact_name_str = str(artifact_name)

        saver_cls = resolve_adapter_class(node_.type, list(self.saver_classes))
        if saver_cls is None:
            msg = f"No saver class found for type: {node_.type!r} (fn={fn.__qualname__})"
            raise InvalidDecoratorException(msg)

        adapter_factory = AdapterFactory(saver_cls, **self.kwargs)
        dependencies, resolved_kwargs = resolve_kwargs(self.kwargs)
        dependencies_inverted = {v: k for k, v in dependencies.items()}
        resolved_kwargs_typed = cast("dict[str, object]", resolved_kwargs)

        def save_data(
            __adapter_factory: AdapterFactory = adapter_factory,
            __dependencies: dict[str, str] = dependencies_inverted,
            __resolved_kwargs: dict[str, object] = resolved_kwargs_typed,
            __data_node_name: str = node_to_save_str,
            /,
            **input_kwargs: object,
        ) -> MaterializationMetadata:
            input_args_with_fixed_dependencies = {
                __dependencies.get(key, key): value for key, value in input_kwargs.items()
            }
            merged_kwargs = {**__resolved_kwargs, **input_args_with_fixed_dependencies}
            data_to_save = merged_kwargs[__data_node_name]
            saver_kwargs = {k: v for k, v in merged_kwargs.items() if k != __data_node_name}
            data_saver = __adapter_factory.create_saver(**saver_kwargs)
            metadata = data_saver.save_data(data_to_save)
            return dict(metadata)

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
            key: value for key, value in input_types.items() if key not in resolved_kwargs
        }
        input_types[node_to_save_str] = (node_.type, DependencyType.REQUIRED)

        return h_node.Node(
            name=artifact_name_str,
            callabl=save_data,
            typ=cast("type[object]", MaterializationMetadata),
            input_types=input_types,
            namespace=artifact_namespace,
            tags={
                "hamilton.data_saver": True,
                "hamilton.data_saver.sink": f"{saver_cls.name()}",
                "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
            },
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
