"""Helpers for collecting materialization metadata in native targets."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.tagging import tag_helper

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _collector_signature(param_names: Sequence[str]) -> inspect.Signature:
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=MaterializationMetadata,
        )
        for name in param_names
    ]
    return inspect.Signature(
        params,
        return_annotation=dict[str, MaterializationMetadata],
    )


def _build_collector(
    *,
    domain: str,
    target: str,
    node_name: str,
    mapping: Mapping[str, str],
) -> Callable[..., dict[str, MaterializationMetadata]]:
    def collector(**kwargs: object) -> dict[str, MaterializationMetadata]:
        result: dict[str, MaterializationMetadata] = {}
        for key, param_name in mapping.items():
            if param_name not in kwargs:
                msg = f"Missing materialization metadata for {key}"
                raise ValueError(msg)
            result[key] = cast("MaterializationMetadata", kwargs[param_name])
        return result

    collector.__name__ = node_name
    collector.__doc__ = f"Collect materialization metadata for {target}."
    tagged = tag_helper(domain=domain, target=target)(collector)
    return set_signature(tagged, _collector_signature(tuple(mapping.values())))


def make_artifact_materializations_collector(
    *,
    domain: str,
    target: str,
    artifacts: Sequence[str],
    node_name: str | None = None,
) -> Callable[..., dict[str, MaterializationMetadata]]:
    """Return a collector for artifact saver metadata."""
    mapping = {
        artifact: materialize_node(f"artifact.{artifact}") for artifact in artifacts
    }
    return _build_collector(
        domain=domain,
        target=target,
        node_name=node_name or f"{target}__materializations",
        mapping=mapping,
    )


def make_table_materializations_collector(
    *,
    domain: str,
    target: str,
    table_keys: Sequence[str],
    node_name: str | None = None,
) -> Callable[..., dict[str, MaterializationMetadata]]:
    """Return a collector for table saver metadata."""
    mapping = {table_key: materialize_node(table_key) for table_key in table_keys}
    return _build_collector(
        domain=domain,
        target=target,
        node_name=node_name or f"{target}__table_materializations",
        mapping=mapping,
    )


def make_mixed_materializations_collector(
    *,
    domain: str,
    target: str,
    artifacts: Sequence[str],
    table_keys: Sequence[str],
    node_name: str | None = None,
) -> Callable[..., dict[str, MaterializationMetadata]]:
    """Return a collector for mixed artifact/table saver metadata."""
    mapping = {
        artifact: materialize_node(f"artifact.{artifact}") for artifact in artifacts
    }
    mapping.update({table_key: materialize_node(table_key) for table_key in table_keys})
    return _build_collector(
        domain=domain,
        target=target,
        node_name=node_name or f"{target}__materializations",
        mapping=mapping,
    )


__all__ = [
    "make_artifact_materializations_collector",
    "make_mixed_materializations_collector",
    "make_table_materializations_collector",
]
