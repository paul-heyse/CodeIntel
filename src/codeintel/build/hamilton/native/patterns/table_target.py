"""Spec-driven helpers for dataset-backed table targets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import cast

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    DatasetSaveSpec,
    SaverContext,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.nodes.module_attach import tagged_attach_node
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tag_spec import TagKey, TagSpec, TagValue
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput


@dataclass(frozen=True, slots=True)
class TableTargetTableSpec:
    """Specification for a table output in a table-backed target."""

    table_key: str
    base_node: str
    contract: TableContractSpec | None = None
    save_spec: DatasetSaveSpec | None = None
    node_name: str | None = None
    input_type: object | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None


@dataclass(frozen=True, slots=True)
class TableTargetSpec:
    """Specification for dataset-backed table targets."""

    domain: str
    target_name: str
    tables: tuple[TableTargetTableSpec, ...]
    spec: TargetSpecDescriptor | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None
    table_materializations_node: str | None = None
    anchor_node_name: str | None = None


@dataclass(frozen=True, slots=True)
class _TemplateContext:
    module: ModuleType
    domain: str
    target_name: str
    extra_tags: Mapping[TagKey, TagValue] | None


def attach_table_target_template(module: ModuleType, *, spec: TableTargetSpec) -> None:
    """Attach a table-backed target scaffold to a module.

    This helper generates per-table saver nodes, a materialization collector,
    and a target anchor using the provided spec.

    Raises
    ------
    ValueError
        If the spec is missing tables or table specs are inconsistent.
    """
    if not spec.tables:
        msg = f"{spec.target_name} must declare at least one table"
        raise ValueError(msg)

    context = _TemplateContext(
        module=module,
        domain=spec.domain,
        target_name=spec.target_name,
        extra_tags=spec.extra_tags,
    )

    table_keys: list[str] = []
    table_nodes: set[str] = set()
    for table_spec in spec.tables:
        _validate_table_spec(table_spec)
        table_key = table_spec.table_key
        if table_key in table_keys:
            msg = f"Duplicate table_key in {spec.target_name}: {table_key}"
            raise ValueError(msg)
        table_keys.append(table_key)

        node_name = _resolve_table_node_name(table_spec)
        if node_name in table_nodes:
            msg = f"Duplicate table node name in {spec.target_name}: {node_name}"
            raise ValueError(msg)
        table_nodes.add(node_name)

        _attach_table_node(
            context=context,
            table_spec=table_spec,
            node_name=node_name,
        )

    collector_node = spec.table_materializations_node or f"{spec.target_name}__table_materializations"
    table_collector = make_table_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        table_keys=table_keys,
        node_name=collector_node,
    )
    tagged_attach_node(
        module,
        node_name=collector_node,
        fn=table_collector,
        tag_spec=TagSpec.for_helper(domain=spec.domain, target=spec.target_name),
    )

    anchor_node = spec.anchor_node_name or f"t__{spec.target_name}"
    anchor_fn = _build_anchor(spec=spec, table_collector_node=collector_node, node_name=anchor_node)
    tagged_attach_node(
        module,
        node_name=anchor_node,
        fn=anchor_fn,
        tag_spec=TagSpec.for_materialize(domain=spec.domain, target=spec.target_name),
    )


def _attach_table_node(
    *,
    context: _TemplateContext,
    table_spec: TableTargetTableSpec,
    node_name: str,
) -> None:
    table_key = table_spec.table_key
    base_node = table_spec.base_node
    input_type = _resolve_input_type(table_spec)

    def table_fn(**kwargs: object) -> object:
        if base_node not in kwargs:
            msg = f"Missing dependency {base_node} for {table_key}"
            raise ValueError(msg)
        return kwargs[base_node]

    signature = inspect.Signature(
        [
            inspect.Parameter(
                base_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=input_type,
            )
        ],
        return_annotation=input_type,
    )
    table_fn = set_signature(table_fn, signature)
    table_fn.__name__ = node_name
    table_fn.__doc__ = f"Persist {table_key} rows."

    save_spec = _resolve_save_spec(table_spec)
    merged_tags = _merge_tags(context.extra_tags, table_spec.extra_tags)
    saver_context = SaverContext(
        domain=context.domain,
        target=context.target_name,
        extra_tags=merged_tags,
    )

    decorated = table_fn
    if table_spec.contract is not None:
        decorated = table_contract(table_spec.contract)(decorated)
    decorated = save_dataset(context=saver_context, spec=save_spec)(decorated)

    tagged_attach_node(
        context.module,
        node_name=node_name,
        fn=decorated,
        tag_spec=TagSpec.for_dataset(
            domain=context.domain,
            target=context.target_name,
            table_key=table_key,
            extra_tags=saver_context.extra_tags,
        ),
    )


def _build_anchor(
    *,
    spec: TableTargetSpec,
    table_collector_node: str,
    node_name: str,
) -> Callable[..., TargetRunRecord]:
    def anchor_fn(**kwargs: object) -> TargetRunRecord:
        env = kwargs.get("env")
        catalog = kwargs.get("catalog")
        if not isinstance(env, BuildEnv):
            msg = "Missing BuildEnv for target anchor"
            raise TypeError(msg)
        if not isinstance(catalog, DagCatalog):
            msg = "Missing DagCatalog for target anchor"
            raise TypeError(msg)
        materializations = kwargs.get(table_collector_node)
        if not isinstance(materializations, Mapping):
            msg = f"Missing table materializations for {spec.target_name}"
            raise TypeError(msg)

        context = MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=spec.target_name,
        )
        return record_from_materializations(
            context=context,
            artifact_materializations=None,
            table_materializations=cast("Mapping[str, MaterializationResult]", materializations),
        )

    anchor_fn = set_signature(
        anchor_fn,
        inspect.Signature(
            [
                inspect.Parameter(
                    "env",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=BuildEnv,
                ),
                inspect.Parameter(
                    "catalog",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=DagCatalog,
                ),
                inspect.Parameter(
                    table_collector_node,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=dict[str, MaterializationResult],
                ),
            ],
            return_annotation=TargetRunRecord,
        ),
    )
    anchor_fn.__name__ = node_name
    anchor_fn.__doc__ = f"Finalize {spec.target_name} target materialization."
    return codeintel_target(
        domain=spec.domain,
        target=spec.target_name,
        spec=spec.spec,
    )(anchor_fn)


def _resolve_table_node_name(table_spec: TableTargetTableSpec) -> str:
    if table_spec.node_name is not None:
        return table_spec.node_name
    base_node = table_spec.base_node
    if base_node.endswith("__base"):
        return f"{base_node[:-6]}__table"
    return f"{base_node}__table"


def _resolve_save_spec(table_spec: TableTargetTableSpec) -> DatasetSaveSpec:
    if table_spec.save_spec is None:
        return DatasetSaveSpec(table_key=table_spec.table_key)
    if table_spec.save_spec.table_key != table_spec.table_key:
        msg = (
            "DatasetSaveSpec table_key mismatch: "
            f"{table_spec.save_spec.table_key} != {table_spec.table_key}"
        )
        raise ValueError(msg)
    return table_spec.save_spec


def _resolve_input_type(table_spec: TableTargetTableSpec) -> object:
    if table_spec.input_type is not None:
        return table_spec.input_type
    if table_spec.contract is None:
        return InferableTabularInput
    return pl.LazyFrame


def _merge_tags(
    target_tags: Mapping[TagKey, TagValue] | None,
    table_tags: Mapping[TagKey, TagValue] | None,
) -> Mapping[TagKey, TagValue] | None:
    if not target_tags and not table_tags:
        return None
    merged: dict[TagKey, TagValue] = {}
    if target_tags:
        merged.update({**target_tags})
    if table_tags:
        merged.update({**table_tags})
    return merged


def _validate_table_spec(table_spec: TableTargetTableSpec) -> None:
    if not table_spec.table_key:
        msg = "TableTargetTableSpec.table_key is required"
        raise ValueError(msg)
    if not table_spec.base_node:
        msg = f"TableTargetTableSpec.base_node is required for {table_spec.table_key}"
        raise ValueError(msg)
    if table_spec.contract is None:
        return
    if table_spec.contract.table_key != table_spec.table_key:
        msg = (
            "TableContractSpec table_key mismatch: "
            f"{table_spec.contract.table_key} != {table_spec.table_key}"
        )
        raise ValueError(msg)


__all__ = [
    "TableTargetSpec",
    "TableTargetTableSpec",
    "attach_table_target_template",
]
