"""Spec-driven helpers for dataset-backed table targets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pyarrow as pa

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
    RelationTableSaveSpec,
    SaverContext,
    save_dataset,
    save_relation_table,
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
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from codeintel.build.hamilton.native.patterns.specs import OutputRole
    from codeintel.core.validation.profiles import ValidationProfile


@dataclass(frozen=True, slots=True)
class TableTargetTableSpec:
    """Specification for a table output in a table-backed target."""

    table_key: str
    base_node: str
    contract: TableContractSpec | None = None
    save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None
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
    attach_anchor: bool = True


@dataclass(frozen=True, slots=True)
class TableTargetContext:
    """Context for constructing single-table target specs."""

    domain: str
    target_name: str
    table_key: str
    base_node: str
    contract: TableContractSpec | None = None
    input_type: object | None = None
    save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None
    node_name: str | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None
    table_materializations_node: str | None = None
    anchor_node_name: str | None = None
    attach_anchor: bool = True

    @classmethod
    def from_contract(
        cls,
        *,
        contract: TableContractSpec,
        input_type: object | None = None,
        save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None,
        node_name: str | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TableTargetContext:
        """Build a target context from a contract spec.

        Returns
        -------
        TableTargetContext
            Context derived from the contract input name.
        """
        return cls(
            domain=contract.domain,
            target_name=contract.target,
            table_key=contract.table_key,
            base_node=contract.input_name,
            contract=contract,
            input_type=input_type,
            save_spec=save_spec,
            node_name=node_name,
            extra_tags=extra_tags,
        )

    @staticmethod
    def build_dataset_table_spec(
        *,
        context: TableTargetContext,
        save_options: DatasetSaveSpecOptions | None = None,
    ) -> TableTargetSpec:
        """Build a dataset-backed TableTargetSpec for a single table.

        Returns
        -------
        TableTargetSpec
            Standardized target spec configured for a single table output.
        """
        resolved_context = context
        if context.save_spec is None:
            if save_options is None:
                save_spec = DatasetSaveSpec(table_key=context.table_key)
            else:
                save_spec = DatasetSaveSpec(
                    table_key=context.table_key,
                    partition_columns=save_options.partition_columns,
                    validation_profile=save_options.validation_profile,
                    collect_group=save_options.collect_group,
                    output_role=save_options.output_role,
                    output_name=save_options.output_name,
                )
            resolved_context = replace(context, save_spec=save_spec)
        return build_single_table_target_spec(context=resolved_context)

    @staticmethod
    def build_relation_table_spec(
        *,
        context: TableTargetContext,
        save_options: RelationTableSaveSpecOptions | None = None,
    ) -> TableTargetSpec:
        """Build a relation-backed TableTargetSpec for a single table.

        Returns
        -------
        TableTargetSpec
            Standardized target spec configured for a single table output.
        """
        resolved_context = context
        if context.save_spec is None:
            if save_options is None:
                save_spec = RelationTableSaveSpec(table_key=context.table_key)
            else:
                save_spec = RelationTableSaveSpec(
                    table_key=context.table_key,
                    validation_profile=save_options.validation_profile,
                    output_role=save_options.output_role,
                    output_name=save_options.output_name,
                )
            resolved_context = replace(context, save_spec=save_spec)
        return build_single_table_target_spec(context=resolved_context)


@dataclass(frozen=True, slots=True)
class DatasetSaveSpecOptions:
    """Options for dataset save specs."""

    partition_columns: tuple[str, ...] = ()
    validation_profile: ValidationProfile | None = None
    collect_group: str | None = None
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class RelationTableSaveSpecOptions:
    """Options for relation table save specs."""

    validation_profile: ValidationProfile | None = None
    output_role: OutputRole | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class TableTargetTableContext:
    """Context for constructing table specs inside multi-table targets."""

    table_key: str
    base_node: str
    contract: TableContractSpec | None = None
    save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None
    node_name: str | None = None
    input_type: object | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None

    @classmethod
    def from_contract(
        cls,
        *,
        contract: TableContractSpec,
        node_name: str | None = None,
        save_spec: DatasetSaveSpec | RelationTableSaveSpec | None = None,
        input_type: object | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TableTargetTableContext:
        """Build a table context from a contract spec.

        Returns
        -------
        TableTargetTableContext
            Context derived from the contract input name.
        """
        return cls(
            table_key=contract.table_key,
            base_node=contract.input_name,
            contract=contract,
            save_spec=save_spec,
            node_name=node_name,
            input_type=input_type,
            extra_tags=extra_tags,
        )


@dataclass(frozen=True, slots=True)
class MultiTableTargetContext:
    """Context for constructing multi-table target specs."""

    domain: str
    target_name: str
    tables: tuple[TableTargetTableSpec, ...]
    spec: TargetSpecDescriptor | None = None
    extra_tags: Mapping[TagKey, TagValue] | None = None
    table_materializations_node: str | None = None
    anchor_node_name: str | None = None
    attach_anchor: bool = True
    save_spec_factory: Callable[[str], DatasetSaveSpec | RelationTableSaveSpec] | None = None
    default_input_type: object | None = None

    @staticmethod
    def build_table_spec(
        *,
        context: TableTargetTableContext,
        save_spec_factory: Callable[[str], DatasetSaveSpec | RelationTableSaveSpec] | None = None,
        default_input_type: object | None = None,
    ) -> TableTargetTableSpec:
        """Build a table spec for a multi-table target.

        Returns
        -------
        TableTargetTableSpec
            Table specification derived from the provided context.
        """
        resolved_save_spec = context.save_spec
        if resolved_save_spec is None and save_spec_factory is not None:
            resolved_save_spec = save_spec_factory(context.table_key)
        if resolved_save_spec is None:
            resolved_save_spec = DatasetSaveSpec(table_key=context.table_key)
        resolved_input_type = context.input_type or default_input_type
        return TableTargetTableSpec(
            table_key=context.table_key,
            base_node=context.base_node,
            contract=context.contract,
            save_spec=resolved_save_spec,
            node_name=context.node_name,
            input_type=resolved_input_type,
            extra_tags=context.extra_tags,
        )

    @staticmethod
    def build_dataset_table_spec(
        *,
        context: TableTargetTableContext,
        save_options: DatasetSaveSpecOptions | None = None,
        default_input_type: object | None = None,
    ) -> TableTargetTableSpec:
        """Build a dataset table spec for a multi-table target.

        Returns
        -------
        TableTargetTableSpec
            Table specification derived from the provided context.
        """
        resolved_context = context
        if context.save_spec is None:
            if save_options is None:
                save_spec = DatasetSaveSpec(table_key=context.table_key)
            else:
                save_spec = DatasetSaveSpec(
                    table_key=context.table_key,
                    partition_columns=save_options.partition_columns,
                    validation_profile=save_options.validation_profile,
                    collect_group=save_options.collect_group,
                    output_role=save_options.output_role,
                    output_name=save_options.output_name,
                )
            resolved_context = replace(context, save_spec=save_spec)
        return MultiTableTargetContext.build_table_spec(
            context=resolved_context,
            default_input_type=default_input_type,
        )

    @staticmethod
    def build_relation_table_spec(
        *,
        context: TableTargetTableContext,
        save_options: RelationTableSaveSpecOptions | None = None,
        default_input_type: object | None = None,
    ) -> TableTargetTableSpec:
        """Build a relation table spec for a multi-table target.

        Returns
        -------
        TableTargetTableSpec
            Table specification derived from the provided context.
        """
        resolved_context = context
        if context.save_spec is None:
            if save_options is None:
                save_spec = RelationTableSaveSpec(table_key=context.table_key)
            else:
                save_spec = RelationTableSaveSpec(
                    table_key=context.table_key,
                    validation_profile=save_options.validation_profile,
                    output_role=save_options.output_role,
                    output_name=save_options.output_name,
                )
            resolved_context = replace(context, save_spec=save_spec)
        return MultiTableTargetContext.build_table_spec(
            context=resolved_context,
            default_input_type=default_input_type,
        )


def build_single_table_target_spec(*, context: TableTargetContext) -> TableTargetSpec:
    """Build a TableTargetSpec for a single table target.

    Returns
    -------
    TableTargetSpec
        Standardized target spec configured for a single table output.
    """
    save_spec = context.save_spec or DatasetSaveSpec(table_key=context.table_key)
    table_materializations_node = (
        context.table_materializations_node or f"{context.target_name}__table_materializations"
    )
    anchor_node_name = None
    if context.attach_anchor:
        anchor_node_name = context.anchor_node_name or f"t__{context.target_name}"
    return TableTargetSpec(
        domain=context.domain,
        target_name=context.target_name,
        tables=(
            TableTargetTableSpec(
                table_key=context.table_key,
                base_node=context.base_node,
                contract=context.contract,
                save_spec=save_spec,
                node_name=context.node_name or f"{context.target_name}__table",
                input_type=context.input_type,
                extra_tags=context.extra_tags,
            ),
        ),
        extra_tags=context.extra_tags,
        table_materializations_node=table_materializations_node,
        anchor_node_name=anchor_node_name,
        attach_anchor=context.attach_anchor,
    )


def build_multi_table_target_spec(*, context: MultiTableTargetContext) -> TableTargetSpec:
    """Build a TableTargetSpec for a multi-table target.

    Returns
    -------
    TableTargetSpec
        Standardized target spec configured for a multi-table output.
    """
    table_materializations_node = (
        context.table_materializations_node or f"{context.target_name}__table_materializations"
    )
    anchor_node_name = None
    if context.attach_anchor:
        anchor_node_name = context.anchor_node_name or f"t__{context.target_name}"
    return TableTargetSpec(
        domain=context.domain,
        target_name=context.target_name,
        tables=context.tables,
        spec=context.spec,
        extra_tags=context.extra_tags,
        table_materializations_node=table_materializations_node,
        anchor_node_name=anchor_node_name,
        attach_anchor=context.attach_anchor,
    )


def build_multi_table_target_spec_from_contexts(
    *,
    context: MultiTableTargetContext,
    table_contexts: Sequence[TableTargetTableContext],
) -> TableTargetSpec:
    """Build a TableTargetSpec from table contexts for multi-table targets.

    Returns
    -------
    TableTargetSpec
        Standardized target spec configured for a multi-table output.
    """
    tables = tuple(
        MultiTableTargetContext.build_table_spec(
            context=table_context,
            save_spec_factory=context.save_spec_factory,
            default_input_type=context.default_input_type,
        )
        for table_context in table_contexts
    )
    return build_multi_table_target_spec(context=replace(context, tables=tables))


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
    table_materialization_nodes: dict[str, str] = {}
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

        save_spec = table_spec.save_spec
        if save_spec is not None and save_spec.output_name is not None:
            table_materialization_nodes[table_key] = save_spec.output_name

        _attach_table_node(
            context=context,
            table_spec=table_spec,
            node_name=node_name,
        )

    collector_node = (
        spec.table_materializations_node or f"{spec.target_name}__table_materializations"
    )
    table_collector = make_table_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        table_keys=table_keys,
        node_name=collector_node,
        materialization_nodes=table_materialization_nodes or None,
    )
    tagged_attach_node(
        module,
        node_name=collector_node,
        fn=table_collector,
        tag_spec=TagSpec.for_helper(domain=spec.domain, target=spec.target_name),
    )

    if spec.attach_anchor:
        anchor_node = spec.anchor_node_name or f"t__{spec.target_name}"
        anchor_fn = _build_anchor(
            spec=spec,
            table_collector_node=collector_node,
            node_name=anchor_node,
        )
        anchor_tags: dict[TagKey, TagValue] = {
            cast("TagKey", ht.TAG_KIND): "target",
            cast("TagKey", ht.TAG_SCHEMA_REF): spec.target_name,
        }
        tagged_attach_node(
            module,
            node_name=anchor_node,
            fn=anchor_fn,
            tag_spec=TagSpec.for_materialize(domain=spec.domain, target=spec.target_name),
            extra_tags=anchor_tags,
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
    contract_tags = _contract_tags(table_spec)
    merged_tags = _merge_tags(merged_tags, contract_tags)
    saver_context = SaverContext(
        domain=context.domain,
        target=context.target_name,
        extra_tags=merged_tags,
    )

    decorated = table_fn
    if table_spec.contract is not None:
        decorated = table_contract(table_spec.contract)(decorated)
    if isinstance(save_spec, RelationTableSaveSpec):
        decorated = save_relation_table(context=saver_context, spec=save_spec)(decorated)
    else:
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


def _resolve_save_spec(
    table_spec: TableTargetTableSpec,
) -> DatasetSaveSpec | RelationTableSaveSpec:
    if table_spec.save_spec is None:
        return DatasetSaveSpec(table_key=table_spec.table_key)
    if table_spec.save_spec.table_key != table_spec.table_key:
        msg = (
            "SaveSpec table_key mismatch: "
            f"{table_spec.save_spec.table_key} != {table_spec.table_key}"
        )
        raise ValueError(msg)
    return table_spec.save_spec


def _resolve_input_type(table_spec: TableTargetTableSpec) -> object:
    if table_spec.input_type is not None:
        if table_spec.contract is not None and table_spec.input_type is pa.Table:
            return InferableTabularInput
        return table_spec.input_type
    if table_spec.contract is None:
        return InferableTabularInput
    return InferableTabularInput


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


def _contract_tags(
    table_spec: TableTargetTableSpec,
) -> Mapping[TagKey, TagValue] | None:
    contract = table_spec.contract
    if contract is None:
        return None
    tags: dict[TagKey, TagValue] = {}
    if contract.contract_version:
        tags[cast("TagKey", ht.TAG_CONTRACT_VERSION)] = contract.contract_version
    if contract.contract_hash:
        tags[cast("TagKey", ht.TAG_CONTRACT_HASH)] = contract.contract_hash
    if not tags:
        return None
    return tags


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
    "DatasetSaveSpecOptions",
    "MultiTableTargetContext",
    "RelationTableSaveSpecOptions",
    "TableTargetContext",
    "TableTargetSpec",
    "TableTargetTableContext",
    "TableTargetTableSpec",
    "attach_table_target_template",
    "build_multi_table_target_spec",
    "build_multi_table_target_spec_from_contexts",
    "build_single_table_target_spec",
]
