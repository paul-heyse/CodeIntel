"""Schema inference helpers for Arrow Acero plan construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import acero

from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.primitives import TableSchema


class JoinSpec(Protocol):
    """Protocol for plan join specifications."""

    @property
    def left_keys(self) -> Sequence[str]:
        """Left join key names."""
        ...

    @property
    def right_keys(self) -> Sequence[str]:
        """Right join key names."""
        ...

    @property
    def left_output(self) -> Sequence[str] | None:
        """Left-side output column selection."""
        ...

    @property
    def right_output(self) -> Sequence[str] | None:
        """Right-side output column selection."""
        ...

    @property
    def output_suffix_for_left(self) -> str | None:
        """Suffix applied to left columns when collisions occur."""
        ...

    @property
    def output_suffix_for_right(self) -> str | None:
        """Suffix applied to right columns when collisions occur."""
        ...


class PlanLike(Protocol):
    """Protocol for plan-like objects carrying schemas."""

    @property
    def schema(self) -> pa.Schema | None:
        """Resolved schema for the plan."""
        ...


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """Resolved aggregate spec for schema inference."""

    target: object
    func: str
    options: object | None
    name: str


def infer_project_schema(
    schema: pa.Schema | None,
    expressions: Sequence[pc.Expression],
    *,
    names: Sequence[str] | None,
) -> pa.Schema | None:
    """Infer schema for a project node.

    Returns
    -------
    pa.Schema | None
        Projected schema when it can be derived.
    """
    if schema is None:
        return None
    empty = empty_table_from_schema(schema)
    decl = acero.Declaration("table_source", acero.TableSourceNodeOptions(empty))
    options = acero.ProjectNodeOptions(list(expressions), names=names)
    proj = acero.Declaration("project", options, inputs=[decl])
    try:
        return proj.to_table().schema
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
        return None


def infer_filter_schema(schema: pa.Schema | None) -> pa.Schema | None:
    """Filter nodes preserve the input schema.

    Returns
    -------
    pa.Schema | None
        The unchanged input schema.
    """
    return schema


def infer_order_by_schema(schema: pa.Schema | None) -> pa.Schema | None:
    """Order-by nodes preserve the input schema.

    Returns
    -------
    pa.Schema | None
        The unchanged input schema.
    """
    return schema


def infer_hash_join_schema(
    left: pa.Schema | None,
    right: pa.Schema | None,
    *,
    spec: JoinSpec,
) -> pa.Schema | None:
    """Infer schema for hash joins based on input schemas and output selections.

    Returns
    -------
    pa.Schema | None
        Joined schema when it can be derived.
    """
    if left is None or right is None:
        return None
    left_fields = _resolve_fields(left, output=spec.left_output)
    right_fields = _resolve_fields(right, output=spec.right_output)
    if left_fields is None or right_fields is None:
        return None
    if spec.output_suffix_for_left or spec.output_suffix_for_right:
        left_fields, right_fields = _apply_join_suffixes(
            left_fields,
            right_fields,
            left_suffix=spec.output_suffix_for_left,
            right_suffix=spec.output_suffix_for_right,
        )
    return pa.schema([*left_fields, *right_fields])


def infer_aggregate_schema(
    schema: pa.Schema | None,
    *,
    keys: Sequence[pc.Expression] | None,
    aggregates: Sequence[tuple[object, str, object | None, str]],
) -> pa.Schema | None:
    """Infer schema for aggregate nodes using Arrow group_by aggregation.

    Returns
    -------
    pa.Schema | None
        Aggregated schema when it can be derived.
    """
    if schema is None:
        return None
    key_names = _resolve_key_names(keys)
    if key_names is None:
        return None
    empty = empty_table_from_schema(schema)
    resolved = _resolve_aggregate_specs(aggregates)
    agg_specs = _group_by_specs(resolved)
    try:
        grouped = empty.group_by(list(key_names)).aggregate(agg_specs)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
        return None
    if not resolved:
        return grouped.schema
    renamed = _rename_aggregate_columns(
        grouped,
        output_names=[spec.name for spec in resolved],
        key_count=len(key_names),
    )
    return renamed.schema


def compile_plan_schema(plan: PlanLike, *, table_key: str) -> TableSchema:
    """Compile a TableSchema from a plan with an attached Arrow schema.

    Returns
    -------
    TableSchema
        Derived schema for the plan.

    Raises
    ------
    ValueError
        If the plan schema is missing.
    """
    schema = plan.schema
    if schema is None:
        msg = f"Plan schema is missing for {table_key}"
        raise ValueError(msg)
    return table_schema_from_arrow_schema(arrow_schema=schema, table_key=table_key)


def _resolve_key_names(keys: Sequence[pc.Expression] | None) -> tuple[str, ...] | None:
    if keys is None:
        return ()
    resolved: list[str] = []
    for expr in keys:
        name = _expr_field_name(expr)
        if name is None:
            return None
        resolved.append(name)
    return tuple(resolved)


def _expr_field_name(expr: pc.Expression) -> str | None:
    raw = str(expr).strip()
    if not raw:
        return None
    if _is_simple_field(raw):
        return raw
    if raw.startswith("FieldRef.Nested(") and raw.endswith(")"):
        names = _parse_nested_field(raw)
        if not names:
            return None
        return ".".join(names)
    return None


def _parse_nested_field(raw: str) -> list[str]:
    tokens: list[str] = []
    marker = "FieldRef.Name("
    start = 0
    while True:
        idx = raw.find(marker, start)
        if idx == -1:
            break
        start_idx = idx + len(marker)
        end_idx = raw.find(")", start_idx)
        if end_idx == -1:
            break
        name = raw[start_idx:end_idx].strip()
        if name:
            tokens.append(name)
        start = end_idx + 1
    return tokens


def _is_simple_field(raw: str) -> bool:
    if raw[0].isalpha() or raw[0] == "_":
        return all(char.isalnum() or char == "_" for char in raw)
    return False


def _resolve_fields(
    schema: pa.Schema,
    *,
    output: Sequence[str] | None,
) -> list[pa.Field] | None:
    names = list(output) if output is not None else list(schema.names)
    fields: list[pa.Field] = []
    for name in names:
        if name not in schema.names:
            return None
        fields.append(schema.field(name))
    return fields


def _apply_join_suffixes(
    left_fields: Sequence[pa.Field],
    right_fields: Sequence[pa.Field],
    *,
    left_suffix: str | None,
    right_suffix: str | None,
) -> tuple[list[pa.Field], list[pa.Field]]:
    left_names = [field.name for field in left_fields]
    right_names = [field.name for field in right_fields]
    duplicates = set(left_names).intersection(right_names)
    if not duplicates:
        return list(left_fields), list(right_fields)
    resolved_left: list[pa.Field] = []
    resolved_right: list[pa.Field] = []
    for field in left_fields:
        if field.name in duplicates and left_suffix:
            resolved_left.append(field.with_name(f"{field.name}{left_suffix}"))
        else:
            resolved_left.append(field)
    for field in right_fields:
        if field.name in duplicates and right_suffix:
            resolved_right.append(field.with_name(f"{field.name}{right_suffix}"))
        else:
            resolved_right.append(field)
    return resolved_left, resolved_right


def _resolve_aggregate_specs(
    aggregates: Sequence[tuple[object, str, object | None, str]],
) -> list[AggregateSpec]:
    return [
        AggregateSpec(target=target, func=func, options=options, name=name)
        for target, func, options, name in aggregates
    ]


def _group_by_specs(specs: Sequence[AggregateSpec]) -> list[tuple[object, str, object | None]]:
    resolved: list[tuple[object, str, object | None]] = []
    for spec in specs:
        target = _resolve_aggregate_target(spec.target)
        if target is None:
            continue
        resolved.append((target, spec.func, spec.options))
    return resolved


def _resolve_aggregate_target(target: object) -> object | None:
    if isinstance(target, pc.Expression):
        return _expr_field_name(target)
    if isinstance(target, str):
        return target
    if isinstance(target, (list, tuple)):
        names: list[str] = []
        for item in target:
            name = _expr_field_name(item) if isinstance(item, pc.Expression) else str(item)
            if not name:
                return None
            names.append(name)
        return names
    return None


def _rename_aggregate_columns(
    table: pa.Table,
    *,
    output_names: Sequence[str],
    key_count: int,
) -> pa.Table:
    if not output_names:
        return table
    column_names = list(table.column_names)
    agg_names = column_names[key_count:]
    if len(agg_names) != len(output_names):
        return table
    updated = column_names[:key_count] + list(output_names)
    return table.rename_columns(updated)


__all__ = [
    "AggregateSpec",
    "JoinSpec",
    "PlanLike",
    "compile_plan_schema",
    "infer_aggregate_schema",
    "infer_filter_schema",
    "infer_hash_join_schema",
    "infer_order_by_schema",
    "infer_project_schema",
]
