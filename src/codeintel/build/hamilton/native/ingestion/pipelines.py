"""Ingestion preprocessing pipelines for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ParamSpec, Protocol, TypeVar, cast

import pyarrow as pa
from hamilton.function_modifiers import mutate as h_mutate
from hamilton.function_modifiers import pipe_input, resolve_from_config, step, value
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import Plan, materialize_plan
from codeintel.build.tabular.types import InferableTabularInput

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


class _TransformCarrier(Protocol):
    transform: list[NodeTransformLifecycle]


def _drop_null_rows(
    rows: InferableTabularInput | None,
    *,
    required_cols: tuple[str, ...],
) -> pa.Table | None:
    if rows is None:
        return None
    if not required_cols:
        return tabular_to_arrow_table(rows)
    table = tabular_to_arrow_table(rows)
    required = [name for name in required_cols if name in table.schema.names]
    if not required:
        return table
    exprs = [E.is_valid(name) for name in required]
    plan = Plan.table(table).filter(E.and_(*exprs))
    return materialize_plan(plan, use_threads=True)


def _pipe_ingest_rows(
    _clean_mode: str,
    *,
    required_cols: tuple[str, ...],
    input_name: str,
) -> NodeTransformLifecycle:
    return pipe_input(
        step(_drop_null_rows, required_cols=value(required_cols)).when(clean_mode="strict"),
        on_input=input_name,
        namespace="prep",
    )


def pipe_ingest_rows(
    *,
    required_cols: Sequence[str] = (),
    input_name: str = "rows",
) -> NodeTransformLifecycle:
    """Return a resolve_from_config pipe_input decorator for ingestion frames.

    Parameters
    ----------
    required_cols
        Column names required for strict-mode filtering.
    input_name
        Input parameter name to target for ingestion row cleanup.

    Returns
    -------
    DecoratorFactory
        Decorator applying ingestion row cleanup steps.
    """
    required_tuple = tuple(str(name) for name in required_cols)

    def _factory(clean_mode: str = "lenient") -> NodeTransformLifecycle:
        return _pipe_ingest_rows(
            clean_mode,
            required_cols=required_tuple,
            input_name=input_name,
        )

    return resolve_from_config(decorate_with=_factory)


def mutate_ingest_rows(
    *targets: object,
    **mutating_fn_kwargs: object,
) -> Decorator[P, R]:
    """Return a mutate decorator that preserves save-to transform ordering.

    Parameters
    ----------
    *targets
        Mutate targets passed to Hamilton's mutate decorator.
    **mutating_fn_kwargs
        Additional keyword arguments forwarded to Hamilton mutate.

    Returns
    -------
    Decorator[P, R]
        Decorator that preserves save-to transform ordering.
    """
    mutate = cast("Callable[..., Decorator[P, R]]", h_mutate)
    decorator = mutate(*targets, **mutating_fn_kwargs)

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        result = decorator(fn)
        for target in targets:
            target_fn = _resolve_mutate_target_fn(target)
            if target_fn is not None:
                _reorder_save_to_transforms(target_fn)
        return result

    return apply


def _resolve_mutate_target_fn(target: object) -> Callable[..., object] | None:
    if callable(target):
        return target
    target_fn = getattr(target, "target_fn", None)
    if callable(target_fn):
        return target_fn
    return None


def _reorder_save_to_transforms(fn: Callable[..., object]) -> None:
    transforms = getattr(fn, "transform", None)
    if not isinstance(transforms, list):
        return
    saver_transforms = [
        transform
        for transform in transforms
        if isinstance(transform, SaveToObjectMetadataDecorator)
    ]
    if not saver_transforms:
        return
    other_transforms = [
        transform
        for transform in transforms
        if not isinstance(transform, SaveToObjectMetadataDecorator)
    ]
    carrier = cast("_TransformCarrier", fn)
    carrier.transform = other_transforms + saver_transforms


__all__ = ["mutate_ingest_rows", "pipe_ingest_rows"]
