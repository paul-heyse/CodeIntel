"""Canonical decorator factories for table input/output variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType

from hamilton.function_modifiers import pipe_input, resolve_from_config, step, value

from codeintel.build.hamilton.transforms.tabular_steps import (
    _clip_numeric,
    _drop_bad_rows_pandas,
    _drop_bad_rows_polars,
    _drop_bad_rows_polars_lazy,
    _normalize_nulls,
)
from codeintel.build.hamilton.transforms.with_columns_backend import select_with_columns

DecoratorFactory = Callable[[Callable[..., object]], Callable[..., object]]


def _pipe_cleaning(
    df_backend: str,
    clean_mode: str,
    null_policy: str,
    max_loc_clip: int,
) -> DecoratorFactory:
    if clean_mode == "off":
        return lambda fn: fn
    if df_backend == "polars_lazy":
        drop = _drop_bad_rows_polars_lazy
    elif df_backend == "polars":
        drop = _drop_bad_rows_polars
    else:
        drop = _drop_bad_rows_pandas
    return pipe_input(
        step(drop, required_cols=value(("loc", "cyclo"))).when(clean_mode="strict"),
        step(_normalize_nulls, policy=value(null_policy)).named("nulls", namespace="prep"),
        step(_clip_numeric, col=value("loc"), max_value=value(max_loc_clip)).named(
            "loc_clip",
            namespace="prep",
        ),
        on_input="df",
        namespace="prep",
    )


def pipe_clean_df() -> DecoratorFactory:
    """Return a config-driven pipe_input decorator for cleaning steps.

    Returns
    -------
    DecoratorFactory
        Decorator that wires the cleaning steps for configured backends.
    """
    return resolve_from_config(decorate_with=_pipe_cleaning)


def _decorate_features(
    *,
    df_backend: str,
    feature_sets: dict[str, tuple[str, ...]],
    table_key: str,
    columns_to_pass: tuple[str, ...],
    ops_module: ModuleType,
) -> DecoratorFactory:
    selected = feature_sets.get(table_key, ())
    if not selected:
        return lambda fn: fn
    with_columns = select_with_columns(df_backend)
    return with_columns(
        ops_module,
        columns_to_pass=list(columns_to_pass),
        select=list(selected),
        namespace=f"feat__{table_key}",
    )


def with_features(
    *,
    table_key: str,
    columns_to_pass: Sequence[str],
    ops_module: ModuleType,
) -> DecoratorFactory:
    """Return a resolve_from_config decorator for feature column subDAGs.

    Returns
    -------
    DecoratorFactory
        Decorator that injects configured feature column operations.
    """

    def _factory(df_backend: str, feature_sets: dict[str, tuple[str, ...]]) -> DecoratorFactory:
        return _decorate_features(
            df_backend=df_backend,
            feature_sets=feature_sets,
            table_key=table_key,
            columns_to_pass=tuple(columns_to_pass),
            ops_module=ops_module,
        )

    return resolve_from_config(decorate_with=_factory)


__all__ = ["pipe_clean_df", "with_features"]
