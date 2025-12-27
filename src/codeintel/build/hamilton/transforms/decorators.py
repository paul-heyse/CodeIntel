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
    *,
    required_cols: tuple[str, ...],
    clip_column: str | None,
) -> DecoratorFactory:
    if clean_mode == "off":
        return lambda fn: fn
    if df_backend == "polars_lazy":
        drop = _drop_bad_rows_polars_lazy
    elif df_backend == "polars":
        drop = _drop_bad_rows_polars
    else:
        drop = _drop_bad_rows_pandas
    steps = [
        step(drop, required_cols=value(required_cols)).when(clean_mode="strict"),
        step(_normalize_nulls, policy=value(null_policy)).named("nulls", namespace="prep"),
    ]
    if clip_column is not None:
        steps.append(
            step(_clip_numeric, col=value(clip_column), max_value=value(max_loc_clip)).named(
                "loc_clip",
                namespace="prep",
            )
        )
    return pipe_input(*steps, on_input="df", namespace="prep")


def pipe_clean_df(
    *,
    required_cols: Sequence[str] = ("loc", "cyclo"),
    clip_column: str | None = "loc",
) -> DecoratorFactory:
    """Return a config-driven pipe_input decorator for cleaning steps.

    Parameters
    ----------
    required_cols
        Required columns used for strict-mode row filtering.
    clip_column
        Optional column name to apply numeric clipping.

    Returns
    -------
    DecoratorFactory
        Decorator that wires the cleaning steps for configured backends.
    """
    required_cols_tuple = tuple(required_cols)

    def _factory(
        df_backend: str,
        clean_mode: str,
        null_policy: str,
        max_loc_clip: int,
    ) -> DecoratorFactory:
        return _pipe_cleaning(
            df_backend,
            clean_mode,
            null_policy,
            max_loc_clip,
            required_cols=required_cols_tuple,
            clip_column=clip_column,
        )

    return resolve_from_config(decorate_with=_factory)


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

    Parameters
    ----------
    table_key
        Table key used to select configured feature ops.
    columns_to_pass
        Column names passed through to feature ops.
    ops_module
        Module containing column-op functions.

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
