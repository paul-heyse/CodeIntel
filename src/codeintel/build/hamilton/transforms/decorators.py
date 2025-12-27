"""Canonical decorator factories for table input/output variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

from hamilton.function_modifiers import pipe_input, resolve_from_config, step, value
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.hamilton.transforms.tabular_steps import (
    clip_numeric_pandas,
    clip_numeric_polars,
    clip_numeric_polars_lazy,
    drop_bad_rows_pandas,
    drop_bad_rows_polars,
    drop_bad_rows_polars_lazy,
    normalize_nulls_pandas,
    normalize_nulls_polars,
    normalize_nulls_polars_lazy,
)
from codeintel.build.hamilton.transforms.with_columns_backend import select_with_columns


@dataclass(frozen=True, slots=True)
class _CleaningPolicy:
    required_cols: tuple[str, ...]
    clip_column: str | None
    input_name: str


class _NoOpTransform(NodeTransformLifecycle):
    """No-op decorator used to satisfy resolve_from_config when disabled."""

    @classmethod
    def get_lifecycle_name(cls) -> str:
        return "codeintel_noop"

    @classmethod
    def allows_multiple(cls) -> bool:
        return True

    def validate(self, fn: Callable[..., object]) -> None:
        _ = (self, fn)

    def __call__(self, fn: Callable[..., object]) -> Callable[..., object]:
        return fn


def _pipe_cleaning(
    df_backend: str,
    clean_mode: str,
    null_policy: str,
    max_loc_clip: int,
    policy: _CleaningPolicy,
) -> NodeTransformLifecycle:
    if clean_mode == "off":
        return _NoOpTransform()
    if df_backend == "polars_lazy":
        drop = drop_bad_rows_polars_lazy
        normalize = normalize_nulls_polars_lazy
        clip = clip_numeric_polars_lazy
    elif df_backend == "polars":
        drop = drop_bad_rows_polars
        normalize = normalize_nulls_polars
        clip = clip_numeric_polars
    else:
        drop = drop_bad_rows_pandas
        normalize = normalize_nulls_pandas
        clip = clip_numeric_pandas
    steps = [
        step(drop, required_cols=value(policy.required_cols)).when(clean_mode="strict"),
        step(normalize, policy=value(null_policy)).named("nulls", namespace="prep"),
    ]
    if policy.clip_column is not None:
        steps.append(
            step(
                clip,
                col=value(policy.clip_column),
                max_value=value(max_loc_clip),
            ).named("loc_clip", namespace="prep")
        )
    return pipe_input(*steps, on_input=policy.input_name, namespace="prep")


def pipe_clean_df(
    *,
    required_cols: Sequence[str] = ("loc", "cyclo"),
    clip_column: str | None = "loc",
    input_name: str = "df",
) -> NodeTransformLifecycle:
    """Return a config-driven pipe_input decorator for cleaning steps.

    Parameters
    ----------
    required_cols
        Required columns used for strict-mode row filtering.
    clip_column
        Optional column name to apply numeric clipping.
    input_name
        Input parameter name to target for cleaning steps.

    Returns
    -------
    DecoratorFactory
        Decorator that wires the cleaning steps for configured backends.
    """
    required_cols_tuple = tuple(required_cols)
    policy = _CleaningPolicy(
        required_cols=required_cols_tuple,
        clip_column=clip_column,
        input_name=input_name,
    )

    def _factory(
        df_backend: str,
        clean_mode: str,
        null_policy: str,
        max_loc_clip: int,
    ) -> NodeTransformLifecycle:
        return _pipe_cleaning(
            df_backend,
            clean_mode,
            null_policy,
            max_loc_clip,
            policy,
        )

    return resolve_from_config(decorate_with=_factory)


def _decorate_features(
    *,
    df_backend: str,
    feature_sets: dict[str, tuple[str, ...]],
    table_key: str,
    columns_to_pass: tuple[str, ...],
    ops_module: ModuleType,
) -> NodeTransformLifecycle:
    selected = feature_sets.get(table_key, ())
    if not selected:
        return _NoOpTransform()
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
) -> NodeTransformLifecycle:
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

    def _factory(
        df_backend: str, feature_sets: dict[str, tuple[str, ...]]
    ) -> NodeTransformLifecycle:
        return _decorate_features(
            df_backend=df_backend,
            feature_sets=feature_sets,
            table_key=table_key,
            columns_to_pass=tuple(columns_to_pass),
            ops_module=ops_module,
        )

    return resolve_from_config(decorate_with=_factory)


__all__ = ["pipe_clean_df", "with_features"]
