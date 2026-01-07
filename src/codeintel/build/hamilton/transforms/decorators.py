"""Canonical decorator factories for table input/output variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

from hamilton.function_modifiers import (
    pipe_input,
    pipe_output,
    resolve_from_config,
    step,
    value,
)
from hamilton.function_modifiers.base import NodeTransformLifecycle

from codeintel.build.contracts.types import ContractPolicy
from codeintel.build.hamilton.transforms.tabular_steps import (
    align_contract_output,
    clip_numeric,
    drop_bad_rows,
    normalize_nulls,
    sort_columns,
)
from codeintel.build.hamilton.transforms.with_columns_backend import select_with_columns
from codeintel.build.schemas import column_order_for_table_key
from codeintel.build.tabular.types import InferableTabularInput


@dataclass(frozen=True, slots=True)
class _CleaningPolicy:
    required_cols: tuple[str, ...]
    clip_column: str | None
    input_name: str


@dataclass(frozen=True, slots=True)
class _CleaningRuntimeConfig:
    df_backend: str
    clean_mode: str
    null_policy: str
    max_loc_clip: int
    namespace: str


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


@dataclass(frozen=True, slots=True)
class _CanonicalizationPolicy:
    table_key: str
    namespace: str


@dataclass(frozen=True, slots=True)
class _CanonicalizationRuntimeConfig:
    enable_canonicalization: bool


@dataclass(frozen=True, slots=True)
class _AlignmentPolicy:
    table_key: str
    target_name: str | None
    policy: ContractPolicy | None
    namespace: str


@dataclass(frozen=True, slots=True)
class _AlignmentRuntimeConfig:
    enable_contract_alignment: bool


def _canonicalize_output(
    frame: InferableTabularInput,
    *,
    table_key: str,
) -> InferableTabularInput:
    try:
        column_order = column_order_for_table_key(table_key)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return frame
    if not column_order:
        return frame
    return sort_columns(frame, column_order)


def _pipe_canonical_output(
    config: _CanonicalizationRuntimeConfig,
    policy: _CanonicalizationPolicy,
) -> NodeTransformLifecycle:
    if not config.enable_canonicalization:
        return _NoOpTransform()
    canonical_step = step(
        _canonicalize_output,
        table_key=value(policy.table_key),
    ).named("canonicalize", namespace=policy.namespace)
    return pipe_output(canonical_step, namespace=policy.namespace)


def _pipe_contract_alignment(
    config: _AlignmentRuntimeConfig,
    policy: _AlignmentPolicy,
) -> NodeTransformLifecycle:
    if not config.enable_contract_alignment:
        return _NoOpTransform()
    alignment_step = step(
        align_contract_output,
        table_key=value(policy.table_key),
        target_name=value(policy.target_name),
        policy=value(policy.policy),
    ).named("align_contract", namespace=policy.namespace)
    return pipe_output(alignment_step, namespace=policy.namespace)


def _pipe_cleaning(
    config: _CleaningRuntimeConfig,
    policy: _CleaningPolicy,
) -> NodeTransformLifecycle:
    if config.clean_mode == "off":
        return _NoOpTransform()
    if config.df_backend not in {"polars_lazy", "arrow"}:
        msg = f"Unsupported df_backend={config.df_backend!r}"
        raise ValueError(msg)
    drop = drop_bad_rows
    normalize = normalize_nulls
    clip = clip_numeric
    steps = [
        step(drop, required_cols=value(policy.required_cols)).when(clean_mode="strict"),
        step(normalize, policy=value(config.null_policy)).named(
            "nulls",
            namespace=config.namespace,
        ),
    ]
    if policy.clip_column is not None:
        steps.append(
            step(
                clip,
                col=value(policy.clip_column),
                max_value=value(config.max_loc_clip),
            ).named("loc_clip", namespace=config.namespace)
        )
    return pipe_input(*steps, on_input=policy.input_name, namespace=config.namespace)


def pipe_clean_df(
    *,
    required_cols: Sequence[str] = ("loc", "cyclo"),
    clip_column: str | None = "loc",
    input_name: str = "df",
    namespace: str = "prep",
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
    namespace
        Namespace prefix for generated transformation nodes.

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
        df_backend: str = "polars_lazy",
        clean_mode: str = "lenient",
        null_policy: str = "preserve",
        max_loc_clip: int = 10_000,
    ) -> NodeTransformLifecycle:
        return _pipe_cleaning(
            _CleaningRuntimeConfig(
                df_backend=df_backend,
                clean_mode=clean_mode,
                null_policy=null_policy,
                max_loc_clip=max_loc_clip,
                namespace=namespace,
            ),
            policy,
        )

    return resolve_from_config(decorate_with=_factory)


def pipe_canonical_output(
    *,
    table_key: str,
    namespace: str,
) -> NodeTransformLifecycle:
    """Return a config-driven pipe_output decorator for canonicalization.

    Returns
    -------
    NodeTransformLifecycle
        Config-driven transform that applies canonicalization policies.
    """

    def _factory(
        *,
        enable_canonicalization: bool = True,
    ) -> NodeTransformLifecycle:
        return _pipe_canonical_output(
            _CanonicalizationRuntimeConfig(
                enable_canonicalization=enable_canonicalization,
            ),
            _CanonicalizationPolicy(
                table_key=table_key,
                namespace=namespace,
            ),
        )

    return resolve_from_config(decorate_with=_factory)


def pipe_contract_alignment(
    *,
    table_key: str,
    target_name: str | None,
    policy: ContractPolicy | None,
    namespace: str,
) -> NodeTransformLifecycle:
    """Return a config-driven pipe_output decorator for contract alignment.

    Returns
    -------
    NodeTransformLifecycle
        Config-driven transform that applies contract alignment.
    """

    def _factory(
        *,
        enable_contract_alignment: bool = True,
    ) -> NodeTransformLifecycle:
        return _pipe_contract_alignment(
            _AlignmentRuntimeConfig(
                enable_contract_alignment=enable_contract_alignment,
            ),
            _AlignmentPolicy(
                table_key=table_key,
                target_name=target_name,
                policy=policy,
                namespace=namespace,
            ),
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
    if df_backend == "arrow":
        msg = "with_features is not supported for df_backend='arrow'"
        raise ValueError(msg)
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
        df_backend: str = "polars_lazy",
        feature_sets: dict[str, tuple[str, ...]] | None = None,
    ) -> NodeTransformLifecycle:
        return _decorate_features(
            df_backend=df_backend,
            feature_sets=feature_sets or {},
            table_key=table_key,
            columns_to_pass=tuple(columns_to_pass),
            ops_module=ops_module,
        )

    return resolve_from_config(decorate_with=_factory)


__all__ = [
    "pipe_canonical_output",
    "pipe_clean_df",
    "pipe_contract_alignment",
    "with_features",
]
