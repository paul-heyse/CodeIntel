"""Table-level decorator composition for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType

from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.decorators import pipe_clean_df, with_features


def table_contract(
    *,
    table_key: str,
    domain: str,
    target: str,
    ops_module: ModuleType | None,
    columns_to_pass: Sequence[str],
    required_cols: Sequence[str] = ("loc", "cyclo"),
    clip_column: str | None = "loc",
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator applying canonical table policies.

    Parameters
    ----------
    table_key
        Fully qualified table key for tagging.
    domain
        Target domain name for tagging.
    target
        Target name for tagging.
    ops_module
        Module containing column-op functions for ``with_columns``. When None,
        feature subDAGs are skipped for this table.
    columns_to_pass
        Column names passed through to feature ops.
    required_cols
        Columns required for strict-mode row filtering.
    clip_column
        Optional column name to apply numeric clipping.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that tags the node and applies standardized transforms.
    """

    def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
        fn = tag_dataset(domain=domain, target=target, table_key=table_key)(fn)
        fn = pipe_clean_df(required_cols=required_cols, clip_column=clip_column)(fn)
        if ops_module is not None:
            fn = with_features(
                table_key=table_key,
                columns_to_pass=tuple(columns_to_pass),
                ops_module=ops_module,
            )(fn)
        return fn

    return _decorator


__all__ = ["table_contract"]
