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
    ops_module: ModuleType,
    columns_to_pass: Sequence[str],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator applying canonical table policies.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that tags the node and applies standardized transforms.
    """

    def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
        fn = tag_dataset(domain=domain, target=target, table_key=table_key)(fn)
        fn = pipe_clean_df()(fn)
        fn = with_features(
            table_key=table_key,
            columns_to_pass=tuple(columns_to_pass),
            ops_module=ops_module,
        )(fn)
        return fn

    return _decorator


__all__ = ["table_contract"]
