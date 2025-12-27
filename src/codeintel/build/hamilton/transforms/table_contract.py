"""Table-level decorator composition for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.decorators import pipe_clean_df, with_features


@dataclass(frozen=True, slots=True)
class TableContractSpec:
    """Specification for canonical table policies."""

    table_key: str
    domain: str
    target: str
    ops_module: ModuleType | None
    columns_to_pass: Sequence[str]
    required_cols: Sequence[str] = ("loc", "cyclo")
    clip_column: str | None = "loc"
    input_name: str = "df"


def table_contract(
    spec: TableContractSpec,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator applying canonical table policies.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that applies tag, cleaning, and feature policies.
    """

    def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
        fn = tag_dataset(
            domain=spec.domain,
            target=spec.target,
            table_key=spec.table_key,
        )(fn)
        fn = pipe_clean_df(
            required_cols=spec.required_cols,
            clip_column=spec.clip_column,
            input_name=spec.input_name,
        )(fn)
        if spec.ops_module is not None:
            fn = with_features(
                table_key=spec.table_key,
                columns_to_pass=tuple(spec.columns_to_pass),
                ops_module=spec.ops_module,
            )(fn)
        return fn

    return _decorator


__all__ = ["TableContractSpec", "table_contract"]
