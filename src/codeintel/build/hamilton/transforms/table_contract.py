"""Table-level decorator composition for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import ModuleType

from codeintel.build.contracts.types import ContractPolicy
from codeintel.build.hamilton.naming import sanitize_pipeline_component
from codeintel.build.hamilton.transforms.decorators import (
    pipe_canonical_output,
    pipe_clean_df,
    with_features,
)


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
    policy: ContractPolicy = field(default_factory=ContractPolicy)


def table_contract(
    spec: TableContractSpec,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator applying canonical table policies.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that applies cleaning and feature policies.
    """

    def _decorator(fn: Callable[..., object]) -> Callable[..., object]:
        clean_namespace = f"prep__{sanitize_pipeline_component(spec.table_key)}"
        fn = pipe_clean_df(
            required_cols=spec.required_cols,
            clip_column=spec.clip_column,
            input_name=spec.input_name,
            namespace=clean_namespace,
        )(fn)
        if spec.ops_module is not None:
            fn = with_features(
                table_key=spec.table_key,
                columns_to_pass=tuple(spec.columns_to_pass),
                ops_module=spec.ops_module,
            )(fn)
        output_namespace = f"post__{sanitize_pipeline_component(spec.table_key)}"
        return pipe_canonical_output(
            table_key=spec.table_key,
            namespace=output_namespace,
        )(fn)

    return _decorator


__all__ = ["TableContractSpec", "table_contract"]
