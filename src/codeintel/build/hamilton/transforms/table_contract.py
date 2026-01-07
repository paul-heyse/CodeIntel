"""Table-level decorator composition for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable

from codeintel.build.contracts.registry import ContractedTableContext
from codeintel.build.contracts.types import TableContractSpec
from codeintel.build.hamilton.naming import sanitize_pipeline_component
from codeintel.build.hamilton.transforms.decorators import (
    ContractOutputNamespaces,
    pipe_clean_df,
    pipe_contract_output,
    with_features,
)


def table_contract(
    spec: TableContractSpec,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator applying canonical table policies.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that applies cleaning and feature policies.
    """
    return contract_pipeline(spec=spec)


def contract_pipeline(
    *,
    spec: TableContractSpec,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Return a decorator that applies the canonical contract pipeline.

    Returns
    -------
    Callable[[Callable[..., object]], Callable[..., object]]
        Decorator that applies cleaning, feature, alignment, and canonicalization steps.
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
        align_namespace = f"align__{sanitize_pipeline_component(spec.table_key)}"
        alignment_context = ContractedTableContext(contract=spec, policy=spec.policy)
        output_namespace = f"post__{sanitize_pipeline_component(spec.table_key)}"
        namespaces = ContractOutputNamespaces(
            align=align_namespace,
            canonical=output_namespace,
        )
        return pipe_contract_output(
            table_key=spec.table_key,
            target_name=spec.target,
            policy=spec.policy,
            context=alignment_context,
            namespaces=namespaces,
        )(fn)

    return _decorator


__all__ = ["TableContractSpec", "contract_pipeline", "table_contract"]
