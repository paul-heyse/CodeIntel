"""Utilities to reflect serving components into operation contract rows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_type_hints

import pandas as pd

from codeintel.config.datasets.operation_contracts_dataset import (
    OPERATION_CONTRACT_TABLE_KEY,
    TRANSPORT_KINDS,
)
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
from codeintel.serving.operations import iter_operations

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.config.datasets.schema import DatasetSchema


@dataclass(frozen=True)
class ComponentSpec:
    """Describe a component to reflect for contract validation."""

    component: type[object]
    transport: str
    label: str | None = None

    def component_name(self) -> str:
        """Return the display label for the component.

        Returns
        -------
        str
            Explicit label when provided, otherwise the component name.
        """
        if self.label is not None:
            return self.label
        return getattr(self.component, "__name__", repr(self.component))


def _annotation_to_str(annotation: object) -> str:
    """Render a type annotation to a stable string.

    Returns
    -------
    str
        String form of the annotation or ``"Any"`` when missing.
    """
    if annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation)


def _extract_signature_rows(
    component: type[object],
    *,
    transport: str,
    label: str,
    method_names: Sequence[str],
) -> list[dict[str, object]]:
    """Reflect component methods into contract rows.

    Returns
    -------
    list[dict[str, object]]
        Reflected operation records for the provided component.
    """
    rows: list[dict[str, object]] = []
    for method_name in method_names:
        attr = getattr(component, method_name, None)
        if not callable(attr):
            continue
        signature = inspect.signature(attr)
        try:
            hints: Mapping[str, object] = get_type_hints(attr, include_extras=True)
        except Exception:  # noqa: BLE001
            hints = {}

        arg_names: list[str] = []
        arg_types: list[str] = []
        scope_type: str | None = None

        for name, param in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            annotation = hints.get(name, param.annotation)
            annotation_str = _annotation_to_str(annotation)
            arg_names.append(name)
            arg_types.append(annotation_str)
            if name == "scope":
                scope_type = annotation_str

        return_annotation = hints.get("return", signature.return_annotation)

        rows.append(
            {
                "component": label,
                "transport": transport,
                "method": method_name,
                "args": tuple(arg_names),
                "arg_types": tuple(arg_types),
                "return_type": _annotation_to_str(return_annotation),
                "scope_type": scope_type,
            }
        )
    return rows


def build_operation_contract_dataframe(
    components: Iterable[ComponentSpec],
) -> pd.DataFrame:
    """Return a DataFrame of reflected operation contracts.

    Returns
    -------
    pandas.DataFrame
        Reflected operation contract rows for all components.

    Raises
    ------
    ValueError
        If a component uses an unsupported transport kind.
    """
    method_names = sorted({spec.backend_method for spec in iter_operations()})
    records: list[dict[str, object]] = []

    for spec in components:
        if spec.transport not in TRANSPORT_KINDS:
            message = f"Unsupported transport kind: {spec.transport}"
            raise ValueError(message)
        records.extend(
            _extract_signature_rows(
                spec.component,
                transport=spec.transport,
                label=spec.component_name(),
                method_names=method_names,
            )
        )

    return pd.DataFrame.from_records(records)


def get_operation_contract_schema() -> DatasetSchema:
    """Return the registered DatasetSchema for operation contracts.

    Returns
    -------
    DatasetSchema
        Registered schema for operation contract rows.
    """
    return SCHEMA_REGISTRY.require(OPERATION_CONTRACT_TABLE_KEY)


def validate_operation_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a reflected contract DataFrame using the registered schema.

    Parameters
    ----------
    df
        Reflected contract DataFrame to validate.

    Returns
    -------
    pandas.DataFrame
        DataFrame validated against the operation contract schema.
    """
    schema = get_operation_contract_schema()
    return schema.validate(df)


__all__ = [
    "ComponentSpec",
    "build_operation_contract_dataframe",
    "get_operation_contract_schema",
    "validate_operation_contracts",
]
