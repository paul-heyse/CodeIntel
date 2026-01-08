"""Lazy contract references for build-time targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, Unpack

from codeintel.build.contracts.types import UNSET, ContractOverrides

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from codeintel.build.contracts.types import ContractPolicy


class ContractRefOverrides(TypedDict, total=False):
    """Override values accepted by contract_ref_for_table."""

    ops_module: ModuleType | None
    columns_to_pass: Sequence[str]
    required_cols: Sequence[str]
    clip_column: str | None
    policy: ContractPolicy


@dataclass(frozen=True, slots=True)
class ContractRef:
    """Lightweight contract handle resolved at runtime."""

    table_key: str
    target_name: str
    input_name: str
    overrides: ContractOverrides | None = None

    @property
    def domain(self) -> str:
        """Return the domain inferred from the table key."""
        return _domain_from_table_key(self.table_key)


def contract_ref_for_table(
    *,
    table_key: str,
    target_name: str,
    input_name: str,
    **overrides: Unpack[ContractRefOverrides],
) -> ContractRef:
    """Return a contract reference for a target table.

    Returns
    -------
    ContractRef
        Lazy contract reference for the table key.
    """
    resolved_overrides = None
    if overrides:
        resolved_overrides = ContractOverrides(
            input_name=UNSET,
            **overrides,
        )
    return ContractRef(
        table_key=table_key,
        target_name=target_name,
        input_name=input_name,
        overrides=resolved_overrides,
    )


def _domain_from_table_key(table_key: str) -> str:
    if "." not in table_key:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    domain, table_name = table_key.split(".", maxsplit=1)
    if not domain or not table_name:
        msg = f"Invalid table key: {table_key!r}"
        raise ValueError(msg)
    return domain


__all__ = ["ContractRef", "ContractRefOverrides", "contract_ref_for_table"]
