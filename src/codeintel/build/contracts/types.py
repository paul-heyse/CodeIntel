"""Contract types for build-time table contract resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Final

from codeintel.core.schemas.arrow_gen import ExtrasPolicy
from codeintel.core.validation.profiles import ValidationProfile


@dataclass(frozen=True, slots=True)
class ContractPolicy:
    """Policy surface for contract alignment and validation behavior."""

    extras_policy: ExtrasPolicy | None = None
    validation_profile: ValidationProfile | None = None
    coerce_types: bool = True
    allow_nulls: bool = True


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
    contract_version: str | None = None
    contract_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ContractDescriptor:
    """Contract identity metadata for table outputs."""

    table_key: str
    contract_version: str
    contract_hash: str


class UnsetType:
    """Sentinel type for optional override values."""

    def __repr__(self) -> str:
        """Return a stable representation for the sentinel.

        Returns
        -------
        str
            String representation of the unset sentinel.
        """
        return "UNSET"


UNSET: Final[UnsetType] = UnsetType()


@dataclass(frozen=True, slots=True)
class ContractOverrides:
    """Override inputs for building a table contract spec."""

    input_name: str | UnsetType = UNSET
    ops_module: ModuleType | UnsetType | None = UNSET
    columns_to_pass: Sequence[str] | UnsetType = UNSET
    required_cols: Sequence[str] | UnsetType = UNSET
    clip_column: str | UnsetType | None = UNSET
    policy: ContractPolicy | UnsetType = UNSET


__all__ = [
    "UNSET",
    "ContractDescriptor",
    "ContractOverrides",
    "ContractPolicy",
    "TableContractSpec",
    "UnsetType",
]
