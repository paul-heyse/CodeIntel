"""Contract types for build-time table contract resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


class UnsetType:
    """Sentinel type for optional override values."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[UnsetType] = UnsetType()


@dataclass(frozen=True, slots=True)
class ContractOverrides:
    """Override inputs for building a table contract spec."""

    input_name: str | UnsetType = UNSET
    ops_module: ModuleType | None | UnsetType = UNSET
    columns_to_pass: Sequence[str] | UnsetType = UNSET
    required_cols: Sequence[str] | UnsetType = UNSET
    clip_column: str | None | UnsetType = UNSET
    policy: ContractPolicy | UnsetType = UNSET


__all__ = [
    "ContractOverrides",
    "ContractPolicy",
    "UNSET",
    "UnsetType",
]
