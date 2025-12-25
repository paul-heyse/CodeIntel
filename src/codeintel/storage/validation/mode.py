"""Validation mode definitions for storage contract checks."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ContractValidationMode"]


class ContractValidationMode(StrEnum):
    """Controls dataset contract validation behavior."""

    OFF = "off"
    LENIENT = "lenient"
    STRICT = "strict"
