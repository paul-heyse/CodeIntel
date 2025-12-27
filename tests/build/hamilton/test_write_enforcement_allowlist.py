"""Tests for write enforcement allowlists."""

from __future__ import annotations

import pytest

from codeintel.build.errors import ContractViolationError
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer


def test_contract_enforcer_allows_declared_writes() -> None:
    """Allowed table/artifact writes should pass in strict mode."""
    allowed_tables = frozenset({"core.alpha"})
    allowed_artifacts = frozenset({"alpha_meta"})

    with ContractEnforcer.for_target(
        "alpha",
        strict=True,
        allowed_tables=allowed_tables,
        allowed_artifacts=allowed_artifacts,
    ):
        ContractEnforcer.validate_table_write("core.alpha")
        ContractEnforcer.validate_artifact_write("alpha_meta")


def test_contract_enforcer_blocks_undeclared_writes() -> None:
    """Undeclared table/artifact writes should raise in strict mode."""
    allowed_tables = frozenset({"core.alpha"})
    allowed_artifacts = frozenset({"alpha_meta"})

    with ContractEnforcer.for_target(
        "alpha",
        strict=True,
        allowed_tables=allowed_tables,
        allowed_artifacts=allowed_artifacts,
    ):
        with pytest.raises(ContractViolationError):
            ContractEnforcer.validate_table_write("core.beta")
        with pytest.raises(ContractViolationError):
            ContractEnforcer.validate_artifact_write("beta_meta")
