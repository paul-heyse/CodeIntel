"""Lenient contract validation tests."""

from __future__ import annotations

from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.gateway.factory import MemoryGatewayOptions
from codeintel.storage.validation import collect_contract_issues_lenient
from tests._helpers.gateway import seed_contract_catalog


def test_lenient_validation_ignores_missing_tables() -> None:
    """Missing tables should not raise in lenient validation."""
    gateway = open_memory_gateway(
        options=MemoryGatewayOptions(
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        ),
        seed_contract_catalog=seed_contract_catalog,
    )
    try:
        issues = collect_contract_issues_lenient(gateway.con, include_views=False)
    finally:
        gateway.close()

    assert isinstance(issues, list)
