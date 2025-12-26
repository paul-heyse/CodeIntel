"""Lenient contract validation tests."""

from __future__ import annotations

from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.validation import collect_contract_issues_lenient


def test_lenient_validation_ignores_missing_tables() -> None:
    """Missing tables should not raise in lenient validation."""
    gateway = open_memory_gateway(
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    try:
        issues = collect_contract_issues_lenient(gateway.con, include_views=False)
    finally:
        gateway.close()

    assert isinstance(issues, list)
