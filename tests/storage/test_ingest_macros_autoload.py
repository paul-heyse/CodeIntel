"""Smoke test to ensure ingest macros are available on fresh gateways."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.macros import list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS


def test_ingest_macros_registered_on_gateway() -> None:
    """All ingest macros should be registered automatically for new gateways."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    con = gateway.con
    macros = list_ingest_macros(con)
    missing = {macro.lower() for macro in INGEST_MACROS.values() if macro.lower() not in macros}
    if missing:
        pytest.fail(f"Missing ingest macros: {sorted(missing)}")
    gateway.close()
