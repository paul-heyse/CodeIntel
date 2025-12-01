"""Validate provision helper ensures ingest macros are registered."""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.ingest_macros import list_ingest_macros
from codeintel.storage.metadata_bootstrap import INGEST_MACROS
from tests._helpers.fixtures import provision_gateway_with_repo


def test_provision_helper_registers_macros(tmp_path: Path) -> None:
    """
    Provisioned gateways must expose ingest macros for downstream ingestion.

    Raises
    ------
    AssertionError
        If any ingest macros are missing after provisioning.
    """
    with provision_gateway_with_repo(tmp_path) as ctx:
        macros = list_ingest_macros(ctx.gateway.con)
        missing = {m.lower() for m in INGEST_MACROS.values() if m.lower() not in macros}
    if missing:
        message = f"Missing ingest macros on provisioned gateway: {sorted(missing)}"
        raise AssertionError(message)
