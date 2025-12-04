"""End-to-end macro availability checks for seeded/provisioned gateways."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.storage.macros import list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS
from tests._helpers import provision_gateway_with_repo
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway


@pytest.mark.parametrize("strict_schema", [True, False])
def test_architecture_seed_helper_has_macros(tmp_path: Path, *, strict_schema: bool) -> None:
    """
    Architecture seed helper must expose all ingest macros.

    Raises
    ------
    AssertionError
        If ingest macros are missing on the architecture gateway.
    """
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / f"arch-{strict_schema}.duckdb",
        strict_schema=strict_schema,
    )
    macros = list_ingest_macros(gateway.con)
    missing = {m.lower() for m in INGEST_MACROS.values() if m.lower() not in macros}
    gateway.close()
    if missing:
        message = f"Missing ingest macros on architecture gateway: {sorted(missing)}"
        raise AssertionError(message)


def test_provision_helper_has_macros(tmp_path: Path) -> None:
    """
    Provisioned gateway helper must expose all ingest macros.

    Raises
    ------
    AssertionError
        If macros are missing on the provisioned gateway.
    """
    with provision_gateway_with_repo(tmp_path) as ctx:
        macros = list_ingest_macros(ctx.gateway.con)
    missing = {m.lower() for m in INGEST_MACROS.values() if m.lower() not in macros}
    if missing:
        message = f"Missing ingest macros on provisioned gateway: {sorted(missing)}"
        raise AssertionError(message)
