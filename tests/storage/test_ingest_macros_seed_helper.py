"""Ensure architecture seed helper provisions ingest macros."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.storage.macros import list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway


def test_seed_helper_registers_ingest_macros(tmp_path: Path) -> None:
    """Seeded architecture gateways must expose all ingest macros."""
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "seed.duckdb",
        strict_schema=True,
    )
    macros = list_ingest_macros(gateway.con)
    missing = {m.lower() for m in INGEST_MACROS.values() if m.lower() not in macros}
    gateway.close()
    if missing:
        pytest.fail(f"Missing ingest macros on seeded gateway: {sorted(missing)}")
