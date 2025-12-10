"""End-to-end macro availability checks for seeded/provisioned gateways."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._helpers import provision_gateway_with_repo
from tests._helpers.macros import assert_all_ingest_macros
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway


@pytest.mark.parametrize("strict_schema", [True, False])
def test_architecture_seed_helper_has_macros(tmp_path: Path, *, strict_schema: bool) -> None:
    """Architecture seed helper must expose all ingest macros."""
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / f"arch-{strict_schema}.duckdb",
        strict_schema=strict_schema,
    )
    assert_all_ingest_macros(gateway.con)
    gateway.close()


def test_provision_helper_has_macros(tmp_path: Path) -> None:
    """Provisioned gateway helper must expose all ingest macros."""
    with provision_gateway_with_repo(tmp_path) as ctx:
        assert_all_ingest_macros(ctx.gateway.con)
