"""CLI tests for Iceberg commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from codeintel.core.config.settings import IcebergSettings
from codeintel.storage.iceberg.migration import IcebergAddFilesRequest, add_files_to_iceberg
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import assert_success

if TYPE_CHECKING:
    from tests._helpers.cli_project import CLIProjectHarness

pytestmark = pytest.mark.xdist_group("cli_shared_flags")

TABLE_KEY = "core.modules"
REF_NAME = "commit/test"


def _iceberg_settings(tmp_path: Path) -> IcebergSettings:
    catalog_dir = tmp_path / "iceberg"
    catalog_path = catalog_dir / "catalog.duckdb"
    warehouse_path = catalog_dir / "warehouse"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    return IcebergSettings(
        read_enabled=True,
        write_enabled=True,
        catalog_type="sql",
        catalog_uri=f"duckdb:///{catalog_path}",
        catalog_warehouse=str(warehouse_path),
    )


def _seed_iceberg_table(
    tmp_path: Path,
    *,
    table_key: str,
) -> tuple[IcebergSettings, int]:
    settings = _iceberg_settings(tmp_path)
    data_path = tmp_path / "data" / "rows.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    pq.write_table(table, data_path)
    result = add_files_to_iceberg(
        IcebergAddFilesRequest(table_key=table_key, file_paths=(data_path,)),
        settings=settings,
    )
    snapshot_id = result.snapshot_id
    if snapshot_id is None:
        msg = "Expected snapshot_id after add_files"
        raise AssertionError(msg)
    return settings, snapshot_id


def _configure_iceberg_env(
    harness: CLIProjectHarness,
    *,
    settings: IcebergSettings,
) -> None:
    if settings.catalog_uri is None or settings.catalog_warehouse is None:
        msg = "Iceberg settings missing catalog paths"
        raise AssertionError(msg)
    env = {
        "CODEINTEL_ICEBERG_READ_ENABLED": "1",
        "CODEINTEL_ICEBERG_WRITE_ENABLED": "1",
        "CODEINTEL_ICEBERG_CATALOG_TYPE": settings.catalog_type or "sql",
        "CODEINTEL_ICEBERG_CATALOG_URI": settings.catalog_uri,
        "CODEINTEL_ICEBERG_CATALOG_WAREHOUSE": settings.catalog_warehouse,
    }
    harness.harness = harness.harness.with_env(**env)


def _invoke_json(harness: CLIProjectHarness, args: list[str]) -> dict[str, object]:
    result = harness.invoke([*args, "--output-format", "json"])
    assert_success(result)
    payload = json.loads(result.stdout)
    expect_is_instance(payload, dict)
    return payload


def test_iceberg_inspect_and_refs(
    tmp_path: Path,
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Inspect and refs should return Iceberg metadata payloads."""
    settings, _snapshot_id = _seed_iceberg_table(tmp_path, table_key=TABLE_KEY)
    _configure_iceberg_env(cli_project_harness, settings=settings)

    payload = _invoke_json(
        cli_project_harness,
        ["iceberg", "inspect", "--table", TABLE_KEY, "--snapshots"],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    expect_equal(data.get("table_key"), TABLE_KEY)
    snapshots = data.get("snapshots")
    expect_is_instance(snapshots, list)
    snapshots = cast("list[object]", snapshots)
    expect_true(len(snapshots) > 0, message="Expected snapshot rows")

    payload = _invoke_json(
        cli_project_harness,
        ["iceberg", "refs", "--table", TABLE_KEY],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    refs = data.get("refs")
    expect_is_instance(refs, list)
    refs = cast("list[object]", refs)
    expect_true(len(refs) > 0, message="Expected ref rows")


def test_iceberg_manage_snapshots_create_remove(
    tmp_path: Path,
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Manage snapshots should create and remove refs."""
    settings, snapshot_id = _seed_iceberg_table(tmp_path, table_key=TABLE_KEY)
    _configure_iceberg_env(cli_project_harness, settings=settings)

    payload = _invoke_json(
        cli_project_harness,
        [
            "iceberg",
            "manage-snapshots",
            "--table",
            TABLE_KEY,
            "--snapshot-id",
            str(snapshot_id),
            "--ref-name",
            REF_NAME,
            "--ref-type",
            "tag",
        ],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    expect_equal(data.get("action"), "created")
    expect_equal(data.get("ref_name"), REF_NAME)

    payload = _invoke_json(
        cli_project_harness,
        [
            "iceberg",
            "manage-snapshots",
            "--table",
            TABLE_KEY,
            "--ref-name",
            REF_NAME,
            "--ref-type",
            "tag",
            "--remove",
            "--confirm",
        ],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    expect_equal(data.get("action"), "removed")


def test_iceberg_expire_snapshots_dry_run(
    tmp_path: Path,
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Expire snapshots should support dry-run mode."""
    settings, _snapshot_id = _seed_iceberg_table(tmp_path, table_key=TABLE_KEY)
    _configure_iceberg_env(cli_project_harness, settings=settings)

    payload = _invoke_json(
        cli_project_harness,
        [
            "iceberg",
            "expire-snapshots",
            "--table",
            TABLE_KEY,
            "--retention-days",
            "1",
            "--dry-run",
        ],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    expect_equal(data.get("table_key"), TABLE_KEY)
    expect_equal(data.get("dry_run"), expected=True)
    expired = data.get("expired_snapshot_ids")
    expect_is_instance(expired, list)
    expired = cast("list[object]", expired)


def test_iceberg_refresh_cache(
    tmp_path: Path,
    cli_project_harness: CLIProjectHarness,
) -> None:
    """Refresh cache should return updated counts."""
    settings, _snapshot_id = _seed_iceberg_table(tmp_path, table_key=TABLE_KEY)
    _configure_iceberg_env(cli_project_harness, settings=settings)

    payload = _invoke_json(
        cli_project_harness,
        [
            "iceberg",
            "refresh-cache",
            "--table",
            TABLE_KEY,
        ],
    )
    data = payload.get("data")
    expect_is_instance(data, dict)
    data = cast("dict[str, object]", data)
    expect_equal(data.get("refreshed"), 1)
