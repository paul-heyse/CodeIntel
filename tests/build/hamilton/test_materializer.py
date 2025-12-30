"""Tests for Hamilton materializer utilities.

These tests validate the Hamilton-native DataSaver implementations used for
DAG-visible I/O, replacing the legacy ``native.materializer`` utilities.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import polars as pl

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.materializers import IcebergDatasetSaver
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.config.view import SettingsView
from codeintel.core.hashing import stable_hash
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.schema import iceberg_field_ids_for_table_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.validation.mode import ContractValidationMode
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.catalog import (
    CatalogBuildOptions,
    build_catalog,
    make_table_output,
    make_target_descriptor,
)
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import Column


def _modules_rows(*, repo: str, commit: str, count: int) -> pl.DataFrame:
    rows = []
    for idx in range(count):
        row = {
            "module": f"m{idx}",
            "path": f"pkg/mod_{idx}.py",
            "repo": repo,
            "commit": commit,
            "language": "python",
            "tags": [],
            "owners": [],
        }
        row["row_hash"] = stable_hash(row)
        rows.append(row)
    return pl.DataFrame(rows)


def _make_graph() -> DagCatalog:
    """Create a minimal catalog that contains a modules target.

    Returns
    -------
    DagCatalog
        Catalog containing only the modules target.
    """
    return build_catalog(
        targets=(
            make_target_descriptor(
                name="modules",
                module="ingestion",
            ),
        ),
        table_keys_by_target={"modules": ("core.modules",)},
    )


def _module_row_for_schema(
    *,
    repo: str,
    commit: str,
    schema_columns: tuple[Column, ...],
) -> tuple[object, ...]:
    """Build a row tuple matching the schema column ordering.

    Returns
    -------
    tuple[object, ...]
        Row tuple matching schema column ordering.
    """
    column_names = tuple(column.name for column in schema_columns)
    values_by_column: dict[str, object] = {}
    for column in schema_columns:
        col_name = column.name
        col_type = column.type
        if col_type == "JSON":
            values_by_column[col_name] = []
        elif col_type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
            values_by_column[col_name] = 1
        elif col_type == "DOUBLE":
            values_by_column[col_name] = 1.0
        elif col_type == "BOOLEAN":
            values_by_column[col_name] = True
        else:
            values_by_column[col_name] = f"value_{col_name}"
    values_by_column["repo"] = repo
    values_by_column["commit"] = commit
    return tuple(values_by_column[name] for name in column_names)


def test_iceberg_saver_writes_table(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should persist rows and expose snapshot metadata."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=snapshot.repo, commit=snapshot.commit, count=2)
    meta = saver.save_data(df.lazy())
    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=2)
    snapshot_id = meta.get("iceberg_snapshot_id")
    expect_true(
        isinstance(snapshot_id, int),
        message="Expected iceberg_snapshot_id to be an integer",
    )

    settings_view = SettingsView.from_build_env(env)
    provider = IcebergCatalogProvider(settings_view.build.iceberg)
    table = provider.load_table("core.modules")
    reader = table.scan().to_arrow_batch_reader()
    written_rows = sum(batch.num_rows for batch in reader)
    expect_equal(written_rows, expected=2)


def test_iceberg_saver_snapshot_properties_persisted(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should persist snapshot properties."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=1)
    meta = saver.save_data(df.lazy())
    expect_equal(meta["status"], expected="succeeded")

    settings_view = SettingsView.from_build_env(env)
    provider = IcebergCatalogProvider(settings_view.build.iceberg)
    table = provider.load_table("core.modules")
    snapshot = table.current_snapshot()
    expect_true(snapshot is not None, message="Expected snapshot after write")
    summary = getattr(snapshot, "summary", None)
    props = summary.additional_properties if summary is not None else {}
    expect_true(isinstance(props, dict), message="Expected snapshot properties mapping")
    expect_equal(props.get("table_key"), expected="core.modules")
    expect_equal(props.get("repo"), expected=env.snapshot.repo)
    expect_equal(props.get("commit"), expected=env.snapshot.commit)
    expected_hash = schema_hash(get_schema_service().require_table_schema("core.modules"))
    expect_equal(props.get("schema_hash"), expected=expected_hash)


def test_iceberg_saver_refreshes_metadata_cache(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should refresh the metadata cache."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=1)
    meta = saver.save_data(df.lazy())
    expect_equal(meta["status"], expected="succeeded")

    tables_ref = meta_table_ref("metadata.iceberg_tables")
    row = env.gateway.con.execute(
        f"SELECT current_snapshot_id, current_schema_id FROM {tables_ref} WHERE table_key = ?",
        ["core.modules"],
    ).fetchone()
    row = expect_is_not_none(row, message="Expected iceberg_tables row")
    expect_true(isinstance(row[0], int), message="Expected current_snapshot_id")
    expect_true(isinstance(row[1], int), message="Expected current_schema_id")


def test_iceberg_saver_schema_evolution_preserves_field_ids(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should preserve field IDs across schema evolution."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    frame = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=1)
    meta = saver.save_data(frame.lazy())
    expect_equal(meta["status"], expected="succeeded")

    expected_schema = get_schema_service().require_table_schema("core.modules")
    expected_ids = iceberg_field_ids_for_table_schema(expected_schema)

    frame = frame.with_columns(pl.lit("extra").alias("extra_flag"))
    meta = saver.save_data(frame.lazy())
    expect_equal(meta["status"], expected="succeeded")

    provider = IcebergCatalogProvider(SettingsView.from_build_env(env).build.iceberg)
    table = provider.load_table("core.modules")
    table.refresh()
    iceberg_schema = table.schema()
    extra_field = iceberg_schema.find_field("extra_flag")
    expect_true(extra_field is not None, message="Expected evolved column in Iceberg schema")
    for name, expected_id in expected_ids.items():
        field = iceberg_schema.find_field(name)
        expect_equal(field.field_id, expected=expected_id)


def test_iceberg_saver_appends_tombstones(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should emit tombstones for deleted rows."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    settings = replace(
        env.settings,
        iceberg=replace(env.settings.iceberg, tombstones_enabled=True),
    )
    env = replace(env, settings=settings)
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df_first = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=2)
    first_meta = saver.save_data(df_first.lazy())
    expect_equal(first_meta["status"], expected="succeeded")

    df_second = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=1)
    second_meta = saver.save_data(df_second.lazy())
    expect_equal(second_meta["status"], expected="succeeded")

    settings_view = SettingsView.from_build_env(env)
    provider = IcebergCatalogProvider(settings_view.build.iceberg)
    tombstone_table = provider.load_table("core.modules__tombstones")
    reader = tombstone_table.scan().to_arrow_batch_reader()
    tombstone_rows = sum(batch.num_rows for batch in reader)
    expect_true(tombstone_rows > 0, message="Expected tombstone rows")


def test_materialize_table_validates_when_schema_available(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should succeed when schema validation is enabled."""
    harness = build_harness.with_force_targets("modules")
    env = replace(harness.build_env(), validate_outputs=True)
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=repo, commit=commit, count=2)
    meta = saver.save_data(df.lazy())
    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=df.height)
    expect_equal(meta["validation_status"], expected="passed")


def test_iceberg_saver_accepts_duckdb_relation(
    build_harness: HamiltonBuildHarness,
) -> None:
    """IcebergDatasetSaver should persist DuckDBRelation inputs."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    graph = _make_graph()
    table_key = "core.modules"
    schema = get_schema_service().require_table_schema(table_key)
    row = _module_row_for_schema(
        repo=repo,
        commit=commit,
        schema_columns=tuple(schema.columns),
    )
    column_names = tuple(column.name for column in schema.columns)
    frame = pl.DataFrame([row], schema=list(column_names))
    env.gateway.con.register("tmp_modules_relation", frame)
    relation = env.gateway.con.table("tmp_modules_relation")

    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key=table_key,
    )

    meta = saver.save_data(relation)

    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=1)


def test_materialize_table_persists_validation_record(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Materializers should persist validation results when enabled."""
    harness = build_harness.with_force_targets("modules")
    env = replace(harness.build_env(), validate_outputs=True)
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=env.snapshot.repo, commit=env.snapshot.commit, count=1)
    meta = saver.save_data(df.lazy())

    expect_equal(meta["status"], expected="succeeded")
    validations_ref = meta_table_ref("metadata.materialization_validations")
    row = env.gateway.con.execute(
        f"SELECT status FROM {validations_ref} WHERE table_key = ? "
        "ORDER BY created_at DESC LIMIT 1",
        ["core.modules"],
    ).fetchone()
    expect_true(row is not None, message="Expected validation record")
    row_tuple = cast("tuple[object, ...]", row)
    expect_equal(row_tuple[0], expected="passed")


def test_strict_validation_fails_on_missing_columns(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Strict validation should fail when required columns are missing."""
    harness = build_harness.with_force_targets("modules")
    env = replace(
        harness.build_env(),
        validate_outputs=True,
        validation_mode=ContractValidationMode.STRICT,
    )
    graph = _make_graph()
    saver = IcebergDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    frame = pl.DataFrame(
        {
            "module": ["m1"],
            "path": ["pkg/mod.py"],
            "repo": [env.snapshot.repo],
            "commit": [env.snapshot.commit],
        }
    ).lazy()

    meta = saver.save_data(frame)

    expect_equal(meta["status"], expected="failed")
    validations_ref = meta_table_ref("metadata.materialization_validations")
    row = env.gateway.con.execute(
        f"SELECT status FROM {validations_ref} WHERE table_key = ? "
        "ORDER BY created_at DESC LIMIT 1",
        ["core.modules"],
    ).fetchone()
    expect_true(row is not None, message="Expected validation record")
    row_tuple = cast("tuple[object, ...]", row)
    expect_equal(row_tuple[0], expected="failed")


def test_internal_outputs_skip_contract_checks(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Internal outputs should skip contract validations even in strict mode."""
    harness = build_harness.with_force_targets("modules")
    env = replace(
        harness.build_env(),
        validate_outputs=True,
        validation_mode=ContractValidationMode.STRICT,
    )
    internal_graph = build_catalog(
        targets=(make_target_descriptor(name="modules", module="ingestion"),),
        options=CatalogBuildOptions(
            table_outputs_by_target={
                "modules": (
                    make_table_output(
                        table_key="core.modules",
                        target="modules",
                        role="internal",
                    ),
                )
            }
        ),
    )
    saver = IcebergDatasetSaver(
        env=env,
        catalog=internal_graph,
        target_name="modules",
        table_key="core.modules",
        output_role="internal",
    )

    frame = pl.DataFrame(
        {
            "module": ["m1"],
            "path": ["pkg/mod.py"],
            "repo": [env.snapshot.repo],
            "commit": [env.snapshot.commit],
        }
    ).lazy()

    meta = saver.save_data(frame)
    expect_equal(meta["status"], expected="succeeded")

    validations_ref = meta_table_ref("metadata.materialization_validations")
    row = env.gateway.con.execute(
        f"SELECT validation_scope, status FROM {validations_ref} WHERE table_key = ? "
        "ORDER BY created_at DESC LIMIT 1",
        ["core.modules"],
    ).fetchone()
    expect_true(row is not None, message="Expected validation record")
    row_tuple = cast("tuple[object, ...]", row)
    expect_equal(row_tuple[0], expected="internal")
    expect_equal(row_tuple[1], expected="passed")
