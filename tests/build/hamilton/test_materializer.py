"""Tests for Hamilton materializer utilities.

These tests validate the Hamilton-native DataSaver implementations used for
DAG-visible I/O, replacing the legacy ``native.materializer`` utilities.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
from codeintel.build.hamilton.materializers import ArrowDatasetSaver, DuckDBRelationSaver
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    RunRecordInputs,
    create_run_record,
)
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.hashing import stable_hash
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.time import utc_now
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.tracking.schema_catalog_models import (
    DerivedSettingsPayload,
    SchemaObservationRecord,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)
from tests._helpers.catalog import build_catalog, make_target_descriptor
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.env import BuildEnv
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


def _record_schema_observation(
    *,
    env: BuildEnv,
    table_key: str,
    derived_settings: DerivedSettingsPayload,
    drift_summary: Mapping[str, object] | None = None,
) -> None:
    schema = get_schema_service().require_table_schema(table_key)
    schema_json = schema.to_json_obj()
    schema_digest = fingerprint(schema_json)
    schema_hash_value = schema_hash(schema)
    arrow_schema = arrow_schema_from_table_schema(table_schema=schema)
    observation = SchemaObservationRecord(
        table_key=table_key,
        schema_digest=schema_digest,
        schema_hash=schema_hash_value,
        arrow_schema_ipc_b64=schema_to_ipc_payload(arrow_schema),
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        target_name="modules",
        derived_settings=derived_settings,
        drift_summary=drift_summary,
        observed_at=utc_now(),
    )
    env.gateway.schemas.record_schema_observations_batch([observation])


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


def test_materialize_table_uses_policy_and_insert_select(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBRelationSaver should replace snapshot rows via Warehouse policy."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    graph = _make_graph()
    saver = DuckDBRelationSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df1 = _modules_rows(repo=repo, commit=commit, count=1)
    env.gateway.con.register("tmp_modules_1", df1)
    rel1 = env.gateway.con.table("tmp_modules_1")
    meta1 = saver.save_data(rel1)
    expect_equal(meta1["status"], expected="succeeded")
    expect_equal(meta1["row_count"], expected=1)

    df2 = _modules_rows(repo=repo, commit=commit, count=2)
    env.gateway.con.register("tmp_modules_2", df2)
    rel2 = env.gateway.con.table("tmp_modules_2")
    meta2 = saver.save_data(rel2)
    expect_equal(meta2["status"], expected="succeeded")
    expect_equal(meta2["row_count"], expected=2)

    row = env.gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row is not None, message="Expected COUNT(*) query to return a row")
    row_tuple = cast("tuple[int, ...]", row)
    expect_equal(row_tuple[0], expected=2)


def test_materialize_table_validates_when_schema_available(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBRelationSaver should succeed when schema validation is enabled."""
    harness = build_harness.with_force_targets("modules")
    env = replace(harness.build_env(), validate_outputs=True)
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    graph = _make_graph()
    saver = DuckDBRelationSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=repo, commit=commit, count=2)
    env.gateway.con.register("tmp_modules_validate", df)
    rel = env.gateway.con.table("tmp_modules_validate")
    meta = saver.save_data(rel)
    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=df.height)


def test_relation_saver_accepts_lazyframe(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBRelationSaver should persist LazyFrame inputs."""
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
    frame = pl.DataFrame([row], schema=list(column_names)).lazy()

    saver = DuckDBRelationSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key=table_key,
    )

    meta = saver.save_data(frame.lazy())

    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=1)

    row_result = env.gateway.con.execute(
        "SELECT * FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row_result is not None, message="Expected row materialization to persist data")
    persisted = cast("tuple[object, ...]", row_result)
    expect_equal(persisted, expected=row)


def test_arrow_dataset_saver_writes_manifest(
    build_harness: HamiltonBuildHarness,
) -> None:
    """ArrowDatasetSaver should emit a dataset manifest for persisted data."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    graph = _make_graph()
    saver = ArrowDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key="core.modules",
    )

    frame = pl.DataFrame(
        {
            "module": ["m1"],
            "path": ["pkg/mod.py"],
            "repo": [snapshot.repo],
            "commit": [snapshot.commit],
        }
    )

    meta = saver.save_data(frame)
    expect_equal(meta["status"], expected="succeeded")
    manifest_path = meta.get("dataset_manifest_path")
    expect_true(
        isinstance(manifest_path, str),
        message="Expected dataset_manifest_path to be a string",
    )
    manifest_path_str = cast("str", manifest_path)
    manifest = read_dataset_manifest(Path(manifest_path_str))
    expect_equal(manifest.table_key, expected="core.modules")
    expect_equal(manifest.snapshot_id, expected=snapshot.commit)
    expect_equal(manifest.row_count, expected=1)


def test_arrow_dataset_saver_emits_inferred_settings(
    build_harness: HamiltonBuildHarness,
) -> None:
    """ArrowDatasetSaver should persist inferred settings in dataset manifests."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    graph = _make_graph()
    table_key = "core.modules"
    derived_settings: DerivedSettingsPayload = {
        "extras_policy": "retain",
        "dictionary_encode_columns": ["module", "path"],
        "dictionary_max_cardinality": 42,
        "unify_dictionaries": True,
        "row_group_size": 12345,
        "data_page_size": 65536,
    }
    _record_schema_observation(
        env=env,
        table_key=table_key,
        derived_settings=derived_settings,
    )
    saver = ArrowDatasetSaver(
        env=env,
        catalog=graph,
        target_name="modules",
        table_key=table_key,
    )
    frame = pl.DataFrame(
        {
            "module": ["m1"],
            "path": ["pkg/mod.py"],
            "repo": [snapshot.repo],
            "commit": [snapshot.commit],
        }
    )

    meta = saver.save_data(frame)

    manifest_path = meta.get("dataset_manifest_path")
    expect_true(
        isinstance(manifest_path, str),
        message="Expected manifest path to be set",
    )
    manifest = read_dataset_manifest(Path(cast("str", manifest_path)))
    extras = manifest.extras
    expect_true(
        isinstance(extras, dict),
        message="Expected manifest extras to be present",
    )
    extras_map = cast("dict[str, object]", extras)
    inferred_settings = cast("dict[str, object]", extras_map.get("inferred_settings"))
    write_settings = cast("dict[str, object]", extras_map.get("write_settings"))
    expect_equal(inferred_settings, expected=derived_settings, label="inferred_settings")
    expect_equal(
        write_settings.get("row_group_size"),
        expected=12345,
        label="row_group_size",
    )
    expect_equal(
        write_settings.get("data_page_size"),
        expected=65536,
        label="data_page_size",
    )
    expect_equal(
        write_settings.get("dictionary_encode_columns"),
        expected=["module", "path"],
        label="dictionary_encode_columns",
    )
    expect_equal(
        write_settings.get("dictionary_max_cardinality"),
        expected=42,
        label="dictionary_max_cardinality",
    )


def test_create_run_record_includes_drift_summaries(
    build_harness: HamiltonBuildHarness,
) -> None:
    """create_run_record should capture drift summaries from observations."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    graph = _make_graph()
    table_key = "core.modules"
    drift_summary: dict[str, object] = {"extra_columns": 1, "missing_columns": ["owners"]}
    minimal_settings: DerivedSettingsPayload = {"extras_policy": "retain"}
    _record_schema_observation(
        env=env,
        table_key=table_key,
        derived_settings=minimal_settings,
        drift_summary=drift_summary,
    )
    target = graph.get_target("modules")
    expect_true(target is not None, message="Expected modules target to exist")
    target_descriptor = cast("TargetDescriptor", target)
    run_info = NativeRunInfo(
        input_hash="input_hash",
        options_hash="options_hash",
        duration_ms=12.0,
        row_counts={table_key: 1},
    )

    record = create_run_record(
        target_descriptor,
        "succeeded",
        "input_hash",
        inputs=RunRecordInputs(env=env, run=run_info, catalog=graph),
    )

    expect_equal(
        record.drift_summaries.get(table_key),
        expected=drift_summary,
        label="drift_summary",
    )
