"""Tests for dataset tuning metadata in serving manifests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import ArrowDatasetSaver
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.manifests import ServingSnapshotManifest, SnapshotDatasetEntry
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.time import utc_now
from codeintel.storage.datasets.manifest_index import load_dataset_manifests
from codeintel.storage.datasets.manifests import read_dataset_manifest
from codeintel.storage.tracking.schema_catalog_models import (
    DerivedSettingsPayload,
    SchemaObservationRecord,
)
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.catalog import build_catalog, make_target_descriptor
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def _make_graph() -> DagCatalog:
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
) -> None:
    schema = get_schema_service().require_table_schema(table_key)
    schema_digest = fingerprint(schema.to_json_obj())
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
        observed_at=utc_now(),
    )
    gateway = env.gateway
    if gateway is None:
        msg = "Schema observation recording requires a build gateway."
        raise RuntimeError(msg)
    gateway.schemas.record_schema_observations_batch([observation])


def test_manifest_tuning_metadata_round_trips_through_serving(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Serving manifest loader should expose inferred/write settings."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
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
        catalog=_make_graph(),
        target_name="modules",
        table_key=table_key,
    )
    frame = pl.DataFrame(
        {
            "module": ["m1"],
            "path": ["pkg/mod.py"],
            "repo": [env.snapshot.repo],
            "commit": [env.snapshot.commit],
        }
    )
    meta = saver.save_data(frame)
    manifest_path = meta.get("dataset_manifest_path")
    expect_true(isinstance(manifest_path, str), message="Expected dataset manifest path")
    manifest = read_dataset_manifest(Path(cast("str", manifest_path)))
    expect_true(manifest.schema_hash is not None, message="Expected schema_hash in manifest")
    snapshot_manifest = ServingSnapshotManifest(
        run_id="run-test",
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        published_at=utc_now().isoformat(),
        db_path="memory",
        semantic_registry_path="semantic_registry.json",
        schema_manifest_path="schema_manifest.json",
        buildspec_path="buildspec.json",
        semantic_layer_version="v1",
        datasets={
            table_key: SnapshotDatasetEntry(
                manifest_path=str(manifest_path),
                schema_hash=manifest.schema_hash,
                partition_columns=manifest.partition_columns,
                row_count=manifest.row_count,
                stats=manifest.stats,
            )
        },
    )
    index = load_dataset_manifests(snapshot_manifest)
    entry = index.get(table_key)
    if entry is None:
        pytest.fail("Expected dataset manifest entry")
    manifest_entry = entry
    inferred_settings = manifest_entry.inferred_settings
    write_settings = manifest_entry.write_settings
    expect_equal(inferred_settings, expected=derived_settings, label="inferred_settings")
    if not isinstance(write_settings, dict):
        pytest.fail("Expected write_settings payload")
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
