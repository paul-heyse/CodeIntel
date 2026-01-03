"""Msgspec boundary decode round-trip tests."""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from codeintel.core.manifests import ArrowDatasetManifest, read_manifest_json, write_manifest_json
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.contract_serde import (
    contract_from_payload,
    contract_payload_from_contract,
    contract_payload_to_json_obj,
)
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.serialization.msgspec_json import encode_json_bytes


def _sample_table_schema() -> TableSchema:
    return TableSchema(
        schema="core",
        name="sample_contracts",
        columns=(
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("example_count", "INTEGER"),
        ),
        primary_key=("repo", "commit"),
    )


def _sample_contract() -> DatasetContract:
    return DatasetContract(
        table_key="core.sample_contracts",
        name="sample_contracts",
        schema=_sample_table_schema(),
        row_binding=None,
        json_schema_id="sample_contracts",
        jsonl_filename="sample_contracts.jsonl",
        parquet_filename="sample_contracts.parquet",
        is_view=False,
        owner_package="core",
        tags=frozenset({"core", "contract"}),
        description="Sample contract payload",
        family="core",
        owner="codeintel",
        freshness_sla="daily",
        retention_policy="30d",
        stable_id="sample-contracts-v1",
        schema_version="v1",
        upstream_dependencies=("core.modules",),
        validation_profile="strict",
        composition=None,
    )


def test_contract_payload_roundtrip_msgpack() -> None:
    """Contract payloads should decode from msgpack with strict typing."""
    contract = _sample_contract()
    payload = contract_payload_from_contract(contract)
    encoded = msgspec.msgpack.encode(payload)
    decoded = contract_from_payload(encoded)
    if decoded.table_key != contract.table_key:
        pytest.fail("Decoded contract table_key mismatch")
    if decoded.schema != contract.schema:
        pytest.fail("Decoded contract schema mismatch")
    if decoded.tags != contract.tags:
        pytest.fail("Decoded contract tags mismatch")
    if decoded.validation_profile != contract.validation_profile:
        pytest.fail("Decoded contract validation_profile mismatch")


def test_contract_payload_legacy_json_normalization() -> None:
    """Legacy payloads with extra fields should still decode."""
    contract = _sample_contract()
    payload = contract_payload_to_json_obj(contract_payload_from_contract(contract))
    payload["unexpected"] = "ignored"
    decoded = contract_from_payload(payload)
    if decoded.table_key != contract.table_key:
        pytest.fail("Legacy payload normalization failed to decode contract")


def test_manifest_roundtrip_with_payload_type(tmp_path: Path) -> None:
    """Manifests should round-trip through JSON with typed decoding."""
    manifest = ArrowDatasetManifest(
        dataset_id="dataset-1",
        snapshot_id="snapshot-1",
        table_key="core.modules",
        partition_columns=("repo", "commit"),
        files=("part-000.parquet",),
        schema_hash="schema-hash",
        row_count=123,
        stats={"min_rows": 1},
        created_at="2024-01-01T00:00:00Z",
        extras={"source": "unit-test"},
    )
    path = tmp_path / "arrow_manifest.json"
    write_manifest_json(path, manifest)
    decoded = read_manifest_json(path, payload_type=ArrowDatasetManifest)
    if decoded != manifest:
        pytest.fail("Manifest payload did not round-trip")


def test_manifest_legacy_json_normalization(tmp_path: Path) -> None:
    """Manifest decoding should drop unknown fields from legacy payloads."""
    manifest = ArrowDatasetManifest(
        dataset_id="dataset-2",
        snapshot_id="snapshot-2",
        table_key="core.repo_map",
        partition_columns=(),
        files=("part-001.parquet",),
    )
    payload = msgspec.to_builtins(manifest)
    payload["unexpected"] = "ignored"
    path = tmp_path / "arrow_manifest_legacy.json"
    path.write_bytes(encode_json_bytes(payload, indent=2, newline=True))
    decoded = read_manifest_json(path, payload_type=ArrowDatasetManifest)
    if decoded.dataset_id != manifest.dataset_id:
        pytest.fail("Legacy manifest normalization failed to decode dataset_id")
