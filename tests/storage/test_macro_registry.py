"""Validate macro registry entries and hashes match bootstrap expectations."""

from __future__ import annotations

import hashlib
import re

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.metadata_bootstrap import (
    METADATA_SCHEMA_DDL,
    NORMALIZED_MACROS,
    _canonical_type,
)


def _canonicalize_ddl(stmt: str) -> str:
    return " ".join(stmt.split())


def _collect_macro_hashes() -> dict[str, str]:
    macro_hashes: dict[str, str] = {}
    for stmt in METADATA_SCHEMA_DDL:
        match = re.search(r"CREATE\\s+OR\\s+REPLACE\\s+MACRO\\s+([\\w\\.]+)", stmt, re.IGNORECASE)
        if match is None:
            continue
        macro_name = match.group(1)
        normalized = _canonicalize_ddl(stmt)
        macro_hashes[macro_name] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return macro_hashes


def _expected_schema_hash(table_key: str) -> str:
    schema = TABLE_SCHEMAS[table_key]
    parts = []
    for column in schema.columns:
        canonical_type = _canonical_type(column.type)
        parts.append(f"{column.name}:{canonical_type}")
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_macro_registry_hashes(fresh_gateway: StorageGateway) -> None:
    """All macros defined in DDL must be present with matching hashes."""
    con = fresh_gateway.con
    expected_hashes = _collect_macro_hashes()
    macro_to_dataset = {v: k for k, v in NORMALIZED_MACROS.items()}

    rows = con.execute(
        "SELECT macro_name, dataset_table_key, ddl_hash, schema_hash FROM metadata.macro_registry"
    ).fetchall()
    actual = {
        str(name): (
            str(dataset) if dataset is not None else None,
            str(ddl_hash),
            str(schema_hash) if schema_hash is not None else None,
        )
        for name, dataset, ddl_hash, schema_hash in rows
    }

    missing = sorted(set(expected_hashes) - set(actual))
    if missing:
        raise AssertionError(f"Missing macro registry entries: {', '.join(missing)}")

    mismatched: list[str] = []
    dataset_mismatch: list[str] = []
    schema_mismatch: list[str] = []
    for name, expected_hash in expected_hashes.items():
        _, actual_hash, schema_hash = actual[name]
        if actual_hash != expected_hash:
            mismatched.append(name)
        expected_dataset = macro_to_dataset.get(name)
        actual_dataset, _, _ = actual[name]
        if expected_dataset != actual_dataset:
            dataset_mismatch.append(name)
        if expected_dataset is not None:
            expected_schema_hash = _expected_schema_hash(expected_dataset)
            if schema_hash != expected_schema_hash:
                schema_mismatch.append(name)

    if mismatched:
        raise AssertionError(f"Hash drift detected for: {', '.join(sorted(mismatched))}")
    if dataset_mismatch:
        raise AssertionError(f"Dataset mapping drift for: {', '.join(sorted(dataset_mismatch))}")
    if schema_mismatch:
        raise AssertionError(f"Schema hash drift for: {', '.join(sorted(schema_mismatch))}")
