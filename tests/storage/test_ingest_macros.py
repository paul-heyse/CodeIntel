"""Tests for ingest macro registration and registry validation.

This module tests:
- Ingest macro registration (ensure_ingest_macros, list_ingest_macros)
- Cache behavior and recovery
- Macro registry hashes and drift detection

Consolidated from:
- test_ingest_macros.py (original)
- test_macro_registry.py
"""

from __future__ import annotations

import hashlib
import re

import duckdb
import pytest

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.macros import (
    assert_ingest_macros_present,
    clear_macro_cache_for_connection,
    ensure_ingest_macros,
    list_ingest_macros,
)
from codeintel.storage.metadata import INGEST_MACROS, METADATA_SCHEMA_DDL, NORMALIZED_MACROS
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)


def test_ensure_ingest_macros_registers_all_macros() -> None:
    """Verify ensure_ingest_macros registers all required macros."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}

        # All required macros should be registered
        expect_true(macro_set.issubset(macros))
    finally:
        con.close()


def test_ensure_ingest_macros_is_idempotent() -> None:
    """Verify ensure_ingest_macros can be called multiple times safely."""
    con = duckdb.connect(":memory:")
    try:
        # First call
        ensure_ingest_macros(con)
        macros_first = list_ingest_macros(con)

        # Second call (should use cache)
        ensure_ingest_macros(con)
        macros_second = list_ingest_macros(con)

        # Results should be the same
        expect_equal(macros_first, macros_second)
    finally:
        con.close()


def test_list_ingest_macros_returns_set() -> None:
    """Verify list_ingest_macros returns a set of macro names."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)
        macros = list_ingest_macros(con)

        expect_is_instance(macros, set)
        expect_true(len(macros) > 0)
    finally:
        con.close()


def test_clear_macro_cache_for_connection_with_connection() -> None:
    """Verify clear_macro_cache_for_connection clears cache for a connection."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Clear cache
        clear_macro_cache_for_connection(con)

        # Should still work after cache clear
        ensure_ingest_macros(con)
        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
        expect_true(macro_set.issubset(macros))
    finally:
        con.close()


def test_clear_macro_cache_for_connection_with_int_key() -> None:
    """Verify clear_macro_cache_for_connection accepts integer key."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Clear cache using id
        cache_key = id(con)
        clear_macro_cache_for_connection(cache_key)

        # Should still work
        macros = list_ingest_macros(con)
        expect_true(len(macros) > 0)
    finally:
        con.close()


def test_assert_ingest_macros_present_succeeds_when_macros_exist() -> None:
    """Verify assert_ingest_macros_present passes when all macros exist."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Should not raise
        assert_ingest_macros_present(con)
    finally:
        con.close()


def test_assert_ingest_macros_present_registers_missing_macros() -> None:
    """Verify assert_ingest_macros_present registers macros if missing."""
    con = duckdb.connect(":memory:")
    try:
        # Create metadata schema without full macro registration
        con.execute("CREATE SCHEMA IF NOT EXISTS metadata")

        # assert_ingest_macros_present should register the missing macros
        assert_ingest_macros_present(con)

        # Verify macros are now present
        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
        expect_true(macro_set.issubset(macros))
    finally:
        con.close()


def test_registered_macros_handles_prefixed_function_names() -> None:
    """Verify _registered_macros parses function names with catalog/schema prefixes."""
    con = duckdb.connect(":memory:")
    try:
        # Create a macro with a prefixed name
        con.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
        con.execute(
            """
            CREATE OR REPLACE MACRO test_schema.prefixed_macro(x) AS x + 1
            """
        )

        macros = list_ingest_macros(con)

        # Should have both qualified and unqualified versions
        expect_in("test_schema.prefixed_macro", macros)
        expect_in("prefixed_macro", macros)
    finally:
        con.close()


def test_macros_contain_ingest_prefix() -> None:
    """Verify all ingest macros follow the metadata.ingest_ naming convention."""
    for table_key, macro_name in INGEST_MACROS.items():
        expect_true(
            macro_name.startswith("metadata.ingest_"),
            message=f"Macro {macro_name} for {table_key} doesn't follow naming convention",
        )


def test_ingest_macros_registered_on_gateway(macro_gateway: StorageGateway) -> None:
    """All ingest macros should be registered automatically for new gateways."""
    macros = list_ingest_macros(macro_gateway.con)
    missing = {macro.lower() for macro in INGEST_MACROS.values() if macro.lower() not in macros}
    if missing:
        pytest.fail(f"Missing ingest macros: {sorted(missing)}")


# =============================================================================
# Cache Recovery Tests (merged from test_ingest_macros_cache.py)
# =============================================================================


def test_ingest_macros_re_register_on_cache_hit_after_close() -> None:
    """Macros remain available even if a new connection reuses a prior id."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    ensure_ingest_macros(gateway.con)
    clear_macro_cache_for_connection(gateway.con)
    gateway.close()

    second_gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    ensure_ingest_macros(second_gateway.con)
    assert_ingest_macros_present(second_gateway.con)
    second_gateway.close()


def test_ingest_macros_recover_if_missing_while_cached() -> None:
    """If macros are dropped after caching, ensure_ingest_macros recreates them."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    ensure_ingest_macros(gateway.con)

    # Drop a macro to simulate missing registration despite a cached entry.
    macro_to_drop = next(iter(INGEST_MACROS.values()))
    gateway.con.execute(f"DROP MACRO IF EXISTS {macro_to_drop}")

    ensure_ingest_macros(gateway.con)
    assert_ingest_macros_present(gateway.con)
    gateway.close()


# =============================================================================
# Macro Registry Hash Tests (merged from test_macro_registry.py)
# =============================================================================


def _canonicalize_ddl(stmt: str) -> str:
    """
    Canonicalize DDL whitespace for hash comparison.

    Returns
    -------
    str
        Whitespace-normalized DDL string.
    """
    return " ".join(stmt.split())


def _canonical_type_for_registry(type_str: str) -> str:
    """
    Canonicalize a type string for registry comparison.

    Returns
    -------
    str
        Canonical type representation.
    """
    upper = type_str.upper()
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def _collect_macro_hashes() -> dict[str, str]:
    """
    Collect hashes for all macros defined in METADATA_SCHEMA_DDL.

    Returns
    -------
    dict[str, str]
        Mapping of macro name to DDL hash.
    """
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
    """
    Compute expected schema hash for a table key.

    Returns
    -------
    str
        SHA256 hash of the canonical schema representation.

    Raises
    ------
    ValueError
        If no contract schema exists for the table key.
    """
    contract = get_dataset_contracts_by_table_key().get(table_key)
    if contract is None or contract.schema is None:
        message = f"No contract schema for {table_key}"
        raise ValueError(message)
    parts: list[str] = []
    for column in contract.schema.columns:
        canonical_type = _canonical_type_for_registry(column.type)
        parts.append(f"{column.name}:{canonical_type}")
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_macro_registry_hashes(fresh_gateway: StorageGateway) -> None:
    """All macros defined in DDL must be present with matching hashes."""
    con = fresh_gateway.con
    expected_hashes = _collect_macro_hashes()
    macro_to_dataset = {macro: table_key for table_key, macro in NORMALIZED_MACROS.items()}

    actual = {
        str(name): (
            str(dataset) if dataset is not None else None,
            str(ddl_hash),
            str(schema_hash) if schema_hash is not None else None,
        )
        for name, dataset, ddl_hash, schema_hash in con.execute(
            "SELECT macro_name, dataset_table_key, ddl_hash, schema_hash FROM metadata.macro_registry"
        ).fetchall()
    }

    missing = sorted(set(expected_hashes) - set(actual))
    if missing:
        pytest.fail(f"Missing macro registry entries: {', '.join(missing)}")

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
            expected_schema_hash_value = _expected_schema_hash(expected_dataset)
            if schema_hash != expected_schema_hash_value:
                schema_mismatch.append(name)

    if mismatched:
        pytest.fail(f"Hash drift detected for: {', '.join(sorted(mismatched))}")
    if dataset_mismatch:
        pytest.fail(f"Dataset mapping drift for: {', '.join(sorted(dataset_mismatch))}")
    if schema_mismatch:
        pytest.fail(f"Schema hash drift for: {', '.join(sorted(schema_mismatch))}")
