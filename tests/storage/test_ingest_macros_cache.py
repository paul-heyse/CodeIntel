"""Ensure ingest macro registration survives connection reuse and cache hits."""

from __future__ import annotations

from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.macros import (
    assert_ingest_macros_present,
    clear_macro_cache_for_connection,
    ensure_ingest_macros,
)
from codeintel.storage.metadata import INGEST_MACROS


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
