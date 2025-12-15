"""Shared guardrails for profile writers.

This module provides utilities for writing profile data with schema validation
and bulk insertion support.

For new code, use ``materialize_rows`` from
``codeintel.build.hamilton.native.materializer`` with Hamilton materializers,
or ``write_rows_via_policy_backend`` via ``PolicyWriterConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class PolicyWriterConfig:
    """Configuration for policy-backend-based row writing."""

    table_key: str
    repo: str
    commit: str


def write_rows_via_policy_backend(
    gateway: StorageGateway,
    *,
    rows: Iterable[Mapping[str, object]],
    config: PolicyWriterConfig,
) -> int:
    """Write rows using DuckDBPolicyBackend for bulk insert.

    This function provides a cleaner API that uses the centralized policy
    backend for SQL generation, replacing direct executemany calls.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    rows
        Iterable of row dictionaries to insert.
    config
        Writer configuration including table key, columns, and serializer.

    Returns
    -------
    int
        Number of rows inserted.
    """
    rows_list = list(rows)
    if not rows_list:
        return 0

    backend = gateway.policy

    backend.delete_for_snapshot(config.table_key, repo=config.repo, commit=config.commit)

    return backend.bulk_insert_mappings(config.table_key, rows_list)
