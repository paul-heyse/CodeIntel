"""Shared guardrails for profile writers.

This module provides utilities for writing profile data with schema validation
and bulk insertion support.

For new code, prefer ``write_rows_via_policy_backend`` via ``PolicyWriterConfig``,
or use Hamilton materializers (DataSavers) from ``codeintel.build.hamilton.materializers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.utilities.datasets import write_analytics_rows
from codeintel.build.analytics.utilities.persistence import DeleteScope

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
    """Write rows via the canonical analytics contract writer.

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

    delete_scope = DeleteScope(repo=config.repo, commit=config.commit)
    return write_analytics_rows(
        gateway,
        config.table_key,
        rows_list,
        delete_scope=delete_scope,
    )
