"""Guardrails for semantic serving execution paths."""

from __future__ import annotations

import logging
from collections.abc import Mapping

LOG = logging.getLogger("codeintel.serving.guardrails")


def warn_eager_materialization(*, engine: str, context: str) -> None:
    """Log a warning when an eager materialization path is used.

    Parameters
    ----------
    engine
        Engine name responsible for eager materialization.
    context
        Additional context identifier for the eager materialization site.
    """
    LOG.warning(
        "Eager materialization in serving path",
        extra={"engine": engine, "context": context},
    )


def warn_missing_contract_schema(*, table_key: str) -> None:
    """Log a warning when the Arrow contract schema is missing."""
    LOG.warning("Missing Arrow contract schema", extra={"table_key": table_key})


def warn_contract_metadata_missing(*, table_key: str, field: str) -> None:
    """Log a warning when contract metadata is missing expected fields."""
    LOG.warning(
        "Arrow contract metadata missing",
        extra={"table_key": table_key, "field": field},
    )


def warn_contract_metadata_mismatch(
    *,
    table_key: str,
    field: str,
    expected: str,
    actual: str,
) -> None:
    """Log a warning when contract metadata mismatches expected values."""
    LOG.warning(
        "Arrow contract metadata mismatch",
        extra={"table_key": table_key, "field": field, "expected": expected, "actual": actual},
    )


def warn_schema_drift_observed(*, table_key: str, drift_summary: Mapping[str, object]) -> None:
    """Log a warning when schema drift is observed for a table key."""
    LOG.warning(
        "Schema drift observed",
        extra={"table_key": table_key, "drift_summary": drift_summary},
    )


__all__ = [
    "warn_contract_metadata_mismatch",
    "warn_contract_metadata_missing",
    "warn_eager_materialization",
    "warn_missing_contract_schema",
    "warn_schema_drift_observed",
]
