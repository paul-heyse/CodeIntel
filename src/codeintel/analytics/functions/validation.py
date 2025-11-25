"""Validation helpers for function analytics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from logging import Logger, LoggerAdapter
from typing import Protocol

from codeintel.config.models import FunctionAnalyticsConfig
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import FunctionValidationRow, function_validation_row_to_tuple
from codeintel.storage.gateway import StorageGateway


class FunctionAnalyticsResultProtocol(Protocol):
    """Protocol describing the minimal shape of a function analytics result."""

    @property
    def validation_total(self) -> int: ...

    @property
    def parse_failed_count(self) -> int: ...

    @property
    def span_not_found_count(self) -> int: ...


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    """Validation finding for a GOID that could not be processed."""

    rel_path: str
    qualname: str
    issue: str
    detail: str | None


@dataclass
class ValidationReporter:
    """Emit structured validation counters to logging or a metrics sink."""

    emit_counter: Callable[[str, int], None] | None = None
    logger: Logger | LoggerAdapter = log

    def report(self, result: FunctionAnalyticsResultProtocol, *, scope: str) -> None:
        """
        Emit validation counters for observability.

        Parameters
        ----------
        result:
            Completed analytics result to summarize.
        scope:
            Human-readable scope identifier (e.g., repo@commit).
        """
        counts = {
            "total": result.validation_total,
            "parse_failed": result.parse_failed_count,
            "span_not_found": result.span_not_found_count,
        }
        self.logger.info(
            "METRIC function_validation scope=%s total=%d parse_failed=%d span_not_found=%d",
            scope,
            counts["total"],
            counts["parse_failed"],
            counts["span_not_found"],
            extra={"function_validation": counts},
        )
        if self.emit_counter is None:
            return
        for name, value in counts.items():
            self.emit_counter(f"function_validation.{name}", value)


def persist_validation(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsConfig,
    issues: list[ValidationIssue],
    *,
    created_at: datetime,
) -> None:
    """Persist validation issues to analytics.function_validation."""
    rows = [
        function_validation_row_to_tuple(
            FunctionValidationRow(
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=issue.rel_path,
                qualname=issue.qualname,
                issue=issue.issue,
                detail=issue.detail,
                created_at=created_at,
            )
        )
        for issue in issues
    ]
    run_batch(
        gateway,
        "analytics.function_validation",
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
