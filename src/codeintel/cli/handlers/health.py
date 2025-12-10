"""Health check handlers.

Handlers for CLI environment health checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.health import CheckStatus, get_health_checker
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthCheckResult:
    """Result from health check.

    Parameters
    ----------
    checks
        List of individual check results.
    overall_status
        Overall status (ok, warn, fail, skip).
    total_duration_ms
        Total time for all checks in milliseconds.
    """

    checks: list[dict[str, object]]
    overall_status: str
    total_duration_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "checks": self.checks,
            "overall_status": self.overall_status,
            "total_duration_ms": self.total_duration_ms,
        }


def health_check_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[HealthCheckResult]:
    """Run all health checks.

    Parameters
    ----------
    ctx
        Handler context (no params required).

    Returns
    -------
    CliResult[HealthCheckResult]
        Health check results.
    """
    # Use ctx for logging context
    _ = ctx.params  # Acknowledge params even if empty
    LOG.info("Running health checks")

    checker = get_health_checker()
    report = checker.run_all()

    checks: list[dict[str, object]] = [
        {
            "name": check.name,
            "status": check.status.value,
            "message": check.message,
            "duration_ms": check.duration_ms,
        }
        for check in report.checks
    ]

    return CliResult.ok(
        HealthCheckResult(
            checks=checks,
            overall_status=report.overall_status.value,
            total_duration_ms=report.total_duration_ms,
        )
    )


def is_health_check_passing(result: HealthCheckResult) -> bool:
    """Check if health result indicates all checks passed.

    Parameters
    ----------
    result
        Health check result.

    Returns
    -------
    bool
        True if all checks passed (ok or warn).
    """
    return result.overall_status != CheckStatus.FAIL.value


__all__ = [
    "HealthCheckResult",
    "health_check_handler",
    "is_health_check_passing",
]
