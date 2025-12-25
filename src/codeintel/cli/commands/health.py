"""Health check commands.

Provide commands to verify the CLI environment is properly
configured and all dependencies are available using the Command[T] pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cyclopts import App

from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.result_types import HealthCheckResult
from codeintel.cli.handlers.health import get_health_checker
from codeintel.cli.options.shared_flags import SharedFlags, shared_flags_field
from codeintel.cli.options.types import CommandPath

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

health_app = App(name="health", help="Check CLI environment health")

HEALTH_CHECK_PATH: CommandPath = ("health",)

_HEALTH_CHECK_FLAGS_FIELD = shared_flags_field(HEALTH_CHECK_PATH)


@cli_command("health.check", require_storage=False)
@health_app.default
@dataclass(frozen=True)
class HealthCheck(Command[HealthCheckResult]):
    """Run all health checks.

    Verify that the CLI environment is properly configured,
    all dependencies are available, and required services
    are accessible.
    """

    __operation_id__ = "health.check"

    flags: SharedFlags = _HEALTH_CHECK_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[HealthCheckResult]:
        """Execute health checks.

        Parameters
        ----------
        ctx
            Command context (unused - health checks are self-contained).

        Returns
        -------
        CliResult[HealthCheckResult]
            Health check results.
        """
        _ = self.flags
        _ = ctx
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


__all__ = [
    "HealthCheck",
    "health_app",
]
