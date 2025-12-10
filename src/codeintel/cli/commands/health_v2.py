"""Health check commands (new Command[T] pattern).

Provide commands to verify the CLI environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cyclopts import App

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.result_types import HealthCheckResult
from codeintel.cli.handlers.health import get_health_checker

if TYPE_CHECKING:
    from codeintel.cli.deps import Deps

LOG = logging.getLogger(__name__)

health_v2_app = App(name="health", help="Check CLI environment health (v2)")


@cli_command("health_v2.check", require_storage=False)
@health_v2_app.default
@dataclass(frozen=True)
class HealthCheck(Command[HealthCheckResult]):
    """Run all health checks.

    Verify that the CLI environment is properly configured,
    all dependencies are available, and required services
    are accessible.
    """

    __operation_id__ = "health_v2.check"

    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[HealthCheckResult]:
        """Execute health checks.

        Parameters
        ----------
        deps
            Dependencies container (unused - health checks are self-contained).

        Returns
        -------
        CliResult[HealthCheckResult]
            Health check results.
        """
        # Acknowledge self for method signature compatibility
        _ = self.flags  # Access flags for potential future use
        _ = deps
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
    "health_v2_app",
]
