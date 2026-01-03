"""Health check commands."""

from __future__ import annotations

from dataclasses import dataclass

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.health import health_check_handler
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath

health_app = App(name="health", help="Check CLI environment health")

HEALTH_CHECK_PATH: CommandPath = ("health",)

_HEALTH_CHECK_FLAGS_FIELD = shared_flags_field(HEALTH_CHECK_PATH)


_HEALTH_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("health.check", handler=health_check_handler, config=_HEALTH_CONFIG)
@health_app.default
@dataclass(frozen=True)
class HealthCheckCommand:
    """Run all health checks."""

    flags: SharedFlagsProtocol = _HEALTH_CHECK_FLAGS_FIELD


__all__ = [
    "HealthCheckCommand",
    "health_app",
]
