"""Health check commands.

Provide commands to verify the CLI environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cyclopts import App

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.health import health_check_handler

health_app = App(name="health", help="Check CLI environment health")

# Config for health commands - no runtime or gateway needed
_HEALTH_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("health.check", handler=health_check_handler, config=_HEALTH_CONFIG)
@health_app.default
@dataclass
class HealthCheckCommand:
    """Run all health checks.

    Verify that the CLI environment is properly configured,
    all dependencies are available, and required services
    are accessible.
    """

    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "health_app",
]
