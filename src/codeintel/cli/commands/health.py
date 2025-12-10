"""Health check commands.

Provide commands to verify the CLI environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.health import health_check_handler
from codeintel.cli.rendering.types import OutputFormat

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

    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True, help="Verbosity level")] = 0


__all__ = [
    "health_app",
]
