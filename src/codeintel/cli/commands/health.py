"""Health check commands.

Provide commands to verify the CLI environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.health import health_check_handler
from codeintel.cli.rendering.types import OutputFormat

health_app = App(name="health", help="Check CLI environment health")


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

    def __call__(self) -> None:
        """Execute the health check command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        with command_context(
            "health.check",
            runtime_cli,
            output_cli,
            params={},
            require_runtime=False,
        ) as (ctx, renderer):
            result = health_check_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = [
    "health_app",
]
