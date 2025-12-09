"""Health check commands.

Provide commands to verify the CLI environment is properly
configured and all dependencies are available.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.health import CheckStatus, get_health_checker

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
        Literal["text", "json"],
        Parameter(name="--format", help="Output format"),
    ] = "text"

    def __call__(self) -> None:
        """Execute the health check command.

        Raises
        ------
        SystemExit
            If any health check fails.
        """
        checker = get_health_checker()
        report = checker.run_all()

        if self.output_format == "json":
            sys.stdout.write(json.dumps(report.to_dict(), indent=2))
            sys.stdout.write("\n")
            if report.overall_status == CheckStatus.FAIL:
                raise SystemExit(1)
            return

        console = Console()

        # Status symbols
        status_symbols = {
            CheckStatus.OK: "[green]✓[/green]",
            CheckStatus.WARN: "[yellow]![/yellow]",
            CheckStatus.FAIL: "[red]✗[/red]",
            CheckStatus.SKIP: "[dim]-[/dim]",
        }

        table = Table(title="Health Check Results")
        table.add_column("Status", justify="center")
        table.add_column("Check")
        table.add_column("Message")
        table.add_column("Duration", justify="right")

        for check in report.checks:
            symbol = status_symbols.get(check.status, "?")
            table.add_row(
                symbol,
                check.name,
                check.message,
                f"{check.duration_ms:.1f}ms",
            )

        console.print(table)
        console.print()

        overall_symbol = status_symbols.get(report.overall_status, "?")
        console.print(f"Overall: {overall_symbol} {report.overall_status.value.upper()}")
        console.print(f"Total time: {report.total_duration_ms:.1f}ms")

        if report.overall_status == CheckStatus.FAIL:
            raise SystemExit(1)


__all__ = [
    "health_app",
]
