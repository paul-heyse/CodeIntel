"""Health check system and handlers for CLI environment.

Verify that the CLI environment is properly configured and all
dependencies are available before running operations.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.cli.config import DEFAULT_CONFIG_PATHS
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import HealthCheckResult
from codeintel.cli.introspection import get_registry
from codeintel.cli.observability import TelemetryConfig
from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.storage.gateway import open_memory_gateway

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Health check status."""

    OK = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a health check.

    Parameters
    ----------
    name
        Check name.
    status
        Check status.
    message
        Status message.
    duration_ms
        Check duration in milliseconds.
    details
        Additional details.
    """

    name: str
    status: CheckStatus
    message: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Complete health check report.

    Parameters
    ----------
    checks
        Individual check results.
    overall_status
        Overall health status.
    total_duration_ms
        Total check duration.
    """

    checks: list[CheckResult]
    overall_status: CheckStatus
    total_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "overall_status": self.overall_status.value,
            "total_duration_ms": self.total_duration_ms,
            "checks": [c.to_dict() for c in self.checks],
        }


CheckFunction = Callable[[], CheckResult]


def _check_python_version() -> CheckResult:
    """Check Python version.

    Returns
    -------
    CheckResult
        Check result.
    """
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 13):
        return CheckResult(
            name="python_version",
            status=CheckStatus.OK,
            message=f"Python {version_str}",
            details={"version": version_str},
        )
    if version >= (3, 11):
        return CheckResult(
            name="python_version",
            status=CheckStatus.WARN,
            message=f"Python {version_str} (3.13+ recommended)",
            details={"version": version_str},
        )
    return CheckResult(
        name="python_version",
        status=CheckStatus.FAIL,
        message=f"Python {version_str} (3.11+ required)",
        details={"version": version_str},
    )


def _check_config_file() -> CheckResult:
    """Check configuration file.

    Returns
    -------
    CheckResult
        Check result.
    """
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return CheckResult(
                name="config_file",
                status=CheckStatus.OK,
                message=f"Found config: {path}",
                details={"path": str(path)},
            )

    return CheckResult(
        name="config_file",
        status=CheckStatus.WARN,
        message="No config file found (using defaults)",
        details={"searched": [str(p) for p in DEFAULT_CONFIG_PATHS]},
    )


def _check_storage() -> CheckResult:
    """Check storage connectivity.

    Returns
    -------
    CheckResult
        Check result.
    """
    try:
        gateway = open_memory_gateway(
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        gateway.execute("SELECT 1").fetchone()
        gateway.close()
    except StorageConnectionError as exc:
        return CheckResult(
            name="storage_connection",
            status=CheckStatus.FAIL,
            message=f"Storage error: {exc}",
        )

    return CheckResult(
        name="storage_connection",
        status=CheckStatus.OK,
        message="Storage available",
        details={"engine": "duckdb", "api": "storage.gateway"},
    )


def _check_project() -> CheckResult:
    """Check project discovery.

    Returns
    -------
    CheckResult
        Check result.
    """
    cwd = Path.cwd()
    search_paths = [cwd, *cwd.parents]

    for path in search_paths:
        config_path = path / "codeintel.yaml"
        if config_path.exists():
            return CheckResult(
                name="project_discovery",
                status=CheckStatus.OK,
                message=f"Project found: {path}",
                details={"project_root": str(path)},
            )

        toml_path = path / "codeintel.toml"
        if toml_path.exists():
            return CheckResult(
                name="project_discovery",
                status=CheckStatus.OK,
                message=f"Project found: {path}",
                details={"project_root": str(path)},
            )

    return CheckResult(
        name="project_discovery",
        status=CheckStatus.WARN,
        message="No project found in current directory",
    )


def _check_registry() -> CheckResult:
    """Check operation registry.

    Returns
    -------
    CheckResult
        Check result.
    """
    registry = get_registry()
    count = len(registry)

    if count > 0:
        return CheckResult(
            name="operation_registry",
            status=CheckStatus.OK,
            message=f"{count} operations registered",
            details={"operation_count": count},
        )
    return CheckResult(
        name="operation_registry",
        status=CheckStatus.WARN,
        message="No operations registered",
    )


def _check_telemetry() -> CheckResult:
    """Check telemetry configuration.

    Returns
    -------
    CheckResult
        Check result.
    """
    settings = load_runtime_settings().observability
    config = TelemetryConfig.from_settings(settings, default_service_name="codeintel-cli")

    if config.enabled:
        return CheckResult(
            name="telemetry",
            status=CheckStatus.OK,
            message="Telemetry enabled",
            details={"service_name": config.service_name},
        )
    return CheckResult(
        name="telemetry",
        status=CheckStatus.WARN,
        message="Telemetry disabled",
    )


_HEALTH_CHECKS: list[tuple[str, CheckFunction]] = [
    ("python_version", _check_python_version),
    ("config_file", _check_config_file),
    ("storage_connection", _check_storage),
    ("project_discovery", _check_project),
    ("operation_registry", _check_registry),
    ("telemetry", _check_telemetry),
]


class HealthChecker:
    """Run health checks on CLI environment."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self._checks = _HEALTH_CHECKS

    def run_all(self) -> HealthReport:
        """Run all health checks.

        Returns
        -------
        HealthReport
            Complete health report.
        """
        start = time.monotonic()
        results: list[CheckResult] = []

        for name, check_fn in self._checks:
            check_start = time.monotonic()
            try:
                result = check_fn()
                result.duration_ms = (time.monotonic() - check_start) * 1000
            except (OSError, ValueError, RuntimeError) as e:
                result = CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    message=str(e),
                    duration_ms=(time.monotonic() - check_start) * 1000,
                )
            results.append(result)

        total_duration = (time.monotonic() - start) * 1000

        if any(r.status == CheckStatus.FAIL for r in results):
            overall = CheckStatus.FAIL
        elif any(r.status == CheckStatus.WARN for r in results):
            overall = CheckStatus.WARN
        else:
            overall = CheckStatus.OK

        return HealthReport(
            checks=results,
            overall_status=overall,
            total_duration_ms=total_duration,
        )


def get_health_checker() -> HealthChecker:
    """Get health checker instance.

    Returns
    -------
    HealthChecker
        Health checker.
    """
    return HealthChecker()


def health_check_handler(
    ctx: CommandContext,
) -> CliResult[HealthCheckResult]:
    """Run all health checks.

    Parameters
    ----------
    ctx
        Command context (no params required).

    Returns
    -------
    CliResult[HealthCheckResult]
        Health check results.
    """
    _ = ctx.params.raw
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
    "CheckResult",
    "CheckStatus",
    "HealthCheckResult",
    "HealthChecker",
    "HealthReport",
    "get_health_checker",
    "health_check_handler",
    "is_health_check_passing",
]
