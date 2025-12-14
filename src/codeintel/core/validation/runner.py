"""Unified validation runner for executing validation checks.

This module provides a generic validation runner that can be used by
both graph validation and other validation subsystems.

Classes
-------
ValidationRunner
    Generic runner for executing and aggregating validation checks.
CheckProtocol
    Protocol for validation check implementations.
ValidationReport
    Aggregate report from a validation run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codeintel.core.validation.findings import (
    apply_severity_overrides,
    cap_findings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from codeintel.core.validation.options import BaseValidationOptions, ValidationSeverity

log = logging.getLogger(__name__)


@runtime_checkable
class CheckProtocol[TContext](Protocol):
    """Protocol for validation check implementations.

    Validation checks receive a context object and return a sequence
    of finding dictionaries.

    Type Parameters
    ---------------
    TContext
        The type of context object passed to the check.
    """

    @property
    def name(self) -> str:
        """Return unique check identifier.

        Returns
        -------
        str
            Check name used for filtering and reporting.
        """
        ...

    @property
    def description(self) -> str:
        """Return human-readable description.

        Returns
        -------
        str
            Description of what this check validates.
        """
        ...

    @property
    def severity(self) -> ValidationSeverity:
        """Return default severity level.

        Returns
        -------
        ValidationSeverity
            Default severity for findings from this check.
        """
        ...

    def __call__(self, ctx: TContext) -> Sequence[Mapping[str, object]]:
        """Execute the check and return findings.

        Parameters
        ----------
        ctx
            Context object providing data for validation.

        Returns
        -------
        Sequence[Mapping[str, object]]
            Findings from this check. Each finding should have at
            minimum a 'check_name' and 'severity' field.
        """
        ...


@dataclass
class CheckResult[TFinding: Mapping[str, object]]:
    """Result from executing a single check.

    Attributes
    ----------
    check_name
        Name of the check that was executed.
    findings
        Findings from the check execution.
    duration_s
        Execution duration in seconds.
    error
        Error message if the check failed.
    skipped
        Whether the check was skipped.
    """

    check_name: str
    findings: list[TFinding] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None
    skipped: bool = False


@dataclass
class ValidationReport[TFinding: Mapping[str, object]]:
    """Aggregate report from a validation run.

    Attributes
    ----------
    findings
        All findings from all checks.
    check_results
        Individual results per check.
    total_duration_s
        Total validation duration in seconds.
    error_count
        Number of error-severity findings.
    warning_count
        Number of warning-severity findings.
    info_count
        Number of info-severity findings.
    checks_run
        Number of checks executed.
    checks_skipped
        Number of checks skipped.
    checks_failed
        Number of checks that failed with errors.
    """

    findings: list[TFinding] = field(default_factory=list)
    check_results: list[CheckResult[TFinding]] = field(default_factory=list)
    total_duration_s: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    checks_run: int = 0
    checks_skipped: int = 0
    checks_failed: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if any error-level findings exist.

        Returns
        -------
        bool
            True if error_count > 0.
        """
        return self.error_count > 0

    @property
    def passed(self) -> bool:
        """Check if validation passed (no errors).

        Returns
        -------
        bool
            True if no error-level findings.
        """
        return not self.has_errors


@dataclass
class ValidationRunner[TContext, TFinding: Mapping[str, object]]:
    """Generic runner for executing validation checks.

    This runner provides a unified interface for executing checks,
    applying severity overrides, capping findings, and aggregating
    results into a report.

    Type Parameters
    ---------------
    TContext
        Type of context object passed to checks.
    TFinding
        Type of finding dictionaries returned by checks.

    Attributes
    ----------
    checks
        Registered validation checks.
    options
        Validation options (severity overrides, capping, etc.).

    Examples
    --------
    >>> runner = ValidationRunner[MyContext, dict]()
    >>> runner.register(MyCheck())
    >>> report = runner.run(my_context)
    >>> if report.has_errors:
    ...     print(f"Found {report.error_count} errors")
    """

    checks: list[CheckProtocol[TContext]] = field(default_factory=list)
    options: BaseValidationOptions | None = None

    def register(self, check: CheckProtocol[TContext]) -> None:
        """Register a validation check.

        Parameters
        ----------
        check
            Check to register.
        """
        self.checks.append(check)

    def register_all(self, checks: Sequence[CheckProtocol[TContext]]) -> None:
        """Register multiple validation checks.

        Parameters
        ----------
        checks
            Checks to register.
        """
        self.checks.extend(checks)

    def run(
        self,
        ctx: TContext,
        *,
        check_filter: Callable[[CheckProtocol[TContext]], bool] | None = None,
    ) -> ValidationReport[TFinding]:
        """Execute all registered checks and return report.

        Parameters
        ----------
        ctx
            Context object passed to each check.
        check_filter
            Optional filter function to select which checks to run.

        Returns
        -------
        ValidationReport[TFinding]
            Aggregate validation report.
        """
        start = time.perf_counter()
        report: ValidationReport[TFinding] = ValidationReport()
        all_findings: list[TFinding] = []

        for check in self.checks:
            if check_filter is not None and not check_filter(check):
                report.checks_skipped += 1
                report.check_results.append(
                    CheckResult(check_name=check.name, skipped=True)
                )
                continue

            check_start = time.perf_counter()
            try:
                raw_findings = check(ctx)
                findings = [_with_defaults(f, check) for f in raw_findings]
                check_duration = time.perf_counter() - check_start

                result: CheckResult[TFinding] = CheckResult(
                    check_name=check.name,
                    findings=findings,  # type: ignore[arg-type]
                    duration_s=check_duration,
                )
                report.check_results.append(result)
                all_findings.extend(findings)  # type: ignore[arg-type]
                report.checks_run += 1

            except Exception as e:
                log.exception("Check %s failed with error", check.name)
                check_duration = time.perf_counter() - check_start
                report.check_results.append(
                    CheckResult(
                        check_name=check.name,
                        error=str(e),
                        duration_s=check_duration,
                    )
                )
                report.checks_failed += 1

        if self.options is not None:
            all_findings = apply_severity_overrides(
                all_findings,
                self.options.severity_overrides,
            )  # type: ignore[assignment]
            all_findings = cap_findings(
                all_findings,
                self.options.max_findings_per_rule,
            )  # type: ignore[assignment]

        report.findings = all_findings
        report.total_duration_s = time.perf_counter() - start

        for finding in all_findings:
            sev = str(finding.get("severity", "info"))
            if sev == "error":
                report.error_count += 1
            elif sev == "warning":
                report.warning_count += 1
            else:
                report.info_count += 1

        return report


def _with_defaults(
    finding: Mapping[str, object],
    check: CheckProtocol[Any],
) -> dict[str, object]:
    """Add default fields to a finding if missing.

    Parameters
    ----------
    finding
        Original finding.
    check
        Check that produced the finding.

    Returns
    -------
    dict[str, object]
        Finding with defaults applied.
    """
    result = dict(finding)
    if "check_name" not in result:
        result["check_name"] = check.name
    if "severity" not in result:
        result["severity"] = check.severity
    return result


__all__ = [
    "CheckProtocol",
    "CheckResult",
    "ValidationReport",
    "ValidationRunner",
]
