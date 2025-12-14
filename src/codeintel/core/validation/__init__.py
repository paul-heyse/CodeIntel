"""Core validation infrastructure for graphs and ingestion.

This package provides common validation options, helper functions, and
a unified validation runner used by both graph validation and ingestion
validation frameworks.

Key Components
--------------
BaseValidationOptions
    Common options structure for all validation subsystems.
ValidationSeverity
    Type alias for severity levels (info, warning, error).
apply_severity_overrides
    Apply rule-specific severity overrides to findings.
cap_findings
    Limit findings per rule to avoid overwhelming output.
has_error_findings
    Check if any findings have error severity.

Validation Runner
-----------------
ValidationRunner
    Generic runner for executing validation checks.
CheckProtocol
    Protocol for validation check implementations.
CheckResult
    Result from executing a single check.
ValidationReport
    Aggregate report from a validation run.

Example
-------
```python
from codeintel.core.validation import (
    BaseValidationOptions,
    ValidationRunner,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)


options = BaseValidationOptions(
    severity_overrides={"null_check": "error"},
    hard_fail=True,
    max_findings_per_rule=50,
)

runner = ValidationRunner[MyContext, dict](options=options)
runner.register(MyCheck())
report = runner.run(context)

if options.hard_fail and report.has_errors:
    raise RuntimeError("Validation failed with errors")
```
"""

from __future__ import annotations

from codeintel.core.validation.findings import (
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)
from codeintel.core.validation.options import (
    BaseValidationOptions,
    ValidationSeverity,
)
from codeintel.core.validation.runner import (
    CheckProtocol,
    CheckResult,
    ValidationReport,
    ValidationRunner,
)

__all__ = [
    "BaseValidationOptions",
    "CheckProtocol",
    "CheckResult",
    "ValidationReport",
    "ValidationRunner",
    "ValidationSeverity",
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
]
