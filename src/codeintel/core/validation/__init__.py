"""Core validation infrastructure for graphs and ingestion.

This package provides common validation options and helper functions
used by both graph validation and ingestion validation frameworks.

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

Example
-------
```python
from codeintel.core.validation import (
    BaseValidationOptions,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)

# Create options with overrides
options = BaseValidationOptions(
    severity_overrides={"null_check": "error"},
    hard_fail=True,
    max_findings_per_rule=50,
)

# Process findings
findings = run_validation_checks()
findings = apply_severity_overrides(findings, options.severity_overrides)
findings = cap_findings(findings, options.max_findings_per_rule)

if options.hard_fail and has_error_findings(findings):
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

__all__ = [
    "BaseValidationOptions",
    "ValidationSeverity",
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
]
