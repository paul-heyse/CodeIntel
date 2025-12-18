"""Core validation infrastructure for graphs and ingestion.

This package provides common validation options, helper functions, and
a unified validation runner used by both graph validation and ingestion
validation frameworks.

This ``__init__`` intentionally avoids importing heavy dependencies at import
time. Some foundational layers (e.g., `codeintel.core.options`) depend on
`codeintel.core.validation.outcome`, and importing submodules loads this package
first. To prevent circular imports during module initialization, exports are
resolved lazily via ``__getattr__``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

_MODULE_EXPORTS: Final[dict[str, tuple[str, ...]]] = {
    "codeintel.core.validation.findings": (
        "apply_severity_overrides",
        "cap_findings",
        "has_error_findings",
    ),
    "codeintel.core.validation.options": ("BaseValidationOptions", "ValidationSeverity"),
    "codeintel.core.validation.outcome": ("ValidationOutcome",),
    "codeintel.core.validation.reporters": (
        "FUNCTION_VALIDATION_COLS",
        "GRAPH_VALIDATION_COLS",
        "BaseValidationReporter",
        "FunctionValidationReporter",
        "GraphValidationReporter",
        "gateway_timestamp",
    ),
    "codeintel.core.validation.runner": (
        "CheckProtocol",
        "CheckResult",
        "ValidationReport",
        "ValidationRunner",
    ),
}

_EXPORT_TO_MODULE: Final[dict[str, str]] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

__all__: Final[tuple[str, ...]] = tuple(sorted(_EXPORT_TO_MODULE))

if TYPE_CHECKING:
    from codeintel.core.validation.findings import (
        apply_severity_overrides,
        cap_findings,
        has_error_findings,
    )
    from codeintel.core.validation.options import BaseValidationOptions, ValidationSeverity
    from codeintel.core.validation.outcome import ValidationOutcome
    from codeintel.core.validation.reporters import (
        FUNCTION_VALIDATION_COLS,
        GRAPH_VALIDATION_COLS,
        BaseValidationReporter,
        FunctionValidationReporter,
        GraphValidationReporter,
        gateway_timestamp,
    )
    from codeintel.core.validation.runner import (
        CheckProtocol,
        CheckResult,
        ValidationReport,
        ValidationRunner,
    )

    _TYPE_CHECKING_EXPORTS = (
        FUNCTION_VALIDATION_COLS,
        GRAPH_VALIDATION_COLS,
        BaseValidationOptions,
        BaseValidationReporter,
        CheckProtocol,
        CheckResult,
        FunctionValidationReporter,
        GraphValidationReporter,
        ValidationOutcome,
        ValidationReport,
        ValidationRunner,
        ValidationSeverity,
        apply_severity_overrides,
        cap_findings,
        gateway_timestamp,
        has_error_findings,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve validation exports.

    Parameters
    ----------
    name
        Attribute name requested from this package.

    Returns
    -------
    object
        Resolved attribute value.

    Raises
    ------
    AttributeError
        If ``name`` is not a known export.
    """
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes for tab-completion."""
    return sorted(set(globals()) | set(__all__))

