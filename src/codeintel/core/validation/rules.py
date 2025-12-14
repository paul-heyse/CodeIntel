"""Composable validation rule engine.

This module provides a composable validation rule engine
for validating data across domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Severity(Enum):
    """Validation issue severity.

    Attributes
    ----------
    ERROR
        Validation error (blocks processing).
    WARNING
        Validation warning (advisory).
    INFO
        Informational note.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    """A validation issue.

    Attributes
    ----------
    rule
        Rule that found the issue.
    message
        Issue description.
    severity
        Issue severity.
    path
        Path to the problematic field.
    value
        The problematic value.
    """

    rule: str
    message: str
    severity: Severity = Severity.ERROR
    path: str | None = None
    value: object = None


@runtime_checkable
class ValidationRule(Protocol):
    """Protocol for validation rules.

    Examples
    --------
    >>> class RequiredRule:
    ...     RULE_NAME = "required"
    ...
    ...     def validate(self, value: object, path: str) -> list[ValidationIssue]:
    ...         if value is None:
    ...             return [ValidationIssue("required", f"{path} is required")]
    ...         return []
    """

    RULE_NAME: str

    def validate(self, value: object, path: str) -> list[ValidationIssue]:
        """Validate a value.

        Parameters
        ----------
        value
            Value to validate.
        path
            Path to the value.

        Returns
        -------
        list[ValidationIssue]
            List of validation issues.
        """
        ...


@dataclass
class RuleResult:
    """Result of validation rule execution.

    Attributes
    ----------
    issues
        List of validation issues.
    rules_run
        Number of rules executed.
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    rules_run: int = 0

    @property
    def is_valid(self) -> bool:
        """Check if validation passed.

        Returns
        -------
        bool
            True if no errors.
        """
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all errors.

        Returns
        -------
        list[ValidationIssue]
            Error issues.
        """
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warnings.

        Returns
        -------
        list[ValidationIssue]
            Warning issues.
        """
        return [i for i in self.issues if i.severity == Severity.WARNING]


class RuleEngine:
    """Composable validation rule engine.

    Examples
    --------
    >>> engine = RuleEngine()
    >>> engine.register(RequiredRule())
    >>> engine.register(TypeRule(int))
    >>> result = engine.validate(data, "config")
    """

    def __init__(self) -> None:
        """Initialize the rule engine."""
        self._rules: list[ValidationRule] = []

    def register(self, rule: ValidationRule) -> None:
        """Register a validation rule.

        Parameters
        ----------
        rule
            Rule to register.
        """
        self._rules.append(rule)

    def validate(self, value: object, path: str = "") -> RuleResult:
        """Validate a value against all rules.

        Parameters
        ----------
        value
            Value to validate.
        path
            Path to the value.

        Returns
        -------
        RuleResult
            Validation result.
        """
        result = RuleResult()

        for rule in self._rules:
            issues = rule.validate(value, path)
            result.issues.extend(issues)
            result.rules_run += 1

        return result

    def clear(self) -> None:
        """Remove all registered rules."""
        self._rules.clear()

    @property
    def rules(self) -> tuple[ValidationRule, ...]:
        """Get registered rules.

        Returns
        -------
        tuple[ValidationRule, ...]
            Registered rules.
        """
        return tuple(self._rules)


def make_rule(
    name: str,
    validator: Callable[[object, str], Sequence[ValidationIssue]],
) -> ValidationRule:
    """Create a validation rule from a function.

    Parameters
    ----------
    name
        Rule name.
    validator
        Validation function.

    Returns
    -------
    ValidationRule
        A validation rule.

    Examples
    --------
    >>> def check_positive(value: object, path: str) -> list[ValidationIssue]:
    ...     if isinstance(value, int) and value < 0:
    ...         return [ValidationIssue("positive", f"{path} must be positive")]
    ...     return []
    >>> rule = make_rule("positive", check_positive)
    """

    class FunctionRule:
        RULE_NAME = name
        _validator = staticmethod(validator)

        def validate(self, value: object, path: str) -> list[ValidationIssue]:
            return list(self._validator(value, path))

    return FunctionRule()


__all__ = [
    "RuleEngine",
    "RuleResult",
    "Severity",
    "ValidationIssue",
    "ValidationRule",
    "make_rule",
]
