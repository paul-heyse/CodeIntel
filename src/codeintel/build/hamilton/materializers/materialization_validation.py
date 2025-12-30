"""Materialization-time validation helpers for columnar outputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa

from codeintel.build.hamilton.materializers.validation_policy import ValidationScope
from codeintel.build.schemas.observations import SchemaObservationAccumulator
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.serialization.stable import stable_stringify

ValidationSeverity = Literal["error", "warning"]
ValidationStatus = Literal["passed", "warned", "failed", "skipped"]

DEFAULT_PK_UNIQUENESS_MAX_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Single validation issue captured during materialization."""

    code: str
    message: str
    severity: ValidationSeverity
    details: Mapping[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for this issue.

        Returns
        -------
        dict[str, object]
            Payload with the issue details.
        """
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate validation report for a materialized output."""

    table_key: str
    target_name: str
    output_role: str
    scope: ValidationScope
    profile: str | None
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]
    checks: Mapping[str, object]
    skipped_checks: Mapping[str, str]
    row_count: int | None = None

    def issues_payload(self) -> list[dict[str, object]]:
        """Return issue payloads for persistence.

        Returns
        -------
        list[dict[str, object]]
            Issue payloads derived from stored validation issues.
        """
        return [issue.to_payload() for issue in self.issues]


@dataclass(slots=True)
class ValidationCollector:
    """Mutable collector for validation issues and check metadata."""

    table_key: str
    target_name: str
    output_role: str
    scope: ValidationScope
    profile: str | None
    issues: list[ValidationIssue] = field(default_factory=list)
    checks: dict[str, dict[str, object]] = field(default_factory=dict)
    skipped_checks: dict[str, str] = field(default_factory=dict)

    def record_issue(
        self,
        *,
        code: str,
        message: str,
        severity: ValidationSeverity,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Record a validation issue on the collector."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=severity,
                details=details,
            )
        )

    def record_check(
        self,
        *,
        name: str,
        status: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Record a check outcome unless a prior failure is recorded."""
        current = self.checks.get(name)
        if current is not None and current.get("status") == "failed":
            return
        payload: dict[str, object] = {"status": status}
        if details:
            payload.update(details)
        self.checks[name] = payload

    def ensure_check(self, *, name: str, status: str) -> None:
        """Ensure a check is present with the provided status."""
        if name in self.checks or name in self.skipped_checks:
            return
        self.checks[name] = {"status": status}

    def skip_check(self, *, name: str, reason: str) -> None:
        """Record that a check was skipped with a reason."""
        if name in self.checks:
            return
        self.skipped_checks[name] = reason

    def finalize(self, *, row_count: int | None) -> ValidationReport:
        """Freeze the collector into an immutable validation report.

        Returns
        -------
        ValidationReport
            Immutable report constructed from the collector state.
        """
        status = _status_from_issues(
            issues=self.issues,
            checks=self.checks,
            skipped_checks=self.skipped_checks,
        )
        return ValidationReport(
            table_key=self.table_key,
            target_name=self.target_name,
            output_role=self.output_role,
            scope=self.scope,
            profile=self.profile,
            status=status,
            issues=tuple(self.issues),
            checks=dict(self.checks),
            skipped_checks=dict(self.skipped_checks),
            row_count=row_count,
        )


@dataclass(frozen=True, slots=True)
class ContractCheckContext:
    """Inputs required to apply contract checks."""

    collector: ValidationCollector
    declared_schema: TableSchema | None
    arrow_schema: pa.Schema
    observation: SchemaObservationAccumulator
    row_count: int | None
    min_rows: int


@dataclass(slots=True)
class PrimaryKeyTracker:
    """Streaming tracker for primary key uniqueness checks."""

    primary_keys: tuple[str, ...]
    max_rows: int
    row_count: int = 0
    seen: set[tuple[object, ...]] = field(default_factory=set)
    duplicate_key: tuple[object, ...] | None = None
    skipped_reason: str | None = None
    missing_columns: tuple[str, ...] | None = None

    def observe_batch(self, batch: pa.RecordBatch) -> None:
        """Observe a batch to update primary key uniqueness tracking."""
        if self.skipped_reason or self.duplicate_key is not None:
            return
        if not self.primary_keys:
            self.skipped_reason = "primary_key_not_defined"
            return
        if self.max_rows <= 0:
            self.skipped_reason = "primary_key_check_disabled"
            return
        observed_columns = set(batch.schema.names)
        missing = [col for col in self.primary_keys if col not in observed_columns]
        if missing:
            self.missing_columns = tuple(missing)
            self.skipped_reason = "primary_key_columns_missing"
            return
        self.row_count += batch.num_rows
        if self.row_count > self.max_rows:
            self.skipped_reason = f"row_count_exceeds_{self.max_rows}"
            self.seen.clear()
            return
        values_by_column = [
            _column_values(batch=batch, column=column) for column in self.primary_keys
        ]
        for key_tuple in zip(*values_by_column, strict=True):
            key = tuple(_hashable_value(value) for value in key_tuple)
            if key in self.seen:
                self.duplicate_key = key
                return
            self.seen.add(key)


def wrap_reader_for_validation(
    reader: pa.RecordBatchReader,
    *,
    collector: ValidationCollector,
    pk_tracker: PrimaryKeyTracker | None,
) -> pa.RecordBatchReader:
    """Wrap a RecordBatchReader to validate batches during iteration.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader that validates batches while iterating.
    """

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for batch in reader:
            _validate_arrow_batch(batch=batch, collector=collector)
            if pk_tracker is not None:
                pk_tracker.observe_batch(batch)
            yield batch
        collector.ensure_check(name="arrow_integrity", status="passed")

    return pa.RecordBatchReader.from_batches(reader.schema, _iter_batches())


def finalize_primary_key_check(
    *,
    collector: ValidationCollector,
    tracker: PrimaryKeyTracker | None,
    severity: ValidationSeverity,
) -> None:
    """Add primary key uniqueness results to the collector."""
    if tracker is None:
        if "primary_key_uniqueness" in collector.skipped_checks:
            return
        collector.skip_check(name="primary_key_uniqueness", reason="primary_key_not_configured")
        return
    if tracker.skipped_reason is not None:
        reason = tracker.skipped_reason
        if tracker.missing_columns:
            reason = f"{reason}: {', '.join(tracker.missing_columns)}"
        collector.skip_check(name="primary_key_uniqueness", reason=reason)
        return
    if tracker.duplicate_key is not None:
        collector.record_issue(
            code="primary_key_duplicate",
            message="Primary key uniqueness violated",
            severity=severity,
            details={
                "primary_keys": list(tracker.primary_keys),
                "duplicate_key": list(tracker.duplicate_key),
            },
        )
        collector.record_check(
            name="primary_key_uniqueness",
            status="failed",
            details={"row_count": tracker.row_count},
        )
        return
    collector.record_check(
        name="primary_key_uniqueness",
        status="passed",
        details={"row_count": tracker.row_count},
    )


def apply_contract_checks(*, context: ContractCheckContext) -> None:
    """Run contract checks and append results to the collector."""
    severity = _severity_for_profile(context.collector.profile)
    _check_missing_columns(context=context, severity=severity)
    _check_nullability(context=context, severity=severity)
    _check_min_rows(context=context, severity=severity)


def _check_missing_columns(*, context: ContractCheckContext, severity: ValidationSeverity) -> None:
    collector = context.collector
    if context.declared_schema is None:
        collector.skip_check(name="contract_columns", reason="missing_declared_schema")
        return
    observed = set(context.arrow_schema.names)
    missing = [col.name for col in context.declared_schema.columns if col.name not in observed]
    if missing:
        collector.record_issue(
            code="missing_columns",
            message="Missing required columns",
            severity=severity,
            details={"missing_columns": missing},
        )
        collector.record_check(
            name="contract_columns",
            status="failed",
            details={"missing_columns": missing},
        )
        return
    collector.record_check(name="contract_columns", status="passed")


def _check_nullability(*, context: ContractCheckContext, severity: ValidationSeverity) -> None:
    collector = context.collector
    if context.declared_schema is None:
        collector.skip_check(name="contract_nullability", reason="missing_declared_schema")
        return
    if context.row_count is None:
        collector.skip_check(name="contract_nullability", reason="row_count_unavailable")
        return
    if context.row_count == 0:
        collector.skip_check(name="contract_nullability", reason="row_count_zero")
        return
    observed = set(context.arrow_schema.names)
    non_nullable = [col.name for col in context.declared_schema.columns if not col.nullable]
    violations: list[str] = []
    for column in non_nullable:
        if column not in observed:
            continue
        stats = context.observation.column_stats.get(column)
        if stats is None:
            continue
        if stats.null_count > 0:
            violations.append(column)
    if violations:
        collector.record_issue(
            code="nullability_violation",
            message="Non-nullable columns contain nulls",
            severity=severity,
            details={"nullable_violations": violations},
        )
        collector.record_check(
            name="contract_nullability",
            status="failed",
            details={"nullable_violations": violations},
        )
        return
    collector.record_check(name="contract_nullability", status="passed")


def _check_min_rows(*, context: ContractCheckContext, severity: ValidationSeverity) -> None:
    collector = context.collector
    if context.min_rows <= 0:
        collector.skip_check(name="contract_min_rows", reason="min_rows_disabled")
        return
    if context.row_count is None:
        collector.skip_check(name="contract_min_rows", reason="row_count_unavailable")
        return
    if context.row_count < context.min_rows:
        collector.record_issue(
            code="min_rows",
            message="Row count below configured minimum",
            severity=severity,
            details={
                "row_count": context.row_count,
                "min_rows": context.min_rows,
            },
        )
        collector.record_check(
            name="contract_min_rows",
            status="failed",
            details={
                "row_count": context.row_count,
                "min_rows": context.min_rows,
            },
        )
        return
    collector.record_check(
        name="contract_min_rows",
        status="passed",
        details={
            "row_count": context.row_count,
            "min_rows": context.min_rows,
        },
    )


def _validate_arrow_batch(*, batch: pa.RecordBatch, collector: ValidationCollector) -> None:
    try:
        batch.validate(full=True)
    except (pa.ArrowInvalid, TypeError, ValueError) as exc:
        collector.record_issue(
            code="arrow_integrity",
            message="Arrow batch validation failed",
            severity="error",
            details={"error": str(exc)},
        )
        collector.record_check(
            name="arrow_integrity",
            status="failed",
            details={"error": str(exc)},
        )


def _status_from_issues(
    *,
    issues: list[ValidationIssue],
    checks: Mapping[str, object],
    skipped_checks: Mapping[str, str],
) -> ValidationStatus:
    if not checks and not issues and skipped_checks:
        return "skipped"
    if not checks and not issues:
        return "skipped"
    if any(issue.severity == "error" for issue in issues):
        return "failed"
    if any(issue.severity == "warning" for issue in issues):
        return "warned"
    return "passed"


def _severity_for_profile(profile: str | None) -> ValidationSeverity:
    if profile == "strict":
        return "error"
    return "warning"


def _column_values(*, batch: pa.RecordBatch, column: str) -> list[object]:
    index = batch.schema.get_field_index(column)
    if index < 0:
        return []
    return list(batch.column(index).to_pylist())


def _hashable_value(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        try:
            return stable_stringify(value)
        except (TypeError, ValueError):
            return repr(value)
    else:
        return value


__all__ = [
    "DEFAULT_PK_UNIQUENESS_MAX_ROWS",
    "ContractCheckContext",
    "PrimaryKeyTracker",
    "ValidationCollector",
    "ValidationIssue",
    "ValidationReport",
    "ValidationScope",
    "apply_contract_checks",
    "finalize_primary_key_check",
    "wrap_reader_for_validation",
]
