"""Unified plugin result types.

This module defines the result types returned by plugin execution,
providing consistent success/failure/skip semantics across all plugins.

Architecture
------------
- BasePluginResult: Common fields shared across all domains
- PluginResult: Full-featured result for graphs/analytics
- IngestPluginResult (in ingestion): Extends base with ingestion-specific fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

PluginStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class BasePluginResult:
    """Base result type for all plugin executions.

    Provide the common fields shared across all plugin result types.
    Domain-specific result types (PluginResult, IngestPluginResult)
    extend this base with additional fields.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
    error
        Error message if execution failed.
    error_kind
        Classification of the error type.
    skipped
        Whether the plugin was skipped.
    skip_reason
        Reason for skipping if applicable.
    input_hash
        Hash of inputs for caching.
    options_hash
        Hash of options for caching.
    """

    success: bool = True
    row_counts: Mapping[str, int] | None = None
    error: str | None = None
    error_kind: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    input_hash: str | None = None
    options_hash: str | None = None

    @classmethod
    def ok(
        cls,
        *,
        row_counts: Mapping[str, int] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
    ) -> Self:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.

        Returns
        -------
        Self
            Result object marked as successful.
        """
        return cls(
            success=True,
            row_counts=row_counts,
            input_hash=input_hash,
            options_hash=options_hash,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        error_kind: str | None = None,
    ) -> Self:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        error_kind
            Optional classification of the error type.

        Returns
        -------
        Self
            Result object marked as failed.
        """
        return cls(success=False, error=error, error_kind=error_kind)

    @classmethod
    def skip(cls, reason: str) -> Self:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping execution.

        Returns
        -------
        Self
            Result object marked as skipped.
        """
        return cls(success=True, skipped=True, skip_reason=reason)

    @property
    def status(self) -> PluginStatus:
        """Derive the execution status from result fields.

        Returns
        -------
        PluginStatus
            The status of the plugin execution.
        """
        if self.skipped:
            return "skipped"
        return "succeeded" if self.success else "failed"


@dataclass(frozen=True)
class PluginResult(BasePluginResult):
    """Full-featured result for graph and analytics plugin execution.

    Extend BasePluginResult with additional fields for artifacts, warnings,
    and metadata commonly used in graph and analytics plugins.

    Attributes
    ----------
    artifacts
        Mapping of artifact names to artifact data or paths.
    warnings
        Non-fatal warnings from execution.
    meta
        Additional metadata about the execution.
    """

    # Additional fields beyond BasePluginResult
    artifacts: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, object] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> PluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        artifacts
            Optional mapping of produced artifacts.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.
        meta
            Optional execution metadata.

        Returns
        -------
        PluginResult
            Result object marked as successful.
        """
        return cls(
            success=True,
            row_counts=row_counts,
            artifacts=artifacts or {},
            input_hash=input_hash,
            options_hash=options_hash,
            meta=meta or {},
        )

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        error_kind: str | None = None,
        warnings: tuple[str, ...] = (),
    ) -> PluginResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        error_kind
            Optional classification of the error type.
        warnings
            Optional non-fatal warnings collected during execution.

        Returns
        -------
        PluginResult
            Result object marked as failed.
        """
        return cls(
            success=False,
            error=error,
            error_kind=error_kind,
            warnings=warnings,
        )


@dataclass
class BasePluginExecutionRecord:
    """Base execution record for all domains.

    Provide common fields and computed duration properties for plugin execution
    records. Domain-specific record types extend this base with additional fields.

    Extension Points
    ----------------
    Domain-specific execution record classes should extend this base:
    - PluginExecutionRecord (core): Frozen, millisecond-based, for graphs/analytics
    - IngestPluginExecutionRecord (ingestion): Mutable, with row tracking

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    started_at
        Timestamp when execution started.
    ended_at
        Timestamp when execution ended, or None if still running.
    """

    plugin_name: str
    started_at: datetime
    ended_at: datetime | None = None

    @property
    def duration_s(self) -> float:
        """Compute duration in seconds.

        Returns
        -------
        float
            Duration in seconds, or 0.0 if ended_at is None.
        """
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def computed_duration_ms(self) -> float:
        """Compute duration in milliseconds.

        Returns
        -------
        float
            Duration in milliseconds, or 0.0 if ended_at is None.
        """
        return self.duration_s * 1000


@dataclass(frozen=True)
class PluginExecutionRecord:
    """Record of a single plugin execution for graphs and analytics.

    This is the canonical execution record type for graphs and analytics domains.
    It is frozen (immutable) and stores duration directly in milliseconds.

    For ingestion-specific records with row tracking, see IngestPluginExecutionRecord
    in codeintel.ingestion.runtime.executor.

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    status
        Execution status.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Execution duration in milliseconds.
    attempts
        Number of execution attempts.
    partial
        Whether the result is partial.
    result
        Plugin result if available.
    error
        Error message if failed.
    meta
        Additional metadata from execution.
    """

    plugin_name: str
    status: PluginStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int = 1
    partial: bool = False
    result: PluginResult | None = None
    error: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginArtifact:
    """Reference to an artifact produced by a plugin.

    Attributes
    ----------
    name
        Artifact name.
    path
        Path to artifact file if applicable.
    data
        In-memory artifact data if applicable.
    artifact_type
        Type classification for the artifact.
    """

    name: str
    path: Path | None = None
    data: object | None = None
    artifact_type: str | None = None


__all__ = [
    "BasePluginExecutionRecord",
    "BasePluginResult",
    "PluginArtifact",
    "PluginExecutionRecord",
    "PluginResult",
    "PluginStatus",
]
