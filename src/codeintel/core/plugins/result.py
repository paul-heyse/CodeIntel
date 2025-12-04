"""Unified plugin result types.

This module defines the result types returned by plugin execution,
providing consistent success/failure/skip semantics across all plugins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

PluginStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class PluginResult:
    """Result returned by plugin execution.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
    artifacts
        Mapping of artifact names to artifact data or paths.
    input_hash
        Hash of inputs for caching.
    options_hash
        Hash of options for caching.
    error
        Error message if execution failed.
    error_kind
        Classification of the error type.
    skipped
        Whether the plugin was skipped.
    skip_reason
        Reason for skipping if applicable.
    warnings
        Non-fatal warnings from execution.
    meta
        Additional metadata about the execution.
    """

    success: bool = True
    row_counts: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)
    input_hash: str | None = None
    options_hash: str | None = None
    error: str | None = None
    error_kind: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    warnings: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def ok(
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
        return PluginResult(
            success=True,
            row_counts=row_counts or {},
            artifacts=artifacts or {},
            input_hash=input_hash,
            options_hash=options_hash,
            meta=meta or {},
        )

    @staticmethod
    def fail(
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
        return PluginResult(
            success=False,
            error=error,
            error_kind=error_kind,
            warnings=warnings,
        )

    @staticmethod
    def skip(reason: str) -> PluginResult:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping execution.

        Returns
        -------
        PluginResult
            Result object marked as skipped.
        """
        return PluginResult(success=True, skipped=True, skip_reason=reason)

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
class PluginExecutionRecord:
    """Record of a single plugin execution.

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
    "PluginArtifact",
    "PluginExecutionRecord",
    "PluginResult",
    "PluginStatus",
]
