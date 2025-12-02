"""Graph plugin result types.

This module defines the result types returned by graph plugin execution,
providing consistent success/failure/skip semantics across all graph plugins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class GraphPluginResult:
    """Result returned by graph plugin execution.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
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
    artifacts
        Mapping of artifact names to paths produced.
    """

    success: bool = True
    row_counts: Mapping[str, int] | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    error: str | None = None
    error_kind: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    artifacts: Mapping[str, Path] | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def ok(
        *,
        row_counts: Mapping[str, int] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
        artifacts: Mapping[str, Path] | None = None,
        meta: dict[str, object] | None = None,
    ) -> GraphPluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.
        artifacts
            Optional mapping of artifact names to paths.
        meta
            Optional additional metadata.

        Returns
        -------
        GraphPluginResult
            Result object marked as successful.
        """
        return GraphPluginResult(
            success=True,
            row_counts=row_counts,
            input_hash=input_hash,
            options_hash=options_hash,
            artifacts=artifacts,
            meta=meta or {},
        )

    @staticmethod
    def fail(error: str, *, error_kind: str | None = None) -> GraphPluginResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        error_kind
            Optional classification of the error type.

        Returns
        -------
        GraphPluginResult
            Result object marked as failed.
        """
        return GraphPluginResult(success=False, error=error, error_kind=error_kind)

    @staticmethod
    def skip(reason: str) -> GraphPluginResult:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping execution.

        Returns
        -------
        GraphPluginResult
            Result object marked as skipped.
        """
        return GraphPluginResult(success=True, skipped=True, skip_reason=reason)


GraphPluginStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class GraphPluginRunRecord:
    """Record of a single graph plugin execution.

    Attributes
    ----------
    name
        Plugin name.
    status
        Execution status.
    started_at
        When execution started (ISO format).
    ended_at
        When execution ended (ISO format).
    duration_ms
        Execution duration in milliseconds.
    attempts
        Number of execution attempts.
    partial
        Whether the result is partial.
    error
        Error message if failed.
    meta
        Additional metadata from execution.
    """

    name: str
    status: GraphPluginStatus
    started_at: str
    ended_at: str
    duration_ms: float
    attempts: int = 1
    partial: bool = False
    error: str | None = None
    meta: dict[str, object] = field(default_factory=dict)


__all__ = [
    "GraphPluginResult",
    "GraphPluginRunRecord",
    "GraphPluginStatus",
]
