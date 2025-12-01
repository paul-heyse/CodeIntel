"""Shared error taxonomy and Problem Details helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from codeintel.serving.context import get_current_request_context


def generate_correlation_id() -> str:
    """
    Return a new correlation identifier for tracing errors.

    If a RequestContext is active, reuse its correlation_id; otherwise generate a
    new identifier.

    Returns
    -------
    str
        UUID4 correlation identifier or the active RequestContext correlation id.
    """
    ctx = get_current_request_context()
    if ctx is not None and ctx.correlation_id:
        return ctx.correlation_id
    return str(uuid4())


@dataclass(frozen=True)
class ProblemDetail:
    """
    Canonical domain-level Problem Details representation.

    Mirrors RFC 9457/RFC 7807 plus:
    - code: short machine code ("dataset-not-found")
    - extras: arbitrary diagnostic payload.
    """

    type: str = "about:blank"
    title: str = ""
    detail: str | None = None
    status: int | None = None
    instance: str = field(default_factory=generate_correlation_id)
    code: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            Problem detail payload as a plain dictionary.
        """
        payload: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "detail": self.detail,
            "instance": self.instance,
        }
        if self.status is not None:
            payload["status"] = self.status
        if self.code is not None:
            payload["code"] = self.code
        if self.extras:
            payload["extras"] = self.extras
        return payload


@dataclass(frozen=True)
class ProblemDetailParams:
    """Optional parameters for constructing a ProblemDetail."""

    status: int | None = None
    instance: str | None = None
    type_uri: str | None = None
    extras: dict[str, Any] | None = None


def problem(
    code: str,
    title: str,
    detail: str | None,
    params: ProblemDetailParams | None = None,
) -> ProblemDetail:
    """
    Create a ProblemDetail with defaults for type/instance.

    Parameters
    ----------
    code
        Stable problem code (e.g., 'pipeline.task_failed').
    title
        Human-readable error summary.
    detail
        Detailed description of the error.
    params
        Optional bundle containing status, instance, type URI, and extras.

    Returns
    -------
    ProblemDetail
        Structured problem payload.
    """
    effective = params or ProblemDetailParams()
    resolved_instance = effective.instance or generate_correlation_id()
    resolved_type = effective.type_uri or f"https://problems.codeintel.dev/{code}"
    return ProblemDetail(
        type=resolved_type,
        title=title,
        detail=detail,
        status=effective.status,
        instance=resolved_instance,
        code=code,
        extras=effective.extras or {},
    )


def log_problem(logger: logging.Logger | logging.LoggerAdapter, detail: ProblemDetail) -> None:
    """Emit a Problem Detail as a structured error log."""
    logger.error(json.dumps(detail.to_dict()))


class ProblemError(Exception):
    """Base exception carrying a ProblemDetail payload."""

    def __init__(self, detail: ProblemDetail) -> None:
        self.detail = detail
        super().__init__(detail.detail or detail.title)

    @property
    def problem_detail(self) -> ProblemDetail:
        """
        Return the attached problem detail.

        This alias exists for backward compatibility with callers expecting a
        ``problem_detail`` attribute.
        """
        return self.detail


class PipelineError(ProblemError):
    """Pipeline execution failure."""


class ExportError(ProblemError):
    """Export/validation failure."""


class SchemaDriftError(ProblemError):
    """Schema drift detected between expected and actual datasets."""


class ValidationError(ProblemError):
    """Input or configuration validation failure."""


class DatasetNotFoundError(ProblemError):
    """Requested dataset could not be located."""

    @classmethod
    def for_name(cls, dataset_name: str) -> DatasetNotFoundError:
        """
        Build a dataset-not-found problem for a logical dataset name.

        Returns
        -------
        DatasetNotFoundError
            Structured problem error with dataset context.
        """
        return cls(
            ProblemDetail(
                type="https://codeintel/problems/dataset-not-found",
                title="Invalid argument",
                detail=f"Unknown dataset: {dataset_name}",
                status=400,
                code="dataset-not-found",
                extras={"dataset": dataset_name},
            )
        )


class DatasetSchemaDriftError(ProblemError):
    """Schema drift detected between expected and actual datasets."""


class GraphScopeError(ProblemError):
    """Invalid or unsupported graph scope."""


class GraphFeatureDisabledError(ProblemError):
    """Graph-related feature is disabled in current configuration."""


class BackendTimeoutError(ProblemError):
    """Backend operation exceeded its allowed time budget."""


__all__ = [
    "BackendTimeoutError",
    "DatasetNotFoundError",
    "DatasetSchemaDriftError",
    "ExportError",
    "GraphFeatureDisabledError",
    "GraphScopeError",
    "PipelineError",
    "ProblemDetail",
    "ProblemError",
    "SchemaDriftError",
    "ValidationError",
    "generate_correlation_id",
    "log_problem",
    "problem",
]
