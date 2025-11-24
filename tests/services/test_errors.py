"""Unit tests for ProblemDetail helpers and taxonomy."""

from __future__ import annotations

import json
import logging

import pytest

from codeintel.services.errors import (
    ExportError,
    PipelineError,
    SchemaDriftError,
    ValidationError,
    generate_correlation_id,
    log_problem,
    problem,
)


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    pytest.fail(detail)


def test_problem_defaults_and_to_dict() -> None:
    """problem() should populate defaults and serialize to dict."""
    pd = problem(code="demo.code", title="Title", detail="Something went wrong")
    data = pd.to_dict()
    _expect(condition=data["code"] == "demo.code", detail="code mismatch")
    _expect(condition=data["title"] == "Title", detail="title mismatch")
    _expect(condition=data["detail"] == "Something went wrong", detail="detail mismatch")
    _expect(condition=bool(data["instance"]), detail="missing instance")
    _expect(condition=data["type"].endswith("demo.code"), detail="type uri mismatch")


def test_problem_allows_override_instance_and_type() -> None:
    """problem() should honor provided instance/type/status/extras."""
    custom_instance = generate_correlation_id()
    status_code = 400
    pd = problem(
        code="custom",
        title="Custom",
        detail="With overrides",
        status=status_code,
        instance=custom_instance,
        type_uri="https://example.com/problem/custom",
        extras={"foo": "bar"},
    )
    data = pd.to_dict()
    _expect(condition=data["instance"] == custom_instance, detail="instance mismatch")
    _expect(condition=data["type"] == "https://example.com/problem/custom", detail="type mismatch")
    _expect(condition=data["status"] == status_code, detail="status mismatch")
    _expect(condition=data["extras"] == {"foo": "bar"}, detail="extras mismatch")


def test_problem_error_taxonomy_subclasses() -> None:
    """
    ProblemError subclasses should carry the ProblemDetail payload.

    Raises
    ------
    PipelineError
        When the pipeline error is raised intentionally for testing.
    """
    pd = problem(code="pipeline.failed", title="Pipeline failed", detail="trace")
    with pytest.raises(PipelineError) as excinfo:
        raise PipelineError(pd)
    _expect(condition=excinfo.value.problem_detail == pd, detail="problem_detail not stored")

    for exc_cls in (ExportError, SchemaDriftError, ValidationError):
        with pytest.raises(exc_cls):
            raise exc_cls(pd)


def test_log_problem_emits_json(caplog: pytest.LogCaptureFixture) -> None:
    """log_problem should emit JSON with the problem code."""
    logger_name = "test-log-problem"
    caplog.set_level("ERROR", logger=logger_name)
    logger = logging.getLogger(logger_name)
    pd = problem(code="demo.log", title="Log", detail="Logged")
    log_problem(logger, pd)
    _expect(condition=bool(caplog.records), detail="no log records captured")
    payload = json.loads(caplog.records[0].message)
    _expect(condition=payload["code"] == "demo.log", detail="code mismatch in log")
