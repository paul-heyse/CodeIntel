"""Shared HTTP response payload helpers for service mixin tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemProfileResponse,
)
from codeintel.serving.services.errors import ProblemDetail, ProblemError


def make_function_http_responses(meta: ResponseMeta | None = None) -> dict[str, object]:
    """
    Return a basic set of HTTP payloads for function mixin tests.

    Returns
    -------
    dict[str, object]
        Mapping of path to response payloads.
    """
    response_meta = meta or ResponseMeta()
    return {
        "/function/callgraph": CallGraphNeighborsResponse(
            outgoing=[],
            incoming=[],
            meta=response_meta,
        ),
        "/function/tests": {"tests": [], "meta": response_meta.model_dump()},
        "/file/summary": FileSummaryResponse(
            found=True,
            file=None,
            meta=response_meta,
        ),
    }


def make_subsystem_http_responses(meta: ResponseMeta | None = None) -> dict[str, object]:
    """
    Return a basic set of HTTP payloads for subsystem mixin tests.

    Returns
    -------
    dict[str, object]
        Mapping of path to response payloads.
    """
    response_meta = meta or ResponseMeta()
    return {
        "/architecture/subsystems": {
            "subsystems": [],
            "meta": response_meta.model_dump(),
        },
        "/architecture/module-subsystems": {
            "found": True,
            "subsystems": [],
            "meta": response_meta.model_dump(),
        },
        "/ide/hints": {
            "found": True,
            "hints": [],
            "meta": response_meta.model_dump(),
        },
        "/architecture/subsystem": {
            "found": True,
            "modules": [],
            "meta": response_meta.model_dump(),
        },
        "/architecture/subsystem-profiles": SubsystemProfileResponse(
            profiles=[],
            meta=response_meta,
        ),
        "/architecture/subsystem-coverage": SubsystemCoverageResponse(
            coverage=[],
            meta=response_meta,
        ),
    }


class RequestRecorder:
    """Callable HTTP stub that records calls and returns predefined responses."""

    def __init__(
        self,
        responses: dict[str, object],
        *,
        error_paths: set[str] | None = None,
        last_retry_attempts: int = 0,
        problem_factory: Callable[[], ProblemDetail] | None = None,
    ) -> None:
        self.responses = responses
        self.error_paths = error_paths or set()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_retry_attempts = last_retry_attempts
        self.problem_factory = problem_factory or (
            lambda: ProblemDetail(
                type="about:blank", title="missing", detail="not found", status=404
            )
        )

    def request_json(self, path: str, params: dict[str, object]) -> object:
        self.calls.append((path, params))
        if path in self.error_paths:
            raise ProblemError(self.problem_factory())
        return self.responses[path]


def make_retry_sequence(
    *,
    ok_payload: dict[str, object] | None = None,
    retry_payload: dict[str, object] | None = None,
    health_status: int = 200,
    retry_status: int = 500,
) -> list[tuple[int, Mapping[str, object]]]:
    """
    Build a response sequence for retry/circuit tests (health + retry + ok).

    Returns
    -------
    list
        Tuples of (status_code, payload) representing health, retry, and success steps.
    """
    return [
        (health_status, {"ok": True}),
        (
            retry_status,
            retry_payload
            or {
                "type": "about:blank",
                "title": "retry",
                "detail": "server error",
                "status": retry_status,
            },
        ),
        (200, ok_payload or {"ok": True}),
    ]


def make_problem_detail_payload(
    *,
    status: int = 404,
    title: str = "missing",
    detail: str = "not found",
    type_uri: str = "about:blank",
    code: str | None = None,
) -> dict[str, object]:
    """
    Return a problem-detail JSON payload for HTTP error testing.

    Returns
    -------
    dict
        A JSON-serializable problem detail object.
    """
    payload: dict[str, object] = {
        "type": type_uri,
        "title": title,
        "detail": detail,
        "status": status,
    }
    if code is not None:
        payload["code"] = code
    return payload


def assert_scope_serialized(requester: RequestRecorder, path: str) -> None:
    """
    Assert a scope parameter was serialized for the given path.

    Raises
    ------
    AssertionError
        If the scope parameter is missing for the target path.
    """
    for recorded_path, params in requester.calls:
        if recorded_path == path and "scope" in params:
            return
    message = "scope missing for path"
    raise AssertionError(message)


def assert_scope_for_any(requester: RequestRecorder, paths: tuple[str, ...]) -> None:
    """
    Assert a scope parameter was sent for any of the provided paths.

    Raises
    ------
    AssertionError
        If scope is missing for all provided paths.
    """
    for path in paths:
        for recorded_path, params in requester.calls:
            if recorded_path == path and "scope" in params:
                return
    message = "scope missing for all provided paths"
    raise AssertionError(message)


__all__ = [
    "RequestRecorder",
    "assert_scope_for_any",
    "assert_scope_serialized",
    "make_function_http_responses",
    "make_problem_detail_payload",
    "make_retry_sequence",
    "make_subsystem_http_responses",
]
