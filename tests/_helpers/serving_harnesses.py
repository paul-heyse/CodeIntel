"""Public harnesses for serving delegate tests without private imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from codeintel.serving.services.observability import (
    ServiceObservability,
)
from codeintel.serving.services.query_service import HttpQueryService, LocalQueryService
from tests._helpers.serving_stubs import HookedDuckDBQueryApi

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from codeintel.serving.backend import BackendLimits
    from codeintel.serving.services.observability import (
        RequestContext,
        ServiceCallMetrics,
    )

T = TypeVar("T")


class Requester(Protocol):
    """Protocol for objects exposing request_json."""

    def request_json(self, path: str, params: dict[str, object]) -> object: ...


def _normalize_requester(
    requester: Requester | Callable[[str, dict[str, object]], object],
) -> Callable[[str, dict[str, object]], object]:
    candidate = getattr(requester, "request_json", None)
    if callable(candidate):
        return candidate
    if callable(requester):
        return requester
    message = "requester must be callable or expose request_json"
    raise TypeError(message)


class FunctionDelegateHarness(LocalQueryService):
    """Harness around function delegates to drive payloads and record datasets."""

    def __init__(self, payloads: Mapping[str, object]) -> None:
        payload_map = dict(payloads)
        query = HookedDuckDBQueryApi(
            hooks={
                "function_hooks": {
                    "get_function_summary": lambda **_: payload_map["get_function_summary"],
                    "list_high_risk_functions": lambda **_: payload_map["list_high_risk_functions"],
                    "get_callgraph_neighbors": lambda **_: payload_map["get_callgraph_neighbors"],
                    "get_tests_for_function": lambda **_: payload_map["get_tests_for_function"],
                    "get_callgraph_neighborhood": lambda **_: payload_map[
                        "get_callgraph_neighborhood"
                    ],
                    "get_import_boundary": lambda **_: payload_map["get_import_boundary"],
                    "get_function_profile": lambda **_: payload_map.get("get_function_profile"),
                    "get_function_architecture": lambda **_: payload_map.get(
                        "get_function_architecture"
                    ),
                },
                "profile_hooks": {"get_file_summary": lambda **_: payload_map["get_file_summary"]},
            }
        )
        super().__init__(
            query=query,
            dataset_tables={"docs.functions": "docs.v_functions"},
            observability=None,
        )
        self.called: list[tuple[str, str | None]] = []

    def _call(
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        self.called.append((name, dataset))
        return super()._call(
            name,
            func,
            dataset=dataset,
            schema_version=schema_version,
            retries=retries,
        )


class HttpFunctionHarness(HttpQueryService):
    """HTTP harness backed by HttpQueryService with injectable requester."""

    def __init__(
        self,
        *,
        limits: BackendLimits,
        requester: Requester | Callable[[str, dict[str, object]], object],
        observability: ServiceObservability | None = None,
    ) -> None:
        super().__init__(
            request_json=_normalize_requester(requester),
            limits=limits,
            observability=observability,
        )


class SubsystemDelegateHarness(LocalQueryService):
    """Harness around subsystem delegates to drive payloads and record datasets."""

    def __init__(self, payloads: Mapping[str, object]) -> None:
        payload_map = dict(payloads)
        query = HookedDuckDBQueryApi(
            hooks={
                "subsystem_hooks": {
                    "list_subsystems": lambda **_: payload_map["list_subsystems"],
                    "get_module_subsystems": lambda **_: payload_map["get_module_subsystems"],
                    "get_subsystem_modules": lambda **_: payload_map["get_subsystem_modules"],
                    "search_subsystems": lambda **_: payload_map["search_subsystems"],
                    "summarize_subsystem": lambda **_: payload_map["get_subsystem_modules"],
                    "list_subsystem_profiles": lambda **_: payload_map["list_subsystem_profiles"],
                    "list_subsystem_coverage": lambda **_: payload_map["list_subsystem_coverage"],
                },
                "profile_hooks": {"get_file_hints": lambda **_: payload_map["get_file_hints"]},
            }
        )
        super().__init__(
            query=query,
            dataset_tables={"docs.subsystems": "docs.v_subsystems"},
            observability=None,
        )
        self.called: list[tuple[str, str | None]] = []

    def _call(
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        self.called.append((name, dataset))
        return super()._call(
            name,
            func,
            dataset=dataset,
            schema_version=schema_version,
            retries=retries,
        )


class HttpSubsystemHarness(HttpQueryService):
    """HTTP harness backed by HttpQueryService with injectable requester."""

    def __init__(
        self,
        *,
        limits: BackendLimits,
        requester: Requester | Callable[[str, dict[str, object]], object],
    ) -> None:
        super().__init__(
            request_json=_normalize_requester(requester),
            limits=limits,
            observability=None,
        )


class RecordingObservability(ServiceObservability):
    """ServiceObservability stub capturing emitted metrics for assertions."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.records: list[ServiceCallMetrics] = []

    def record(
        self,
        metrics: ServiceCallMetrics,
        context: RequestContext | None = None,
    ) -> None:
        _ = context
        self.records.append(metrics)


__all__ = [
    "FunctionDelegateHarness",
    "HttpFunctionHarness",
    "HttpSubsystemHarness",
    "RecordingObservability",
    "SubsystemDelegateHarness",
]
