"""Lazy loader for legacy test analytics module to avoid circular imports."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast

from codeintel.analytics.tests_profiles.types import (
    FunctionCoverageEntryProtocol,
    SubsystemCoverageEntryProtocol,
    TestGraphMetricsProtocol,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.storage.gateway import DuckDBConnection

__all__ = ["legacy"]


@lru_cache(maxsize=1)
def _load_legacy() -> ModuleType:
    return import_module("codeintel.analytics.tests.profiles")


class _LegacyModule(Protocol):
    DEFAULT_IO_SPEC: dict[str, object]
    CONCURRENCY_LIBS: object

    def load_test_records_public(
        self, con: DuckDBConnection, repo: str, commit: str
    ) -> list[TestRecord]: ...

    def load_functions_covered(
        self, con: DuckDBConnection, repo: str, commit: str
    ) -> dict[str, FunctionCoverageEntryProtocol]: ...

    def load_subsystems_covered(
        self, con: DuckDBConnection, repo: str, commit: str
    ) -> dict[str, SubsystemCoverageEntryProtocol]: ...

    def load_test_graph_metrics_public(
        self, con: DuckDBConnection, repo: str, commit: str
    ) -> dict[str, TestGraphMetricsProtocol]: ...

    def build_behavior_row(self, test: TestRecord, ctx: object) -> tuple[object, ...]: ...

    def build_test_ast_index(
        self,
        repo_root: object,
        tests: list[TestRecord],
        io_spec: dict[str, object],
        concurrency_libs: object,
    ) -> object: ...

    def load_test_profile_context(
        self, con: DuckDBConnection, repo: str, commit: str
    ) -> object: ...

    BehavioralContext: Callable[..., object]

    def infer_behavior_tags(
        self, *, name: str, markers: object, io_flags: object, ast_info: object
    ) -> list[str]: ...
    def build_test_profile_row(self, test: TestRecord, ctx: object) -> tuple[object, ...]: ...

    def build_behavioral_coverage(
        self,
        gateway: object,
        cfg: BehavioralCoverageStepConfig,
        *,
        llm_runner: object | None = None,
    ) -> int: ...

    def build_test_profile(self, gateway: object, cfg: TestProfileStepConfig) -> int: ...

    def compute_test_coverage_edges(self, *args: object, **kwargs: object) -> object: ...
    def compute_test_graph_metrics(self, *args: object, **kwargs: object) -> object: ...
    def compute_flakiness_score(
        self,
        *,
        status: str | None,
        markers: list[str],
        duration_ms: float | None,
        io_flags: object,
        slow_test_threshold_ms: float,
    ) -> float | None: ...

    def compute_importance_score(self, inputs: object) -> float | None: ...


class _LegacyProxy:
    """Proxy that defers attribute access to the legacy module."""

    __slots__ = ()

    def __getattr__(self, name: str) -> object:
        return getattr(_load_legacy(), name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(_load_legacy(), name, value)


legacy: _LegacyModule = cast("_LegacyModule", _LegacyProxy())
