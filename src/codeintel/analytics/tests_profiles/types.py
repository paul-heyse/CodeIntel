"""Shared type definitions for test analytics profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig


class FunctionCoverageEntryProtocol(Protocol):
    """Structural type for test→function coverage entries."""

    @property
    def functions(self) -> list[dict[str, object]]: ...

    @property
    def count(self) -> int: ...

    @property
    def primary(self) -> list[int]: ...


class SubsystemCoverageEntryProtocol(Protocol):
    """Structural type for test→subsystem coverage entries."""

    @property
    def subsystems(self) -> list[dict[str, object]]: ...

    @property
    def count(self) -> int: ...

    @property
    def primary_subsystem_id(self) -> str | None: ...

    @property
    def max_risk_score(self) -> float | None: ...


class TestGraphMetricsProtocol(Protocol):
    """Structural type for graph metrics on test nodes."""

    @property
    def degree(self) -> int | None: ...

    @property
    def weighted_degree(self) -> float | None: ...

    @property
    def proj_degree(self) -> int | None: ...

    @property
    def proj_weight(self) -> float | None: ...

    @property
    def proj_clustering(self) -> float | None: ...

    @property
    def proj_betweenness(self) -> float | None: ...


@dataclass(frozen=True)
class IoFlags:
    """Flags describing IO usage within a test."""

    uses_network: bool = False
    uses_db: bool = False
    uses_filesystem: bool = False
    uses_subprocess: bool = False

    @property
    def io_bound(self) -> bool:
        """Return True when any IO flag is set."""
        return self.uses_network or self.uses_db or self.uses_filesystem or self.uses_subprocess


@dataclass(frozen=True)
class TestAstInfo:
    """AST-derived metrics for a single test span."""

    __test__ = False

    assert_count: int = 0
    raise_count: int = 0
    uses_pytest_raises: bool = False
    uses_concurrency_lib: bool = False
    has_boundary_asserts: bool = False
    uses_fixtures: bool = False
    io_flags: IoFlags = IoFlags()


@dataclass(frozen=True)
class TestRecord:
    """Identity and span information for a test."""

    __test__ = False

    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: list[str]
    flaky: bool | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class ImportanceInputs:
    """Inputs required to compute test importance."""

    functions_covered_count: int
    weighted_degree: float | None
    max_function_count: int
    max_weighted_degree: float
    subsystem_risk: float | None
    max_subsystem_risk: float


@dataclass(frozen=True)
class BehavioralLLMRequest:
    """Payload sent to an LLM classifier for behavioral coverage."""

    repo: str
    commit: str
    test_id: str
    rel_path: str
    qualname: str
    markers: list[str]
    functions_covered: list[dict[str, object]]
    subsystems_covered: list[dict[str, object]]
    assert_count: int
    raise_count: int
    status: str | None
    source: str | None


@dataclass(frozen=True)
class BehavioralLLMResult:
    """LLM classification result for behavioral coverage."""

    tags: list[str]
    model: str | None = None
    run_id: str | None = None


type BehavioralLLMRunner = Callable[[BehavioralLLMRequest], BehavioralLLMResult]


@dataclass(frozen=True)
class TestProfileContext:
    """Shared inputs for building test_profile rows."""

    __test__ = False
    cfg: TestProfileStepConfig
    now: datetime
    max_function_count: int
    max_weighted_degree: float
    max_subsystem_risk: float
    functions_covered: Mapping[str, FunctionCoverageEntryProtocol]
    subsystems_covered: Mapping[str, SubsystemCoverageEntryProtocol]
    tg_metrics: Mapping[str, TestGraphMetricsProtocol]
    ast_info: Mapping[str, TestAstInfo]


@dataclass(frozen=True)
class BehavioralContext:
    """Context for behavioral coverage tagging."""

    cfg: BehavioralCoverageStepConfig
    ast_info: Mapping[str, TestAstInfo]
    profile_ctx: Mapping[str, dict[str, object]]
    now: datetime
    llm_runner: BehavioralLLMRunner | None


__all__ = [
    "BehavioralContext",
    "BehavioralLLMRequest",
    "BehavioralLLMResult",
    "BehavioralLLMRunner",
    "ImportanceInputs",
    "IoFlags",
    "TestAstInfo",
    "TestProfileContext",
    "TestRecord",
]
