"""Shared type definitions for test analytics profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from codeintel.analytics.ast_features.model import IoFlags

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig


class FunctionCoverageEntryProtocol(Protocol):
    """Structural type for test→function coverage entries."""

    @property
    def functions(self) -> list[dict[str, object]]:
        """Return list of covered functions with metadata."""
        ...

    @property
    def count(self) -> int:
        """Return number of functions covered."""
        ...

    @property
    def primary(self) -> list[int]:
        """Return GOID h128 values for primary functions."""
        ...


class SubsystemCoverageEntryProtocol(Protocol):
    """Structural type for test→subsystem coverage entries."""

    @property
    def subsystems(self) -> list[dict[str, object]]:
        """Return list of covered subsystems with metadata."""
        ...

    @property
    def count(self) -> int:
        """Return number of subsystems covered."""
        ...

    @property
    def primary_subsystem_id(self) -> str | None:
        """Return ID of the primary subsystem, if any."""
        ...

    @property
    def max_risk_score(self) -> float | None:
        """Return maximum risk score across covered subsystems."""
        ...


class TestGraphMetricsProtocol(Protocol):
    """Structural type for graph metrics on test nodes."""

    @property
    def degree(self) -> int | None:
        """Return bipartite degree of the test node."""
        ...

    @property
    def weighted_degree(self) -> float | None:
        """Return weighted degree of the test node."""
        ...

    @property
    def proj_degree(self) -> int | None:
        """Return projected graph degree."""
        ...

    @property
    def proj_weight(self) -> float | None:
        """Return projected graph weight."""
        ...

    @property
    def proj_clustering(self) -> float | None:
        """Return clustering coefficient in projected graph."""
        ...

    @property
    def proj_betweenness(self) -> float | None:
        """Return betweenness centrality in projected graph."""
        ...


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
    io_flags: IoFlags = field(default_factory=IoFlags)


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
    "FunctionCoverageEntryProtocol",
    "ImportanceInputs",
    "IoFlags",
    "SubsystemCoverageEntryProtocol",
    "TestAstInfo",
    "TestGraphMetricsProtocol",
    "TestProfileContext",
    "TestRecord",
]
