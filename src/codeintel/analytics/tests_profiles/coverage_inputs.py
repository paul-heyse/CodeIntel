"""Coverage aggregation helpers for test profiles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from codeintel.analytics.tests_profiles.types import (
    FunctionCoverageEntryProtocol,
    SubsystemCoverageEntryProtocol,
    TestGraphMetricsProtocol,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.ingestion.paths import relpath_to_module
from codeintel.storage.gateway import DuckDBConnection

PRIMARY_COVERAGE_THRESHOLD = 0.4


@dataclass(frozen=True)
class FunctionCoverageEntry:
    """Concrete coverage entry for a test→function edge."""

    functions: list[dict[str, object]]
    count: int
    primary: list[int]


@dataclass(frozen=True)
class SubsystemCoverageEntry:
    """Concrete coverage entry for a test→subsystem edge."""

    subsystems: list[dict[str, object]]
    count: int
    primary_subsystem_id: str | None
    max_risk_score: float | None


@dataclass(frozen=True)
class TestGraphMetrics:
    """Concrete graph metrics container for a test node."""

    degree: int | None
    weighted_degree: float | None
    proj_degree: int | None
    proj_weight: float | None
    proj_clustering: float | None
    proj_betweenness: float | None


def aggregate_test_coverage_by_function(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig,
    *,
    loader: Callable[[DuckDBConnection, str, str], Mapping[str, Mapping[str, object]]]
    | None = None,
) -> Mapping[str, FunctionCoverageEntryProtocol]:
    """
    Aggregate coverage signals per test→function edge using an injected loader.

    Returns
    -------
    Mapping[str, object]
        Coverage entries keyed by ``test_id``.

    Raises
    ------
    RuntimeError
        If no loader is provided.
    """
    load_fn = loader
    if load_fn is None:
        msg = "aggregate_test_coverage_by_function requires a loader for function coverage."
        raise RuntimeError(msg)
    rows = load_fn(con, cfg.repo, cfg.commit)
    result: dict[str, FunctionCoverageEntryProtocol] = {}
    for test_id, payload in rows.items():
        module_name = cast("str | None", payload.get("module"))
        rel_path = payload.get("rel_path")
        if module_name is None and rel_path is not None:
            module_name = relpath_to_module(str(rel_path))
        raw_functions = payload.get("functions")
        functions = (
            list(cast("Iterable[dict[str, object]]", raw_functions))
            if isinstance(raw_functions, Iterable)
            else []
        )
        raw_primary = payload.get("primary")
        primary_list = _to_int_list(raw_primary)
        count_raw = payload.get("count")
        count_raw_int = _as_int(count_raw)
        count_value: int = count_raw_int if count_raw_int is not None else len(functions)
        result[str(test_id)] = FunctionCoverageEntry(
            functions=functions,
            count=count_value,
            primary=primary_list,
        )
    return result


def aggregate_test_coverage_by_subsystem(
    con: DuckDBConnection,
    cfg: BehavioralCoverageStepConfig,
    *,
    loader: Callable[[DuckDBConnection, str, str], Mapping[str, Mapping[str, object]]]
    | None = None,
) -> Mapping[str, SubsystemCoverageEntryProtocol]:
    """
    Aggregate subsystem coverage signals per test using an injected loader.

    Returns
    -------
    Mapping[str, object]
        Coverage entries keyed by ``test_id``.

    Raises
    ------
    RuntimeError
        If no loader is provided.
    """
    load_fn = loader
    if load_fn is None:
        msg = "aggregate_test_coverage_by_subsystem requires a loader for subsystem coverage."
        raise RuntimeError(msg)
    rows = load_fn(con, cfg.repo, cfg.commit)

    result: dict[str, SubsystemCoverageEntryProtocol] = {}
    for test_id, payload in rows.items():
        raw_subsystems = payload.get("subsystems")
        subsystems = (
            list(cast("Iterable[dict[str, object]]", raw_subsystems))
            if isinstance(raw_subsystems, Iterable)
            else []
        )
        count_raw = payload.get("count")
        count_raw_int = _as_int(count_raw)
        count_value: int = count_raw_int if count_raw_int is not None else len(subsystems)
        result[str(test_id)] = SubsystemCoverageEntry(
            subsystems=subsystems,
            count=count_value,
            primary_subsystem_id=cast("str | None", payload.get("primary_subsystem_id")),
            max_risk_score=_as_float(payload.get("max_risk_score")),
        )
    return result


def load_test_graph_metrics(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig,
    *,
    loader: Callable[[DuckDBConnection, str, str], Mapping[str, Mapping[str, object]]]
    | None = None,
) -> Mapping[str, TestGraphMetricsProtocol]:
    """
    Load test graph metrics used in importance calculations via an injected loader.

    Returns
    -------
    Mapping[str, object]
        Metrics keyed by ``test_id``.

    Raises
    ------
    RuntimeError
        If no loader is provided.
    """
    load_fn = loader
    if load_fn is None:
        msg = "load_test_graph_metrics requires a loader for graph metrics."
        raise RuntimeError(msg)
    rows = load_fn(con, cfg.repo, cfg.commit)

    metrics: dict[str, TestGraphMetricsProtocol] = {}
    for test_id, payload in rows.items():
        metrics[str(test_id)] = TestGraphMetrics(
            degree=_as_int(payload.get("degree")),
            weighted_degree=_as_float(payload.get("weighted_degree")),
            proj_degree=_as_int(payload.get("proj_degree")),
            proj_weight=_as_float(payload.get("proj_weight")),
            proj_clustering=_as_float(payload.get("proj_clustering")),
            proj_betweenness=_as_float(payload.get("proj_betweenness")),
        )
    return metrics


def load_test_records(
    con: DuckDBConnection,
    cfg: TestProfileStepConfig | BehavioralCoverageStepConfig,
    *,
    loader: Callable[[DuckDBConnection, str, str], Iterable[Mapping[str, Any]]] | None = None,
) -> list[TestRecord]:
    """
    Load test catalog records for the configured snapshot via an injected loader.

    Returns
    -------
    list[TestRecord]
        Test records for the snapshot.

    Raises
    ------
    RuntimeError
        If no loader is provided.
    """
    load_fn = loader
    if load_fn is None:
        msg = "load_test_records requires a loader for test records."
        raise RuntimeError(msg)
    rows = load_fn(con, cfg.repo, cfg.commit)

    records: list[TestRecord] = []
    for payload in rows:
        module_name = cast("str | None", payload.get("module"))
        rel_path = payload.get("rel_path")
        if module_name is None and rel_path is not None:
            module_name = relpath_to_module(str(rel_path))
        records.append(
            TestRecord(
                test_id=str(payload.get("test_id")),
                test_goid_h128=cast("int | None", payload.get("test_goid_h128")),
                urn=cast("str | None", payload.get("urn")),
                rel_path=str(rel_path) if rel_path is not None else "",
                module=module_name,
                qualname=cast("str | None", payload.get("qualname")),
                language=cast("str | None", payload.get("language")),
                kind=cast("str | None", payload.get("kind")),
                status=cast("str | None", payload.get("status")),
                duration_ms=cast("float | None", payload.get("duration_ms")),
                markers=_normalize_markers(
                    cast("list[str] | None", payload.get("markers")),
                ),
                flaky=cast("bool | None", payload.get("flaky")),
                start_line=cast("int | None", payload.get("start_line")),
                end_line=cast("int | None", payload.get("end_line")),
            )
        )
    return records


def _normalize_markers(markers: list[str] | None) -> list[str]:
    if markers is None:
        return []
    return [str(marker) for marker in markers]


def _as_int(value: object | None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _as_float(value: object | None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int_list(value: object | None) -> list[int]:
    if not isinstance(value, Iterable):
        return []
    result: list[int] = []
    for item in value:
        coerced = _as_int(item)
        if coerced is not None:
            result.append(coerced)
    return result
