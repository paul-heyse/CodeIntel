"""Factory helpers for commonly used analytics rows.

Available builders include module_row, function_metrics_row, coverage_line_row,
test_catalog_row, typedness_row, static_diagnostics_row, subsystem_row, config_value_row,
and ast_metric_row. Prefer these over ad hoc tuples to keep schemas consistent in tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypedDict

from codeintel.core.catalog import FunctionSpan
from tests._helpers.builders import FunctionMetricsRow, ModuleRow
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def list_public_exports(module: object) -> tuple[str, ...]:
    """Return a stable tuple of public export names for a module.

    Returns
    -------
    tuple[str, ...]
        Publicly exported attribute names.
    """
    exports_attr: Any = getattr(module, "__all__", None)
    if isinstance(exports_attr, (list, tuple)):
        return tuple(exports_attr)
    return tuple(sorted(name for name in dir(module) if not name.startswith("_")))


def _coerce_int(value: object) -> int:
    """Best-effort conversion to int for metrics fields.

    Returns
    -------
    int
        Converted integer value or 0 on failure.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _coerce_bool(value: object) -> bool:
    """Best-effort conversion to bool for metrics fields.

    Returns
    -------
    bool
        Boolean interpretation of the value.
    """
    return bool(value)


def function_meta(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    line_span: tuple[int, int] = (1, 1),
) -> FunctionSpan:
    """Build a FunctionSpan with consistent URN formatting.

    Returns
    -------
    FunctionMeta
        Catalog entry with normalized URN and line span.
    """
    repo, commit = snapshot
    start_line, end_line = line_span
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    return FunctionSpan(
        goid=goid,
        urn=urn,
        rel_path=rel_path,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
    )


def module_row(
    *,
    module: str,
    path: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
) -> ModuleRow:
    """Create a ModuleRow for analytics modules.

    Returns
    -------
    ModuleRow
        Row ready for insertion into analytics.modules.
    """
    repo, commit = snapshot
    return ModuleRow(module=module, path=path, repo=repo, commit=commit)


def function_metrics_row(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    metrics: Mapping[str, int | str | bool | datetime | float | None] | None = None,
) -> FunctionMetricsRow:
    """Create a FunctionMetricsRow with sensible defaults and override support.

    Returns
    -------
    FunctionMetricsRow
        Row with defaulted metrics for analytics.function_metrics.
    """
    repo, commit = snapshot
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    defaults: dict[str, object] = {
        "language": "python",
        "kind": "function",
        "start_line": 1,
        "end_line": 1,
        "loc": 2,
        "logical_loc": 2,
        "param_count": 0,
        "positional_params": 0,
        "keyword_only_params": 0,
        "has_varargs": False,
        "has_varkw": False,
        "is_async": False,
        "is_generator": False,
        "return_count": 0,
        "yield_count": 0,
        "raise_count": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 1,
        "stmt_count": 2,
        "decorator_count": 0,
        "has_docstring": False,
        "complexity_bucket": "low",
        "created_at": datetime.now(tz=UTC),
    }
    if metrics:
        defaults.update(metrics)

    def _as_int(key: str) -> int:
        return _coerce_int(defaults.get(key, 0))

    def _as_bool(key: str) -> bool:
        value = defaults.get(key)
        if value is None:
            return _coerce_bool(defaults.get(key, False))
        return _coerce_bool(value)

    def _as_str(key: str) -> str:
        value = defaults.get(key)
        if value is None:
            return str(defaults[key])
        return value if isinstance(value, str) else str(value)

    created_at_value = defaults.get("created_at")
    created_at = (
        created_at_value if isinstance(created_at_value, datetime) else datetime.now(tz=UTC)
    )
    return FunctionMetricsRow(
        function_goid_h128=goid,
        urn=urn,
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        language=_as_str("language"),
        kind=_as_str("kind"),
        qualname=qualname,
        start_line=_as_int("start_line"),
        end_line=_as_int("end_line"),
        loc=_as_int("loc"),
        logical_loc=_as_int("logical_loc"),
        param_count=_as_int("param_count"),
        positional_params=_as_int("positional_params"),
        keyword_only_params=_as_int("keyword_only_params"),
        has_varargs=_as_bool("has_varargs"),
        has_varkw=_as_bool("has_varkw"),
        is_async=_as_bool("is_async"),
        is_generator=_as_bool("is_generator"),
        return_count=_as_int("return_count"),
        yield_count=_as_int("yield_count"),
        raise_count=_as_int("raise_count"),
        cyclomatic_complexity=_as_int("cyclomatic_complexity"),
        max_nesting_depth=_as_int("max_nesting_depth"),
        stmt_count=_as_int("stmt_count"),
        decorator_count=_as_int("decorator_count"),
        has_docstring=_as_bool("has_docstring"),
        complexity_bucket=_as_str("complexity_bucket"),
        created_at=created_at,
    )


@dataclass
class DependencyCallSeed:
    library: str
    service_name: str
    qualname: str
    callsite_count: int
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def dependency_call_row(seed: DependencyCallSeed) -> tuple[object, ...]:
    """Row for analytics.external_dependency_calls.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.library,
        seed.service_name,
        seed.qualname,
        seed.callsite_count,
    )


def compute_dep_id(repo: str, commit: str, library: str) -> str:
    """Compute unique dependency identifier used in analytics dependency tables.

    Returns
    -------
    str
        SHA-1 hash prefix as dependency ID.
    """
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class DependencyCallRow:
    """Row model for analytics.external_dependency_calls."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    function_goid_h128: Decimal
    function_urn: str
    rel_path: str
    module: str
    qualname: str
    callsite_count: int
    modes: list[str]
    evidence_json: list[dict[str, object]]
    created_at: datetime


@dataclass(frozen=True)
class DependencyAggregateRow:
    """Row model for analytics.external_dependencies."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


@dataclass
class DependencyCallPayloadSeed:
    library: str
    service_name: str
    qualname: str
    rel_path: str
    module: str
    callsite_count: int
    function_goid: int
    modes: Sequence[str] = ()
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    evidence_json: list[dict[str, object]] | None = None


class DependencyCallPayload(TypedDict):
    """Typed payload for analytics.external_dependency_calls."""

    repo: str
    commit: str
    library: str
    service_name: str
    qualname: str
    rel_path: str
    module: str
    callsite_count: int
    modes: list[str]
    function_goid_h128: Decimal
    function_urn: str
    evidence_json: list[dict[str, object]]
    created_at: datetime


def dependency_call_payload(seed: DependencyCallPayloadSeed) -> DependencyCallPayload:
    """Payload dict for analytics.external_dependency_calls.

    Returns
    -------
    DependencyCallPayload
        Dict matching adapter fields for external_dependency_calls.
    """
    created_at = datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "library": seed.library,
        "service_name": seed.service_name,
        "qualname": seed.qualname,
        "rel_path": seed.rel_path,
        "module": seed.module,
        "callsite_count": seed.callsite_count,
        "modes": list(seed.modes),
        "function_goid_h128": Decimal(seed.function_goid),
        "function_urn": f"urn:{seed.qualname}",
        "evidence_json": seed.evidence_json or [{"type": "call", "line": 1}],
        "created_at": created_at,
    }


@dataclass
class DependencyAggregateSeed:
    library: str
    service_name: str
    category: str | None
    severity: str | None
    criticality: str | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def dependency_aggregate_row(seed: DependencyAggregateSeed) -> tuple[object, ...]:
    """Row for analytics.external_dependencies.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.library,
        seed.service_name,
        seed.category,
        seed.severity,
        seed.criticality,
        seed.risk_score,
        seed.function_count,
        seed.callsite_count,
    )


@dataclass
class DependencyAggregatePayloadSeed:
    library: str
    service_name: str
    category: str | None
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    risk_level: str
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    modules_json: list[str] | None = None
    usage_modes: list[str] | None = None
    config_keys: list[str] | None = None


class DependencyAggregatePayload(TypedDict):
    """Typed payload for analytics.external_dependencies."""

    repo: str
    commit: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


def dependency_aggregate_payload(
    seed: DependencyAggregatePayloadSeed,
) -> DependencyAggregatePayload:
    """Payload dict for analytics.external_dependencies.

    Returns
    -------
    DependencyAggregatePayload
        Dict matching adapter fields for external_dependencies.
    """
    created_at = datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "library": seed.library,
        "service_name": seed.service_name,
        "category": seed.category,
        "language": "python",
        "severity": seed.severity,
        "criticality": seed.criticality,
        "risk_score": seed.risk_score,
        "function_count": seed.function_count,
        "callsite_count": seed.callsite_count,
        "modules_json": seed.modules_json or ["module.default"],
        "usage_modes": seed.usage_modes or ["read"],
        "config_keys": seed.config_keys or [],
        "risk_level": seed.risk_level,
        "created_at": created_at,
    }


def dependency_call_row_from_payload(seed: DependencyCallPayloadSeed) -> DependencyCallRow:
    """Build DependencyCallRow from the canonical payload seed.

    Returns
    -------
    DependencyCallRow
        Row ready for persistence.
    """
    payload = dependency_call_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyCallRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        function_goid_h128=payload["function_goid_h128"],
        function_urn=str(payload["function_urn"]),
        rel_path=str(payload["rel_path"]),
        module=str(payload["module"]),
        qualname=str(payload["qualname"]),
        callsite_count=payload["callsite_count"],
        modes=payload["modes"],
        evidence_json=payload["evidence_json"],
        created_at=payload["created_at"],
    )


def dependency_aggregate_row_from_payload(
    seed: DependencyAggregatePayloadSeed,
) -> DependencyAggregateRow:
    """Build DependencyAggregateRow from the canonical payload seed.

    Returns
    -------
    DependencyAggregateRow
        Row ready for persistence.
    """
    payload = dependency_aggregate_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyAggregateRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        category=payload["category"],
        language=str(payload["language"]),
        severity=payload["severity"],
        criticality=payload["criticality"],
        risk_score=payload["risk_score"],
        function_count=payload["function_count"],
        callsite_count=payload["callsite_count"],
        modules_json=payload["modules_json"],
        usage_modes=payload["usage_modes"],
        config_keys=payload["config_keys"],
        risk_level=str(payload["risk_level"]),
        created_at=payload["created_at"],
    )


@dataclass
class EntrypointSeed:
    entrypoint_id: str
    handler_qualname: str
    kind: str = "api_function"
    command_name: str | None = None
    http_method: str | None = "GET"
    path: str | None = "/"
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def entrypoint_row(seed: EntrypointSeed) -> tuple[object, ...]:
    """Row for analytics.entrypoints.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.entrypoint_id,
        seed.handler_qualname,
        seed.kind,
        seed.command_name,
        seed.http_method,
        seed.path,
    )


@dataclass
class EntrypointPayloadSeed:
    entrypoint_id: str
    handler_qualname: str
    kind: str = "http_endpoint"
    framework: str = "fastapi"
    handler_goid_h128: Decimal | int = Decimal(0)
    handler_rel_path: str = "src/api.py"
    handler_module: str = "api"
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    http_method: str | None = "GET"
    route_path: str | None = "/"
    status_codes: list[int] | None = None
    auth_required: bool = True
    command_name: str | None = None
    arguments_schema: dict[str, object] | None = None
    schedule: str | None = None
    trigger: str | None = None
    extra: dict[str, object] | None = None
    subsystem_id: str | None = None
    subsystem_name: str | None = None
    tags: list[str] | None = None
    owners: list[str] | None = None
    tests_touching: int = 0
    failing_tests: int = 0
    slow_tests: int = 0
    flaky_tests: int = 0
    entrypoint_coverage_ratio: float = 0.0
    last_test_status: str = "unknown"
    created_at: datetime | None = None


def entrypoint_payload(seed: EntrypointPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.entrypoints adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for entrypoints.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "entrypoint_id": seed.entrypoint_id,
        "kind": seed.kind,
        "framework": seed.framework,
        "handler_goid_h128": Decimal(seed.handler_goid_h128),
        "handler_urn": f"urn:{seed.handler_module}:{seed.handler_qualname}",
        "handler_rel_path": seed.handler_rel_path,
        "handler_module": seed.handler_module,
        "handler_qualname": seed.handler_qualname,
        "http_method": seed.http_method,
        "route_path": seed.route_path,
        "status_codes": seed.status_codes or [200],
        "auth_required": seed.auth_required,
        "command_name": seed.command_name,
        "arguments_schema": seed.arguments_schema,
        "schedule": seed.schedule,
        "trigger": seed.trigger,
        "extra": seed.extra or {},
        "subsystem_id": seed.subsystem_id,
        "subsystem_name": seed.subsystem_name,
        "tags": seed.tags or [],
        "owners": seed.owners or [],
        "tests_touching": seed.tests_touching,
        "failing_tests": seed.failing_tests,
        "slow_tests": seed.slow_tests,
        "flaky_tests": seed.flaky_tests,
        "entrypoint_coverage_ratio": seed.entrypoint_coverage_ratio,
        "last_test_status": seed.last_test_status,
        "created_at": created_at,
    }


@dataclass
class EntrypointTestPayloadSeed:
    entrypoint_id: str
    test_id: str
    test_goid_h128: Decimal | int
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    coverage_ratio: float = 0.0
    status: str = "passed"
    duration_ms: float = 0.0
    created_at: datetime | None = None


def entrypoint_test_payload(seed: EntrypointTestPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.entrypoint_tests adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for entrypoint_tests.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "entrypoint_id": seed.entrypoint_id,
        "test_id": seed.test_id,
        "test_goid_h128": Decimal(seed.test_goid_h128),
        "coverage_ratio": seed.coverage_ratio,
        "status": seed.status,
        "duration_ms": seed.duration_ms,
        "created_at": created_at,
    }


@dataclass
class EntrypointTestSeed:
    entrypoint_id: str
    test_qualname: str
    status: str
    coverage_ratio: float
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def entrypoint_test_row(seed: EntrypointTestSeed) -> tuple[object, ...]:
    """Row for analytics.entrypoint_tests.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.entrypoint_id,
        seed.test_qualname,
        seed.status,
        seed.coverage_ratio,
    )


@dataclass
class SemanticRoleFunctionSeed:
    goid: int
    role: str
    confidence: float
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def semantic_role_function_row(seed: SemanticRoleFunctionSeed) -> tuple[object, ...]:
    """Row for analytics.semantic_roles_functions.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.goid, seed.role, seed.confidence)


@dataclass
class SemanticRoleModuleSeed:
    module: str
    role: str
    confidence: float
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def semantic_role_module_row(seed: SemanticRoleModuleSeed) -> tuple[object, ...]:
    """Row for analytics.semantic_roles_modules.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.module, seed.role, seed.confidence)


@dataclass
class SubsystemPayloadSeed:
    subsystem_id: str
    name: str
    description: str
    module_count: int
    modules_json: Sequence[str]
    entrypoints_json: Sequence[str]
    internal_edge_count: int
    external_edge_count: int
    fan_in: int
    fan_out: int
    function_count: int
    risk_level: str
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: int = 0
    created_at: datetime | None = None


def subsystem_payload(seed: SubsystemPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.subsystems adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for subsystems.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "subsystem_id": seed.subsystem_id,
        "name": seed.name,
        "description": seed.description,
        "module_count": seed.module_count,
        "modules_json": list(seed.modules_json),
        "entrypoints_json": list(seed.entrypoints_json),
        "internal_edge_count": seed.internal_edge_count,
        "external_edge_count": seed.external_edge_count,
        "fan_in": seed.fan_in,
        "fan_out": seed.fan_out,
        "function_count": seed.function_count,
        "avg_risk_score": seed.avg_risk_score,
        "max_risk_score": seed.max_risk_score,
        "high_risk_function_count": seed.high_risk_function_count,
        "risk_level": seed.risk_level,
        "created_at": created_at,
    }


@dataclass
class SubsystemModulePayloadSeed:
    subsystem_id: str
    module: str
    role: str | None = None
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def subsystem_module_payload(seed: SubsystemModulePayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.subsystem_modules adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for subsystem_modules.
    """
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "subsystem_id": seed.subsystem_id,
        "module": seed.module,
        "role": seed.role,
    }


@dataclass
class SubsystemSeed:
    subsystem_id: str
    name: str
    risk_level: str
    function_count: int
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    description: str | None = None
    module_count: int = 0
    entrypoints_json: str | None = None
    modules_json: str = "[]"
    internal_edge_count: int = 0
    external_edge_count: int = 0
    fan_in: int = 0
    fan_out: int = 0
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: int = 0
    created_at: str | None = None


def subsystem_row(
    seed: SubsystemSeed,
) -> tuple[
    str,
    str,
    str,
    str,
    str | None,
    int,
    str,
    str | None,
    int,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    int,
    str | None,
    str,
]:
    """Row for analytics.subsystems.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.subsystem_id,
        seed.name,
        seed.description,
        seed.module_count,
        seed.modules_json,
        seed.entrypoints_json,
        seed.internal_edge_count,
        seed.external_edge_count,
        seed.fan_in,
        seed.fan_out,
        seed.function_count,
        seed.avg_risk_score,
        seed.max_risk_score,
        seed.high_risk_function_count,
        seed.risk_level,
        seed.created_at or datetime.now(tz=UTC).isoformat(),
    )


@dataclass
class SubsystemModuleSeed:
    subsystem_id: str
    module: str
    role: str | None = None
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def subsystem_module_row(seed: SubsystemModuleSeed) -> tuple[str, str, str, str, str | None]:
    """Row for analytics.subsystem_modules.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.subsystem_id, seed.module, seed.role)


@dataclass
class DataModelUsagePayloadSeed:
    model_id: str
    goid: Decimal | int
    usage_kinds: Sequence[str]
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    evidence_json: list[dict[str, object]] | None = None
    context_json: dict[str, object] | None = None
    created_at: datetime | None = None


def data_model_usage_payload(seed: DataModelUsagePayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.data_model_usage adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for data_model_usage.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "model_id": seed.model_id,
        "function_goid_h128": Decimal(seed.goid),
        "usage_kinds_json": list(seed.usage_kinds),
        "evidence_json": seed.evidence_json
        or [{"type": "attribute_access", "attr": "field", "line": 1}],
        "context_json": seed.context_json
        or {"file_path": "src/services/service.py", "function_name": "fn"},
        "created_at": created_at,
    }


@dataclass
class DataModelUsageSeed:
    model_id: str
    goid: int
    usage_kinds: Sequence[str]
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT


def data_model_usage_row(seed: DataModelUsageSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_usage.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.model_id, seed.goid, list(seed.usage_kinds))


@dataclass
class DataModelSeed:
    model_id: str
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    goid: int | None = None
    base_classes_json: list[dict[str, str]] | list[str] | None = None
    doc_short: str | None = None
    doc_long: str | None = None
    created_at: datetime | None = None


def data_model_row(seed: DataModelSeed) -> tuple[object, ...]:
    """Row for analytics.data_models.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_models.
    """
    return (
        seed.repo,
        seed.commit,
        seed.model_id,
        seed.goid,
        seed.model_name,
        seed.module,
        seed.rel_path,
        seed.model_kind,
        seed.base_classes_json or [],
        seed.doc_short,
        seed.doc_long,
        (seed.created_at or datetime.now(tz=UTC)).isoformat(),
    )


@dataclass
class DataModelFieldSeed:
    model_id: str
    field_name: str
    field_type: str | None
    required: bool
    has_default: bool
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    default_expr: str | None = None
    constraints_json: dict[str, object] | None = None
    source: str = "tests"
    rel_path: str = "models.py"
    lineno: int | None = 1
    created_at: datetime | None = None


def data_model_field_row(seed: DataModelFieldSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_fields.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_model_fields.
    """
    return (
        seed.repo,
        seed.commit,
        seed.model_id,
        seed.field_name,
        seed.field_type,
        seed.required,
        seed.has_default,
        seed.default_expr,
        seed.constraints_json or {},
        seed.source,
        seed.rel_path,
        seed.lineno,
        (seed.created_at or datetime.now(tz=UTC)).isoformat(),
    )


@dataclass
class DataModelRelationshipSeed:
    source_model_id: str
    target_model_id: str
    field_name: str
    relationship_kind: str
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    target_module: str | None = None
    target_model_name: str | None = None
    multiplicity: str | None = None
    via: str | None = None
    evidence_json: dict[str, object] | list[object] | None = None
    rel_path: str = "models.py"
    lineno: int | None = 1
    created_at: datetime | None = None


def data_model_relationship_row(seed: DataModelRelationshipSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_relationships.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_model_relationships.
    """
    return (
        seed.repo,
        seed.commit,
        seed.source_model_id,
        seed.target_model_id,
        seed.target_module,
        seed.target_model_name,
        seed.field_name,
        seed.relationship_kind,
        seed.multiplicity,
        seed.via,
        seed.evidence_json or {},
        seed.rel_path,
        seed.lineno,
        (seed.created_at or datetime.now(tz=UTC)).isoformat(),
    )


@dataclass
class AstMetricSeed:
    rel_path: str
    complexity: float
    node_count: int = 100
    function_count: int = 10
    class_count: int = 2
    avg_depth: float = 2.5
    max_depth: int = 5
    created_at: datetime | None = None


def ast_metric_row(seed: AstMetricSeed) -> tuple[str, int, int, int, float, int, float, str]:
    """Row for analytics.ast_metrics.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.rel_path,
        seed.node_count,
        seed.function_count,
        seed.class_count,
        seed.avg_depth,
        seed.max_depth,
        seed.complexity,
        (seed.created_at or datetime.now(tz=UTC)).isoformat(),
    )


@dataclass
class CoverageLineSeed:
    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: str


def coverage_line_row(
    seed: CoverageLineSeed,
) -> tuple[str, str, str, int, bool, bool, int, int, str]:
    """Row for analytics.coverage_lines.

    Returns
    -------
    tuple[str, str, str, int, bool, bool, int, int, str]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.rel_path,
        seed.line,
        seed.is_executable,
        seed.is_covered,
        seed.hits,
        seed.context_count,
        seed.created_at,
    )


@dataclass
class TestCatalogSeed:
    test_id: str
    test_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    qualname: str
    kind: str
    status: str
    duration_ms: int
    markers: str
    parametrized: bool
    flaky: bool
    created_at: str


def test_catalog_row(
    seed: TestCatalogSeed,
) -> tuple[
    str,
    int,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    str,
    bool,
    bool,
    str,
]:
    """Row for analytics.test_catalog.

    Returns
    -------
    tuple[str, int, str, str, str, str, str, str, str, int, str, bool, bool, str]
        Row values in schema order.
    """
    return (
        seed.test_id,
        seed.test_goid_h128,
        seed.urn,
        seed.repo,
        seed.commit,
        seed.rel_path,
        seed.qualname,
        seed.kind,
        seed.status,
        seed.duration_ms,
        seed.markers,
        seed.parametrized,
        seed.flaky,
        seed.created_at,
    )


@dataclass
class TypednessSeed:
    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio_json: str
    untyped_defs: int
    overlay_needed: bool


def typedness_row(seed: TypednessSeed) -> tuple[str, str, str, int, str, int, bool]:
    """Row for analytics.typedness.

    Returns
    -------
    tuple[str, str, str, int, str, int, bool]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.path,
        seed.type_error_count,
        seed.annotation_ratio_json,
        seed.untyped_defs,
        seed.overlay_needed,
    )


@dataclass
class StaticDiagnosticsSeed:
    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool


def static_diagnostics_row(
    seed: StaticDiagnosticsSeed,
) -> tuple[str, str, str, int, int, int, int, bool]:
    """Row for analytics.static_diagnostics.

    Returns
    -------
    tuple[str, str, str, int, int, int, int, bool]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.rel_path,
        seed.pyrefly_errors,
        seed.pyright_errors,
        seed.ruff_errors,
        seed.total_errors,
        seed.has_errors,
    )


# =============================================================================
# Profile adapter row helpers
# =============================================================================


def function_profile_row(
    *,
    goid: Decimal = Decimal(12345),
    qualname: str = "module.function_name",
    rel_path: str = "src/module.py",
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    **overrides: object,
) -> dict[str, Any]:
    """Build a function_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.function_profile schema.
    """
    base: dict[str, Any] = {
        "function_goid_h128": goid,
        "urn": f"urn:{repo}::{qualname}",
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "kind": "function",
        "qualname": qualname,
        "start_line": 10,
        "end_line": 30,
        "loc": 50,
        "logical_loc": 40,
        "cyclomatic_complexity": 5,
        "complexity_bucket": "low",
        "param_count": 3,
        "positional_params": 2,
        "keyword_params": 1,
        "vararg": False,
        "kwarg": False,
        "max_nesting_depth": 2,
        "stmt_count": 15,
        "decorator_count": 1,
        "has_docstring": True,
        "total_params": 3,
        "annotated_params": 3,
        "return_type": "str",
        "param_types": ["int", "str", "bool"],
        "fully_typed": True,
        "partial_typed": False,
        "untyped": False,
        "typedness_bucket": "fully_typed",
        "typedness_source": "annotations",
        "file_typed_ratio": 0.95,
        "static_error_count": 0,
        "has_static_errors": False,
        "executable_lines": 40,
        "covered_lines": 35,
        "coverage_ratio": 0.875,
        "tested": True,
        "untested_reason": None,
        "tests_touching": 5,
        "failing_tests": 0,
        "slow_tests": 0,
        "flaky_tests": 0,
        "last_test_status": "passed",
        "dominant_test_status": "passed",
        "slow_test_threshold_ms": 1000.0,
        "created_in_commit": commit,
        "created_at_history": datetime.now(tz=UTC),
        "last_modified_commit": commit,
        "last_modified_at": datetime.now(tz=UTC),
        "age_days": 30,
        "commit_count": 10,
        "author_count": 3,
        "lines_added": 100,
        "lines_deleted": 20,
        "churn_score": 0.5,
        "stability_bucket": "stable",
        "call_fan_in": 5,
        "call_fan_out": 3,
        "call_edge_in_count": 5,
        "call_edge_out_count": 3,
        "call_is_leaf": False,
        "call_is_entrypoint": False,
        "call_is_public": True,
        "risk_score": 0.25,
        "risk_level": "low",
        "risk_component_coverage": 0.1,
        "risk_component_complexity": 0.05,
        "risk_component_static": 0.0,
        "risk_component_hotspot": 0.1,
        "is_pure": True,
        "uses_io": False,
        "touches_db": False,
        "uses_time": False,
        "uses_randomness": False,
        "modifies_globals": False,
        "modifies_closure": False,
        "spawns_threads_or_tasks": False,
        "has_transitive_effects": False,
        "purity_confidence": 0.95,
        "param_nullability_json": [],
        "return_nullability": "non_null",
        "has_preconditions": False,
        "has_postconditions": False,
        "has_raises": False,
        "contract_confidence": 0.9,
        "role": "helper",
        "framework": None,
        "role_confidence": 0.85,
        "role_sources_json": ["path_hint"],
        "tags": [],
        "owners": [],
        "doc_short": "Test function.",
        "doc_long": "A test function for unit tests.",
        "doc_params": {"param1": "int", "param2": "str"},
        "doc_returns": {"type": "str", "description": "A string result"},
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


def file_profile_row(
    *,
    rel_path: str = "src/services/api.py",
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    **overrides: object,
) -> dict[str, Any]:
    """Build a file_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.file_profile schema.
    """
    base: dict[str, Any] = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "node_count": 100,
        "function_count": 10,
        "class_count": 2,
        "avg_depth": 2.5,
        "max_depth": 5,
        "ast_complexity": 15.0,
        "hotspot_score": 0.75,
        "commit_count": 50,
        "author_count": 5,
        "lines_added": 500,
        "lines_deleted": 200,
        "annotation_ratio": 0.85,
        "untyped_defs": 2,
        "overlay_needed": False,
        "type_error_count": 0,
        "static_error_count": 0,
        "has_static_errors": False,
        "total_functions": 10,
        "public_functions": 8,
        "avg_loc": 25.0,
        "max_loc": 100,
        "avg_cyclomatic_complexity": 3.5,
        "max_cyclomatic_complexity": 8,
        "high_risk_function_count": 1,
        "medium_risk_function_count": 3,
        "max_risk_score": 0.65,
        "file_coverage_ratio": 0.85,
        "tested_function_count": 8,
        "untested_function_count": 2,
        "tests_touching": 15,
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


def module_profile_row(
    *,
    module: str = "services.api",
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    **overrides: object,
) -> dict[str, Any]:
    """Build a module_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.module_profile schema.
    """
    base: dict[str, Any] = {
        "repo": repo,
        "commit": commit,
        "module": module,
        "path": module.replace(".", "/"),
        "language": "python",
        "file_count": 5,
        "total_loc": 500,
        "total_logical_loc": 400,
        "function_count": 25,
        "class_count": 5,
        "avg_file_complexity": 10.0,
        "max_file_complexity": 25.0,
        "high_risk_function_count": 2,
        "medium_risk_function_count": 5,
        "low_risk_function_count": 18,
        "max_risk_score": 0.85,
        "avg_risk_score": 0.35,
        "module_coverage_ratio": 0.75,
        "tested_function_count": 20,
        "untested_function_count": 5,
        "import_fan_in": 10,
        "import_fan_out": 15,
        "cycle_group": None,
        "in_cycle": False,
        "role": "service",
        "role_confidence": 0.9,
        "role_sources_json": ["path_hint", "decorator"],
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


@dataclass
class ConfigValueSeed:
    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    value: str | None
    section: str | None
    seq_no: int


def config_value_row(
    seed: ConfigValueSeed,
) -> tuple[str, str, str, str, str, str | None, str | None, int]:
    """Row for analytics.config_values.

    Returns
    -------
    tuple[str, str, str, str, str, str | None, str | None, int]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.config_path,
        seed.format,
        seed.key,
        seed.value,
        seed.section,
        seed.seq_no,
    )


__all__ = [
    "AstMetricSeed",
    "ConfigValueSeed",
    "CoverageLineSeed",
    "DataModelFieldSeed",
    "DataModelRelationshipSeed",
    "DataModelSeed",
    "DataModelUsagePayloadSeed",
    "DataModelUsageSeed",
    "DependencyAggregatePayloadSeed",
    "DependencyAggregateSeed",
    "DependencyCallPayloadSeed",
    "DependencyCallSeed",
    "EntrypointPayloadSeed",
    "EntrypointSeed",
    "EntrypointTestPayloadSeed",
    "EntrypointTestSeed",
    "SemanticRoleFunctionSeed",
    "SemanticRoleModuleSeed",
    "StaticDiagnosticsSeed",
    "SubsystemModulePayloadSeed",
    "SubsystemModuleSeed",
    "SubsystemPayloadSeed",
    "SubsystemSeed",
    "TestCatalogSeed",
    "TypednessSeed",
    "ast_metric_row",
    "config_value_row",
    "coverage_line_row",
    "data_model_field_row",
    "data_model_relationship_row",
    "data_model_row",
    "data_model_usage_payload",
    "data_model_usage_row",
    "dependency_aggregate_payload",
    "dependency_aggregate_row",
    "dependency_aggregate_row_from_payload",
    "dependency_call_payload",
    "dependency_call_row",
    "dependency_call_row_from_payload",
    "entrypoint_payload",
    "entrypoint_row",
    "entrypoint_test_payload",
    "entrypoint_test_row",
    "file_profile_row",
    "function_meta",
    "function_metrics_row",
    "function_profile_row",
    "module_profile_row",
    "module_row",
    "semantic_role_function_row",
    "semantic_role_module_row",
    "static_diagnostics_row",
    "subsystem_module_payload",
    "subsystem_module_row",
    "subsystem_payload",
    "subsystem_row",
    "test_catalog_row",
    "typedness_row",
]
