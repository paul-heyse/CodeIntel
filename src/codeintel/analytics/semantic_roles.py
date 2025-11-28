"""Heuristic semantic role classification for functions and modules."""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from codeintel.analytics.ast_utils import safe_unparse
from codeintel.analytics.context import (
    AnalyticsContext,
    AnalyticsContextConfig,
    ensure_analytics_context,
)
from codeintel.analytics.function_ast_cache import FunctionAst
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import normalize_decimal_id
from codeintel.config import SemanticRolesStepConfig
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)

ROLE_THRESHOLD = 0.35
SERVICE_FAN_IN_THRESHOLD = 5
SERVICE_FAN_OUT_THRESHOLD = 5
HELPER_LOC_THRESHOLD = 20


@dataclass(frozen=True)
class FunctionContext:
    """Classification context for a single function."""

    rel_path: str
    qualname: str
    decorators: list[str]
    effects: dict[str, object]
    contracts: dict[str, object]
    module_tags: list[str]
    module_name: str | None
    graph: dict[str, int]
    loc: int | None

    @property
    def name(self) -> str:
        """
        Return the unqualified function name.

        Returns
        -------
        str
            Function name without module qualifiers.
        """
        return self.qualname.rsplit(".", maxsplit=1)[-1]

    @property
    def rel_path_lower(self) -> str:
        """
        Lower-case relative path for path-based heuristics.

        Returns
        -------
        str
            Path normalized to lower-case.
        """
        return self.rel_path.lower()

    @property
    def module_lower(self) -> str:
        """
        Lower-case module name for module-level hints.

        Returns
        -------
        str
            Module name in lower-case.
        """
        return (self.module_name or "").lower()

    @property
    def tag_strings(self) -> list[str]:
        """
        Normalized module tags.

        Returns
        -------
        list[str]
            Tag strings normalized to lower-case.
        """
        return [str(tag).lower() for tag in self.module_tags if tag is not None]


@dataclass
class RoleAccumulator:
    """Aggregates scoring signals for semantic roles."""

    scores: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    sources: defaultdict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    frameworks: dict[str, str | None] = field(default_factory=dict)

    def bump(
        self, role: str, amount: float, reason: str, framework_hint: str | None = None
    ) -> None:
        """
        Increase a role score and track its provenance.

        Parameters
        ----------
        role
            Role identifier to bump.
        amount
            Score increment.
        reason
            Human-readable reason for the bump.
        framework_hint
            Optional framework name tied to the role.
        """
        self.scores[role] += amount
        self.sources[role].append(reason)
        if framework_hint is not None:
            self.frameworks.setdefault(role, framework_hint)

    def finalize(self) -> tuple[str, float, str | None, dict[str, object]]:
        """
        Return the winning role with confidence and metadata.

        Returns
        -------
        tuple[str, float, str | None, dict[str, object]]
            Role name, confidence, framework, and source signals.
        """
        if not self.scores:
            return "other", 0.0, None, {}
        role, raw_score = max(self.scores.items(), key=lambda item: item[1])
        if raw_score < ROLE_THRESHOLD:
            return "other", 0.0, None, {}
        confidence = min(1.0, raw_score)
        framework = self.frameworks.get(role)
        signals: list[str] = list(self.sources.get(role, []))
        source_payload: dict[str, object] = {"signals": signals}
        return role, confidence, framework, source_payload


@dataclass(frozen=True)
class RoleArtifacts:
    """Pre-loaded metadata used during role classification."""

    module_by_path: dict[str, str]
    module_meta: dict[str, ModuleRecord]
    ast_map: dict[int, FunctionAst]
    effects: dict[int, dict[str, object]]
    contracts: dict[int, dict[str, object]]
    graph_metrics: dict[int, dict[str, int]]


@dataclass(frozen=True)
class ModuleRecord:
    """Metadata for a module path and tags."""

    path: str
    tags: list[str]


def compute_semantic_roles(
    gateway: StorageGateway,
    cfg: SemanticRolesStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    context: AnalyticsContext | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """
    Populate semantic role tables for functions and modules.

    Parameters
    ----------
    gateway:
        Storage gateway providing DuckDB access.
    cfg:
        Semantic role configuration.
    catalog_provider:
        Optional pre-loaded function catalog to reuse across steps.
    context:
        Optional shared analytics context to reuse catalog, module map, and ASTs.
    runtime:
        Optional shared graph runtime used to reuse an existing engine when available.
    """
    con = gateway.con
    ensure_schema(con, "analytics.semantic_roles_functions")
    ensure_schema(con, "analytics.semantic_roles_modules")

    shared_context = ensure_analytics_context(
        gateway,
        cfg=AnalyticsContextConfig(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
            catalog_provider=catalog_provider,
        ),
        context=context,
    )
    resolve_graph_runtime(
        gateway,
        cfg.snapshot,
        runtime or GraphRuntimeOptions(),
        context=shared_context,
    )
    module_by_path = shared_context.module_map
    ast_map = shared_context.function_ast_map

    module_meta = _load_module_meta(con, repo=cfg.repo, commit=cfg.commit)
    function_rows = _load_function_rows(con, repo=cfg.repo, commit=cfg.commit)
    effects = _load_effects(con, repo=cfg.repo, commit=cfg.commit)
    contracts = _load_contracts(con, repo=cfg.repo, commit=cfg.commit)
    graph_metrics = _load_graph_metrics(con, repo=cfg.repo, commit=cfg.commit)

    artifacts = RoleArtifacts(
        module_by_path=module_by_path,
        module_meta=module_meta,
        ast_map=ast_map,
        effects=effects,
        contracts=contracts,
        graph_metrics=graph_metrics,
    )

    now = datetime.now(tz=UTC)
    fn_rows, roles_by_module = _build_function_role_rows(
        function_rows=function_rows,
        artifacts=artifacts,
        repo=cfg.repo,
        commit=cfg.commit,
        now=now,
    )

    run_batch(
        gateway,
        "analytics.semantic_roles_functions",
        fn_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    module_rows = _classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo=cfg.repo,
        commit=cfg.commit,
        now=now,
    )
    run_batch(
        gateway,
        "analytics.semantic_roles_modules",
        module_rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "semantic_roles populated: %d functions, %d modules for %s@%s",
        len(fn_rows),
        len(module_rows),
        cfg.repo,
        cfg.commit,
    )


def _build_function_role_rows(
    *,
    function_rows: list[tuple[int, str, str, int | None]],
    artifacts: RoleArtifacts,
    repo: str,
    commit: str,
    now: datetime,
) -> tuple[list[tuple[object, ...]], dict[str, list[tuple[str, float]]]]:
    fn_rows: list[tuple[object, ...]] = []
    roles_by_module: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for goid, rel_path, qualname, loc in function_rows:
        normalized_path = normalize_rel_path(rel_path)
        module = artifacts.module_by_path.get(normalized_path)
        module_record = artifacts.module_meta.get(module or "")
        module_tags: list[str] = module_record.tags if module_record else []

        ast_info = artifacts.ast_map.get(goid)
        decorators = _decorator_names(ast_info.node.decorator_list) if ast_info else []

        context = FunctionContext(
            rel_path=normalized_path,
            qualname=qualname,
            decorators=decorators,
            effects=artifacts.effects.get(goid, {}),
            contracts=artifacts.contracts.get(goid, {}),
            module_tags=module_tags,
            module_name=module,
            graph=artifacts.graph_metrics.get(goid, {}),
            loc=loc,
        )

        role, confidence, framework, role_sources = _classify_function(context)

        fn_rows.append(
            (
                repo,
                commit,
                goid,
                role,
                framework,
                confidence,
                role_sources,
                now,
            )
        )
        if module:
            roles_by_module[module].append((role, confidence))

    return fn_rows, roles_by_module


def _load_function_rows(
    con: DuckDBConnection, *, repo: str, commit: str
) -> list[tuple[int, str, str, int | None]]:
    rows: Iterable[tuple[object, str, str, int | None]] = con.execute(
        """
        SELECT function_goid_h128, rel_path, qualname, loc
        FROM analytics.function_metrics
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    result: list[tuple[int, str, str, int | None]] = []
    for goid_raw, rel_path, qualname, loc in rows:
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        result.append((goid, str(rel_path), str(qualname), loc))
    return result


def _load_effects(con: DuckDBConnection, *, repo: str, commit: str) -> dict[int, dict[str, object]]:
    rows: Iterable[tuple[object, bool, bool, bool, bool, bool, bool, bool]] = con.execute(
        """
        SELECT
            function_goid_h128,
            touches_db,
            uses_io,
            uses_time,
            uses_randomness,
            modifies_globals,
            modifies_closure,
            spawns_threads_or_tasks
        FROM analytics.function_effects
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    mapping: dict[int, dict[str, object]] = {}
    for (
        goid_raw,
        touches_db,
        uses_io,
        uses_time,
        uses_randomness,
        modifies_globals,
        modifies_closure,
        spawns_threads_or_tasks,
    ) in rows:
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "touches_db": bool(touches_db),
            "uses_io": bool(uses_io),
            "uses_time": bool(uses_time),
            "uses_randomness": bool(uses_randomness),
            "modifies_globals": bool(modifies_globals),
            "modifies_closure": bool(modifies_closure),
            "spawns_threads_or_tasks": bool(spawns_threads_or_tasks),
        }
    return mapping


def _load_contracts(
    con: DuckDBConnection, *, repo: str, commit: str
) -> dict[int, dict[str, object]]:
    rows: Iterable[tuple[object, object, object, object]] = con.execute(
        """
        SELECT function_goid_h128, preconditions_json, raises_json, param_nullability_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    mapping: dict[int, dict[str, object]] = {}
    for goid_raw, preconditions, raises_json, param_nullability in rows:
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "preconditions": _coerce_json(preconditions) or [],
            "raises": _coerce_json(raises_json) or [],
            "param_nullability": _coerce_json(param_nullability) or {},
        }
    return mapping


def _load_graph_metrics(
    con: DuckDBConnection, *, repo: str, commit: str
) -> dict[int, dict[str, int]]:
    rows: Iterable[tuple[object, int | None, int | None]] = con.execute(
        """
        SELECT function_goid_h128, call_fan_in, call_fan_out
        FROM analytics.graph_metrics_functions
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    mapping: dict[int, dict[str, int]] = {}
    for goid_raw, call_fan_in, call_fan_out in rows:
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "call_fan_in": int(call_fan_in or 0),
            "call_fan_out": int(call_fan_out or 0),
        }
    return mapping


def _load_module_meta(con: DuckDBConnection, *, repo: str, commit: str) -> dict[str, ModuleRecord]:
    rows: Iterable[tuple[str, str, list[object] | str | None]] = con.execute(
        """
        SELECT module, path, tags
        FROM core.modules
        WHERE COALESCE(repo, ?) = ?
          AND COALESCE(commit, ?) = ?
        """,
        [repo, repo, commit, commit],
    ).fetchall()
    meta: dict[str, ModuleRecord] = {}
    for module, path, tags in rows:
        normalized_path = normalize_rel_path(path) if path is not None else ""
        normalized_tags = _normalize_tags(tags)
        meta[str(module)] = ModuleRecord(path=normalized_path, tags=normalized_tags)
    return meta


def _classify_function(
    context: FunctionContext,
) -> tuple[str, float, str | None, dict[str, object]]:
    accumulator = RoleAccumulator()
    _score_tests(context, accumulator)
    _score_api_handlers(context, accumulator)
    _score_cli_commands(context, accumulator)
    _score_repositories(context, accumulator)
    _score_services(context, accumulator)
    _score_validators(context, accumulator)
    _score_config_loaders(context, accumulator)
    _score_helpers(context, accumulator)
    _score_module_tags(context, accumulator)
    _score_module_hints(context, accumulator)
    return accumulator.finalize()


def classify_function_role(
    context: FunctionContext,
) -> tuple[str, float, str | None, dict[str, object]]:
    """
    Classify a function using heuristic semantic role signals.

    Returns
    -------
    tuple[str, float, str | None, dict[str, object]]
        Primary role, score, optional secondary role, and debug metadata.
    """
    return _classify_function(context)


def _score_tests(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if context.rel_path_lower.startswith("tests") or "/tests/" in context.rel_path_lower:
        accumulator.bump("test", 0.6, "path:tests")
    if context.name.startswith("test_"):
        accumulator.bump("test", 0.5, "name:test_prefix")
    if any(
        dec.startswith("pytest.fixture") or dec.endswith("fixture") for dec in context.decorators
    ):
        accumulator.bump("test_helper", 0.9, "decorator:pytest.fixture")


def _score_api_handlers(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    for dec in context.decorators:
        dec_lower = dec.lower()
        if "router." in dec_lower or dec_lower.startswith(("get(", "post(")):
            accumulator.bump("api_handler", 0.7, f"decorator:{dec}", framework_hint="fastapi")
        elif ".route" in dec_lower or dec_lower.startswith("route"):
            accumulator.bump("api_handler", 0.6, f"decorator:{dec}", framework_hint="flask")
    if any(term in context.rel_path_lower for term in ("api", "route", "handler")):
        accumulator.bump("api_handler", 0.4, "path:api")
    if context.name.split("_", maxsplit=1)[0] in {"get", "post", "put", "delete", "patch"}:
        accumulator.bump("api_handler", 0.2, "name:http_verb")


def _score_cli_commands(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    for dec in context.decorators:
        dec_lower = dec.lower()
        if dec_lower.startswith("click.") or "click." in dec_lower:
            accumulator.bump("cli_command", 0.8, f"decorator:{dec}", framework_hint="click")
        if dec_lower.startswith("typer.") or "typer." in dec_lower:
            accumulator.bump("cli_command", 0.8, f"decorator:{dec}", framework_hint="typer")
    if any(term in context.rel_path_lower for term in ("cli", "commands", "scripts")):
        accumulator.bump("cli_command", 0.4, "path:cli")
    if context.name in {"main", "cli"}:
        accumulator.bump("cli_command", 0.3, "name:entrypoint")


def _score_repositories(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if context.effects.get("touches_db"):
        accumulator.bump("repository", 0.8, "effects:touches_db")
    if any(
        term in context.rel_path_lower for term in ("repository", "repositories", "database", "db/")
    ):
        accumulator.bump("repository", 0.5, "path:repository")
    if context.name.startswith(("get_", "fetch_", "save_", "update_", "delete_")):
        accumulator.bump("repository", 0.2, "name:data_access")


def _score_services(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if any(term in context.rel_path_lower for term in ("service", "use_case", "usecase")):
        accumulator.bump("service", 0.5, "path:service")
    if context.graph.get("call_fan_in", 0) > SERVICE_FAN_IN_THRESHOLD:
        accumulator.bump("service", 0.2, "graph:fan_in")
    if context.graph.get("call_fan_out", 0) > SERVICE_FAN_OUT_THRESHOLD:
        accumulator.bump("service", 0.2, "graph:fan_out")


def _score_validators(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if context.name.startswith(("validate", "check", "ensure", "assert")):
        accumulator.bump("validator", 0.6, "name:validator")
    raises_entries = _ensure_list(context.contracts.get("raises", []))
    if any(
        isinstance(entry, dict) and str(entry.get("exception", "")).lower().endswith("valueerror")
        for entry in raises_entries
    ):
        accumulator.bump("validator", 0.3, "raises:valueerror")
    if context.contracts.get("preconditions"):
        accumulator.bump("validator", 0.2, "guards:preconditions")


def _score_config_loaders(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if any(term in context.rel_path_lower for term in ("config", "settings", "env")):
        accumulator.bump("config_loader", 0.6, "path:config")
    if context.effects.get("uses_io"):
        accumulator.bump("config_loader", 0.2, "effects:uses_io")


def _score_helpers(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    if (
        not context.effects.get("touches_db")
        and not context.effects.get("uses_io")
        and (context.loc or 0) <= HELPER_LOC_THRESHOLD
    ):
        accumulator.bump("helper", 0.4, "small_pure_helper")


def _score_module_tags(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    for tag in context.tag_strings:
        if tag == "api":
            accumulator.bump("api_handler", 0.3, "tag:api")
        if tag in {"cli", "command"}:
            accumulator.bump("cli_command", 0.3, "tag:cli")
        if tag in {"repository", "db"}:
            accumulator.bump("repository", 0.3, "tag:repository")
        if tag == "service":
            accumulator.bump("service", 0.3, "tag:service")


def _score_module_hints(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    test_helper_score = accumulator.scores.get("test_helper", 0.0)
    if context.module_lower.startswith("tests") and test_helper_score == 0.0:
        accumulator.bump("test", 0.4, "module:tests")


def _classify_modules(
    *,
    module_meta: dict[str, ModuleRecord],
    roles_by_module: dict[str, list[tuple[str, float]]],
    repo: str,
    commit: str,
    now: datetime,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for module_name, meta in module_meta.items():
        tag_signals: list[str] = []
        scores: dict[str, float] = defaultdict(float)

        tags = [tag.lower() for tag in meta.tags if tag is not None]
        if "api" in tags:
            scores["api_handler"] += 0.3
            tag_signals.append("tag:api")
        if "cli" in tags:
            scores["cli_command"] += 0.3
            tag_signals.append("tag:cli")
        if "repository" in tags or "db" in tags:
            scores["repository"] += 0.3
            tag_signals.append("tag:repository")
        if "service" in tags:
            scores["service"] += 0.3
            tag_signals.append("tag:service")

        for role, confidence in roles_by_module.get(module_name, []):
            if role == "other":
                continue
            scores[role] += confidence

        role = "other"
        confidence = 0.0
        if scores:
            role, score = max(scores.items(), key=lambda item: item[1])
            if score >= ROLE_THRESHOLD:
                confidence = min(1.0, score)
            else:
                role = "other"

        role_scores = dict(scores)
        rows.append(
            (
                repo,
                commit,
                module_name,
                role,
                confidence,
                {"function_roles": role_scores, "tag_signals": tag_signals},
                now,
            )
        )
    return rows


def _decorator_names(decorators: list[ast.expr]) -> list[str]:
    names: list[str] = []
    for dec in decorators:
        text = safe_unparse(dec)
        if text:
            names.append(text)
        elif isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
    return names


def _coerce_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_tags(raw: object) -> list[str]:
    tags_obj = _coerce_json(raw)
    if tags_obj is None:
        return []
    if isinstance(tags_obj, list):
        return [str(tag) for tag in tags_obj if tag is not None]
    return [str(tags_obj)]


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
