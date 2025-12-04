"""Pure semantic role classification functions.

This module contains the core classification logic for determining
semantic roles of functions and modules. All functions are pure
(no I/O or side effects).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.analytics.utilities.ast import safe_unparse

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst

ROLE_THRESHOLD = 0.35
SERVICE_FAN_IN_THRESHOLD = 5
SERVICE_FAN_OUT_THRESHOLD = 5
HELPER_LOC_THRESHOLD = 20


@dataclass(frozen=True)
class FunctionContext:
    """Classification context for a single function."""

    goid: int
    rel_path: str
    qualname: str
    decorators: list[str]
    effects: dict[str, object]
    contracts: dict[str, object]
    module_tags: list[str]
    module_name: str | None
    graph: dict[str, int]
    loc: int | None
    features: FunctionAstFeatures | None = None

    @property
    def name(self) -> str:
        """Return the unqualified function name.

        Returns
        -------
        str
            Function name without module qualifiers.
        """
        return self.qualname.rsplit(".", maxsplit=1)[-1]

    @property
    def rel_path_lower(self) -> str:
        """Return lower-case relative path for path-based heuristics.

        Returns
        -------
        str
            Path normalized to lower-case.
        """
        return self.rel_path.lower()

    @property
    def module_lower(self) -> str:
        """Return lower-case module name for module-level hints.

        Returns
        -------
        str
            Module name in lower-case.
        """
        return (self.module_name or "").lower()

    @property
    def tag_strings(self) -> list[str]:
        """Return normalized module tags.

        Returns
        -------
        list[str]
            Tag strings normalized to lower-case.
        """
        return [str(tag).lower() for tag in self.module_tags if tag is not None]


@dataclass
class RoleAccumulator:
    """Aggregate scoring signals for semantic roles."""

    scores: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    sources: defaultdict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    frameworks: dict[str, str | None] = field(default_factory=dict)

    def bump(
        self, role: str, amount: float, reason: str, framework_hint: str | None = None
    ) -> None:
        """Increase a role score and track its provenance.

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
        """Return the winning role with confidence and metadata.

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
    features: dict[int, FunctionAstFeatures]


@dataclass(frozen=True)
class ModuleRecord:
    """Metadata for a module path and tags."""

    path: str
    tags: list[str]


def classify_function_role(
    context: FunctionContext,
) -> tuple[str, float, str | None, dict[str, object]]:
    """Classify a function using heuristic semantic role signals.

    Parameters
    ----------
    context
        Function context containing decorators, effects, path, etc.

    Returns
    -------
    tuple[str, float, str | None, dict[str, object]]
        Primary role, score, optional framework hint, and debug metadata.
    """
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


def classify_modules(
    *,
    module_meta: dict[str, ModuleRecord],
    roles_by_module: dict[str, list[tuple[str, float]]],
    repo: str,
    commit: str,
    now: datetime,
) -> list[tuple[object, ...]]:
    """Classify modules based on their constituent function roles.

    Parameters
    ----------
    module_meta
        Mapping of module names to their metadata.
    roles_by_module
        Mapping of module names to list of (role, confidence) tuples.
    repo
        Repository identifier.
    commit
        Commit identifier.
    now
        Timestamp for row creation.

    Returns
    -------
    list[tuple[object, ...]]
        Rows for module role classification.
    """
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


def decorator_names(decorators: list[ast.expr]) -> list[str]:
    """Extract decorator names from AST nodes.

    Parameters
    ----------
    decorators
        List of decorator AST expressions.

    Returns
    -------
    list[str]
        Human-readable decorator names.
    """
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


# =============================================================================
# Internal scoring functions
# =============================================================================


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
    features = context.features
    decorators = features.decorators if features is not None else context.decorators

    if features is not None and features.http_server_libs:
        libs = ",".join(sorted(features.http_server_libs))
        accumulator.bump("api_handler", 0.7, f"http_server_libs:{libs}")

    _score_api_decorators(
        decorators,
        accumulator,
        fastapi_weight=0.2 if features is not None else 0.7,
        flask_weight=0.2 if features is not None else 0.6,
    )

    if any(term in context.rel_path_lower for term in ("api", "route", "handler")):
        weight = 0.2 if features is not None else 0.4
        accumulator.bump("api_handler", weight, "path:api")
    if context.name.split("_", maxsplit=1)[0] in {"get", "post", "put", "delete", "patch"}:
        weight = 0.1 if features is not None else 0.2
        accumulator.bump("api_handler", weight, "name:http_verb")


def _score_api_decorators(
    decorators: Iterable[str],
    accumulator: RoleAccumulator,
    *,
    fastapi_weight: float,
    flask_weight: float,
) -> None:
    for dec in decorators:
        dec_lower = dec.lower()
        if "router." in dec_lower or dec_lower.startswith(("get(", "post(")):
            accumulator.bump(
                "api_handler",
                fastapi_weight,
                f"decorator:{dec}",
                framework_hint="fastapi",
            )
        elif ".route" in dec_lower or dec_lower.startswith("route"):
            accumulator.bump(
                "api_handler",
                flask_weight,
                f"decorator:{dec}",
                framework_hint="flask",
            )


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

    features = context.features
    if features is not None:
        if "click" in features.libraries_used:
            accumulator.bump("cli_command", 0.5, "library:click")
        if "typer" in features.libraries_used:
            accumulator.bump("cli_command", 0.5, "library:typer")


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


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


__all__ = [
    "FunctionContext",
    "ModuleRecord",
    "RoleAccumulator",
    "RoleArtifacts",
    "classify_function_role",
    "classify_modules",
    "decorator_names",
]
