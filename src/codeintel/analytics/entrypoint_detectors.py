"""Helpers for detecting HTTP/CLI/job entrypoints from source modules."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from codeintel.analytics.ast_utils import (
    literal_bool,
    literal_int,
    literal_int_sequence,
    literal_str,
    literal_value,
    resolve_call_target,
    safe_unparse,
    snippet_from_lines,
)
from codeintel.analytics.evidence import EvidenceCollector


@dataclass(frozen=True)
class EntryPointCandidate:
    """Detected entrypoint metadata before GOID resolution."""

    kind: str
    framework: str | None
    rel_path: str
    module: str
    qualname: str
    lineno: int
    end_lineno: int | None
    http_method: str | None = None
    route_path: str | None = None
    status_codes: list[int] | None = None
    auth_required: bool | None = None
    command_name: str | None = None
    arguments_schema: Mapping[str, object | list[dict[str, object]]] | None = None
    schedule: str | None = None
    trigger: str | None = None
    extra: dict[str, object] | None = None
    evidence: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class DetectorSettings:
    """Toggle detection for supported frameworks."""

    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True
    detect_django: bool = True
    detect_celery: bool = True
    detect_airflow: bool = True
    detect_generic_routes: bool = True


@dataclass(frozen=True)
class ImportContext:
    """Captured import and assignment context for a module."""

    alias_to_lib: dict[str, str]
    fastapi_targets: set[str]
    flask_targets: set[str]
    flask_blueprints: set[str]
    typer_targets: set[str]
    click_groups: set[str]
    django_url_helpers: set[str]
    celery_apps: set[str]


@dataclass
class _ImportTargets:
    fastapi_targets: set[str] = field(default_factory=set)
    flask_targets: set[str] = field(default_factory=set)
    flask_blueprints: set[str] = field(default_factory=set)
    typer_targets: set[str] = field(default_factory=set)
    click_groups: set[str] = field(default_factory=set)
    django_url_helpers: set[str] = field(default_factory=set)
    celery_apps: set[str] = field(default_factory=set)


def detect_entrypoints(
    source: str,
    *,
    rel_path: str,
    module: str,
    settings: DetectorSettings,
) -> list[EntryPointCandidate]:
    """
    Parse module source and return detected entrypoint candidates.

    Parameters
    ----------
    source
        Module source text.
    rel_path
        Repository-relative path of the module.
    module
        Dotted module name.
    settings
        Detection toggles.

    Returns
    -------
    list[EntryPointCandidate]
        Entrypoint candidates detected in the module, or an empty list when parsing fails.
    """
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    ctx = _build_import_context(tree)
    visitor = _EntryPointVisitor(settings, rel_path, module, ctx, lines)
    visitor.visit(tree)
    candidates = list(visitor.candidates)
    if settings.detect_django:
        candidates.extend(
            _detect_django_urlpatterns(tree, rel_path=rel_path, module=module, ctx=ctx, lines=lines)
        )
    return candidates


def _literal_str(node: ast.AST) -> str | None:
    return literal_str(node)


def _literal_bool(node: ast.AST) -> bool | None:
    return literal_bool(node)


def _literal_int(node: ast.AST) -> int | None:
    return literal_int(node)


def _literal_int_list(node: ast.AST) -> list[int] | None:
    return literal_int_sequence(node)


def _literal_value(node: ast.AST | None) -> object | None:
    return literal_value(node)


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = safe_unparse(node)
    return text if text else None


def _decorator_to_str(node: ast.AST) -> str:
    text = safe_unparse(node)
    return text or node.__class__.__name__


def _node_evidence(
    *,
    rel_path: str,
    lines: Sequence[str],
    node: ast.AST,
    details: dict[str, object],
    tags: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    collector = EvidenceCollector()
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", lineno)
    collector.add_sample(
        path=rel_path,
        line_span=(lineno, end_lineno),
        snippet=snippet_from_lines(lines, lineno, end_lineno),
        details=details,
        tags=tuple(tags or ("decorator",)),
    )
    return collector.to_dicts()


def _resolve_call_target(
    func: ast.AST, alias_to_lib: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    target = resolve_call_target(func, alias_to_lib)
    return target.library, target.attribute, target.base


def _has_auth_decorator(decorators: Iterable[ast.AST]) -> bool | None:
    for decorator in decorators:
        name = None
        if isinstance(decorator, ast.Name):
            name = decorator.id
        elif isinstance(decorator, ast.Attribute):
            name = decorator.attr
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
        if name is None:
            continue
        lower = name.lower()
        if "auth" in lower or "login" in lower:
            return True
    return None


def _extract_route_path(call: ast.Call) -> str | None:
    if call.args:
        path = _literal_str(call.args[0])
        if path:
            return path
    for kw in call.keywords:
        if kw.arg in {"path", "route"}:
            path = _literal_str(kw.value)
            if path:
                return path
    return None


def _extract_status_codes(call: ast.Call) -> list[int] | None:
    for kw in call.keywords:
        if kw.arg != "status_code":
            continue
        value = literal_int(kw.value)
        if value is not None:
            return [value]
        list_value = literal_int_sequence(kw.value)
        if list_value is not None:
            return list_value
    return None


def _extract_methods(call: ast.Call) -> list[str] | None:
    for kw in call.keywords:
        if kw.arg != "methods":
            continue
        if isinstance(kw.value, (ast.List, ast.Tuple)):
            methods: list[str] = []
            for elt in kw.value.elts:
                text = _literal_str(elt)
                if text:
                    methods.append(text.upper())
            if methods:
                return methods
    return None


def _extract_click_arguments(
    decorators: Iterable[ast.AST],
    alias_to_lib: dict[str, str],
    click_groups: set[str],
) -> dict[str, object | list[dict[str, object]]] | None:
    options: list[dict[str, object]] = []
    arguments: list[dict[str, object]] = []
    for decorator in decorators:
        parsed = _click_payload(decorator, alias_to_lib, click_groups)
        if parsed is None:
            continue
        kind, payload = parsed
        if kind == "option":
            options.append(payload)
        else:
            arguments.append(payload)
    if not options and not arguments:
        return None
    schema: dict[str, object | list[dict[str, object]]] = {
        "options": options,
        "arguments": arguments,
    }
    return schema


def _click_payload(
    decorator: ast.AST, alias_to_lib: dict[str, str], click_groups: set[str]
) -> tuple[str, dict[str, object]] | None:
    if not isinstance(decorator, ast.Call):
        return None
    kind = _resolve_click_target(decorator, alias_to_lib, click_groups)
    if kind is None:
        return None
    names = [_literal_str(arg) for arg in decorator.args if _literal_str(arg)]
    option_type, required, default = _click_keyword_values(decorator.keywords)
    payload: dict[str, object] = {"flags": names}
    if option_type is not None:
        payload["type"] = option_type
    if required is not None:
        payload["required"] = required
    if default is not None:
        payload["default"] = default
    return kind, payload


def _resolve_click_target(
    decorator: ast.Call, alias_to_lib: dict[str, str], click_groups: set[str]
) -> str | None:
    lib, attr, base = _resolve_call_target(decorator.func, alias_to_lib)
    if attr not in {"option", "argument"}:
        return None
    if lib != "click" and base not in click_groups:
        return None
    return "option" if attr == "option" else "argument"


def _click_keyword_values(
    keywords: list[ast.keyword],
) -> tuple[str | None, bool | None, object | None]:
    option_type = None
    required = None
    default = None
    for kw in keywords:
        if kw.arg == "type":
            option_type = _annotation_to_str(kw.value)
        elif kw.arg == "required":
            required = _literal_bool(kw.value)
        elif kw.arg == "default":
            default = literal_value(kw.value)
    return option_type, required, default


def _typer_arguments_schema(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, object] | None:
    params: list[dict[str, object]] = []
    positional = list(node.args.args)
    defaults = list(node.args.defaults)
    pad = [None] * (len(positional) - len(defaults))
    for arg, default in zip(positional, pad + defaults, strict=False):
        if arg.arg in {"self", "cls"}:
            continue
        params.append(
            {
                "name": arg.arg,
                "annotation": _annotation_to_str(arg.annotation),
                "default": literal_value(default),
            }
        )
    for kwarg, default in zip(node.args.kwonlyargs, list(node.args.kw_defaults), strict=False):
        params.append(
            {
                "name": kwarg.arg,
                "annotation": _annotation_to_str(kwarg.annotation),
                "default": literal_value(default),
            }
        )
    if not params:
        return None
    return {"params": params}


def _build_import_context(tree: ast.AST) -> ImportContext:
    alias_to_lib = _collect_import_aliases(tree)
    targets = _scan_assignment_targets(tree, alias_to_lib)
    return ImportContext(
        alias_to_lib=alias_to_lib,
        fastapi_targets=targets.fastapi_targets,
        flask_targets=targets.flask_targets,
        flask_blueprints=targets.flask_blueprints,
        typer_targets=targets.typer_targets,
        click_groups=targets.click_groups,
        django_url_helpers=targets.django_url_helpers,
        celery_apps=targets.celery_apps,
    )


def _scan_assignment_targets(tree: ast.AST, alias_to_lib: dict[str, str]) -> _ImportTargets:
    targets = _ImportTargets()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = _assignment_target(node)
        if target is None or not isinstance(node.value, ast.Call):
            continue
        lib, callee, _ = _resolve_call_target(node.value.func, alias_to_lib)
        _update_assignment_targets(lib, callee, target, targets)
    return targets


def _update_assignment_targets(
    lib: str | None, callee: str | None, target: str, targets: _ImportTargets
) -> None:
    if lib == "fastapi" and callee in {"FastAPI", "APIRouter"}:
        targets.fastapi_targets.add(target)
        return
    if lib == "flask" and callee in {"Flask", "Blueprint"}:
        if callee == "Flask":
            targets.flask_targets.add(target)
        else:
            targets.flask_blueprints.add(target)
        return
    if lib == "typer" and callee == "Typer":
        targets.typer_targets.add(target)
        return
    if lib == "click" and callee in {"Group", "group"}:
        targets.click_groups.add(target)
        return
    if lib == "django" and callee in {"path", "re_path", "url"}:
        targets.django_url_helpers.add(target)
        return
    if lib == "celery" and callee == "Celery":
        targets.celery_apps.add(target)


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    alias_to_lib: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                alias_to_lib[alias.asname or alias.name] = root
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            for alias in node.names:
                alias_to_lib[alias.asname or alias.name] = root
    return alias_to_lib


def _assignment_target(node: ast.Assign) -> str | None:
    if not node.targets or not isinstance(node.targets[0], ast.Name):
        return None
    return node.targets[0].id


def _view_qualname(module: str, view_node: ast.AST) -> str | None:
    if isinstance(view_node, ast.Name):
        return f"{module}.{view_node.id}"
    if isinstance(view_node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST | None = view_node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return f"{module}." + ".".join(parts)
    return None


def _detect_django_urlpatterns(
    tree: ast.AST,
    *,
    rel_path: str,
    module: str,
    ctx: ImportContext,
    lines: Sequence[str],
) -> list[EntryPointCandidate]:
    candidates: list[EntryPointCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "urlpatterns" for t in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for elt in node.value.elts:
            if not isinstance(elt, ast.Call):
                continue
            lib, attr, base = _resolve_call_target(elt.func, ctx.alias_to_lib)
            if (
                lib != "django"
                and base not in ctx.django_url_helpers
                and attr
                not in {
                    "path",
                    "re_path",
                    "url",
                }
            ):
                continue
            route_path = _literal_str(elt.args[0]) if elt.args else None
            view_node = elt.args[1] if len(elt.args) > 1 else None
            qualname = _view_qualname(module, view_node) if view_node else None
            if qualname is None:
                continue
            candidates.append(
                EntryPointCandidate(
                    kind="http",
                    framework="django",
                    rel_path=rel_path,
                    module=module,
                    qualname=qualname,
                    lineno=int(getattr(elt, "lineno", 0) or 0),
                    end_lineno=getattr(elt, "end_lineno", None),
                    route_path=route_path,
                    evidence=_node_evidence(
                        rel_path=rel_path,
                        lines=lines,
                        node=elt,
                        details={"decorator": _decorator_to_str(elt.func), "kind": "urlpatterns"},
                        tags=("urlpatterns",),
                    ),
                )
            )
    return candidates


class _EntryPointVisitor(ast.NodeVisitor):
    def __init__(
        self,
        settings: DetectorSettings,
        rel_path: str,
        module: str,
        ctx: ImportContext,
        lines: Sequence[str],
    ) -> None:
        self.settings = settings
        self.rel_path = rel_path
        self.module = module
        self.ctx = ctx
        self._lines = lines
        self.scope: list[str] = []
        self.candidates: list[EntryPointCandidate] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = self._qualname(node.name)
        self.scope.append(node.name)
        self._collect_candidates(node, qualname)
        self.generic_visit(node)
        self.scope.pop()

    def _qualname(self, name: str) -> str:
        if self.scope:
            return f"{self.module}." + ".".join([*self.scope, name])
        return f"{self.module}.{name}"

    def _collect_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        auth_required = _has_auth_decorator(node.decorator_list)
        self._collect_http_candidates(node, qualname, auth_required=auth_required)
        self._collect_cli_candidates(node, qualname)
        self._collect_job_candidates(node, qualname)
        self._collect_celery_candidates(node, qualname)
        self._collect_airflow_candidates(node, qualname)
        self._collect_generic_route_candidates(node, qualname)

    def _collect_http_candidates(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualname: str,
        *,
        auth_required: bool | None,
    ) -> None:
        if not (self.settings.detect_fastapi or self.settings.detect_flask):
            return
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            lib, attr, base = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            is_fastapi_target = base in self.ctx.fastapi_targets
            if (
                self.settings.detect_fastapi
                and (lib == "fastapi" or is_fastapi_target)
                and attr in {"get", "post", "put", "delete", "patch", "options", "head"}
                and is_fastapi_target
            ):
                self.candidates.append(
                    EntryPointCandidate(
                        kind="http",
                        framework="fastapi",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        http_method=attr.upper() if attr is not None else None,
                        route_path=_extract_route_path(decorator),
                        status_codes=_extract_status_codes(decorator),
                        auth_required=auth_required,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "fastapi"),
                        ),
                    )
                )
            elif (
                self.settings.detect_flask
                and (
                    lib == "flask"
                    or base in self.ctx.flask_targets
                    or base in self.ctx.flask_blueprints
                )
                and attr == "route"
            ):
                methods = _extract_methods(decorator)
                http_method = methods[0] if methods else "GET"
                self.candidates.append(
                    EntryPointCandidate(
                        kind="http",
                        framework="flask",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        http_method=http_method,
                        route_path=_extract_route_path(decorator),
                        status_codes=_extract_status_codes(decorator),
                        auth_required=auth_required,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "flask"),
                        ),
                    )
                )

    def _collect_cli_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        if not (self.settings.detect_click or self.settings.detect_typer):
            return
        command_name = node.name
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            lib, attr, base = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            explicit_name = None
            for kw in decorator.keywords:
                if kw.arg == "name":
                    explicit_name = _literal_str(kw.value)
                    break
            if (
                self.settings.detect_click
                and attr in {"command", "group"}
                and (lib == "click" or base in self.ctx.click_groups)
            ):
                click_args = _extract_click_arguments(
                    node.decorator_list, self.ctx.alias_to_lib, self.ctx.click_groups
                )
                self.candidates.append(
                    EntryPointCandidate(
                        kind="cli",
                        framework="click",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=explicit_name or command_name,
                        arguments_schema=click_args,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "click"),
                        ),
                    )
                )
            elif (
                self.settings.detect_typer
                and attr == "command"
                and (lib == "typer" or base in self.ctx.typer_targets)
            ):
                typer_schema = _typer_arguments_schema(node)
                self.candidates.append(
                    EntryPointCandidate(
                        kind="cli",
                        framework="typer",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=explicit_name or command_name,
                        arguments_schema=typer_schema,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "typer"),
                        ),
                    )
                )

    def _collect_job_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        if not self.settings.detect_cron:
            return
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            lib, attr, base = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            trigger = None
            schedule = None
            kind = "cron"
            if attr in {"scheduled_job", "cron", "interval"}:
                trigger = attr
                schedule = _literal_str(decorator.args[0]) if decorator.args else None
            elif (
                lib == "fastapi"
                and attr == "on_event"
                and base in self.ctx.fastapi_targets
                and decorator.args
            ):
                trigger = _literal_str(decorator.args[0]) or "event"
                kind = "event"
            else:
                continue
            self.candidates.append(
                EntryPointCandidate(
                    kind=kind,
                    framework="scheduler",
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    lineno=int(getattr(node, "lineno", 0) or 0),
                    end_lineno=getattr(node, "end_lineno", None),
                    schedule=schedule,
                    trigger=trigger,
                    evidence=_node_evidence(
                        rel_path=self.rel_path,
                        lines=self._lines,
                        node=decorator,
                        details={"decorator": _decorator_to_str(decorator)},
                        tags=("decorator", "scheduler"),
                    ),
                )
            )

    def _collect_celery_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        if not self.settings.detect_celery:
            return
        task_name: str = node.name
        queue: str | None = None
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            lib, attr, base = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            is_shared_task = lib == "celery" and attr == "shared_task"
            is_app_task = attr == "task" and (lib == "celery" or base in self.ctx.celery_apps)
            if not (is_shared_task or is_app_task):
                continue
            for kw in decorator.keywords:
                if kw.arg == "name":
                    task_name = _literal_str(kw.value) or task_name
                if kw.arg == "queue":
                    queue = _literal_str(kw.value) or queue
            extra: dict[str, object] = {}
            if queue:
                extra["queue"] = queue
            self.candidates.append(
                EntryPointCandidate(
                    kind="task",
                    framework="celery",
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    lineno=int(getattr(node, "lineno", 0) or 0),
                    end_lineno=getattr(node, "end_lineno", None),
                    command_name=task_name,
                    extra=extra or None,
                    evidence=_node_evidence(
                        rel_path=self.rel_path,
                        lines=self._lines,
                        node=decorator,
                        details={"decorator": _decorator_to_str(decorator)},
                        tags=("decorator", "celery"),
                    ),
                )
            )

    def _collect_airflow_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        if not self.settings.detect_airflow:
            return
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            lib, attr, _ = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            if lib != "airflow":
                continue
            if attr == "task":
                task_id = _literal_str(decorator.args[0]) if decorator.args else None
                self.candidates.append(
                    EntryPointCandidate(
                        kind="task",
                        framework="airflow",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=task_id or node.name,
                        extra={"task_id": task_id} if task_id else None,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "airflow"),
                        ),
                    )
                )
            if attr == "dag":
                dag_id = _literal_str(decorator.args[0]) if decorator.args else None
                self.candidates.append(
                    EntryPointCandidate(
                        kind="dag",
                        framework="airflow",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=dag_id or node.name,
                        extra={"dag_id": dag_id} if dag_id else None,
                        evidence=_node_evidence(
                            rel_path=self.rel_path,
                            lines=self._lines,
                            node=decorator,
                            details={"decorator": _decorator_to_str(decorator)},
                            tags=("decorator", "airflow"),
                        ),
                    )
                )

    def _collect_generic_route_candidates(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str
    ) -> None:
        if not self.settings.detect_generic_routes:
            return
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            _, attr, _ = _resolve_call_target(decorator.func, self.ctx.alias_to_lib)
            if attr not in {"route", "add_route", "add_url_rule"}:
                continue
            route_path = _extract_route_path(decorator)
            self.candidates.append(
                EntryPointCandidate(
                    kind="http",
                    framework="generic",
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    lineno=int(getattr(node, "lineno", 0) or 0),
                    end_lineno=getattr(node, "end_lineno", None),
                    route_path=route_path,
                    evidence=_node_evidence(
                        rel_path=self.rel_path,
                        lines=self._lines,
                        node=decorator,
                        details={"decorator": _decorator_to_str(decorator)},
                        tags=("decorator", "generic_route"),
                    ),
                )
            )
