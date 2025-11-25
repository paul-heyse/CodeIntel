"""Helpers for detecting HTTP/CLI/job entrypoints from source modules."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntryPointCandidate:
    """Detected entrypoint metadata before GOID resolution."""

    kind: str
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
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorSettings:
    """Toggle detection for supported frameworks."""

    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True


@dataclass(frozen=True)
class ImportContext:
    """Captured import and assignment context for a module."""

    alias_to_lib: dict[str, str]
    fastapi_targets: set[str]
    flask_targets: set[str]
    flask_blueprints: set[str]
    typer_targets: set[str]
    click_groups: set[str]


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
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    ctx = _build_import_context(tree)
    visitor = _EntryPointVisitor(settings, rel_path, module, ctx)
    visitor.visit(tree)
    return visitor.candidates


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _literal_int_list(node: ast.AST) -> list[int] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        ints: list[int] = []
        for elt in node.elts:
            value = _literal_int(elt)
            if value is None:
                return None
            ints.append(value)
        return ints
    return None


def _literal_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    return None


def _literal_value(node: ast.AST | None) -> object | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.NameConstant):
        return node.value
    return None


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError, TypeError):
        return None


def _decorator_to_str(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError, TypeError):
        return node.__class__.__name__


def _resolve_call_target(
    func: ast.AST, alias_to_lib: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    if isinstance(func, ast.Name):
        name = func.id
        return alias_to_lib.get(name), name, name
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = func.value.id
        return alias_to_lib.get(base), func.attr, base
    return None, None, None


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
        value = _literal_int(kw.value)
        if value is not None:
            return [value]
        list_value = _literal_int_list(kw.value)
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
            default = _literal_value(kw.value)
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
                "default": _literal_value(default),
            }
        )
    for kwarg, default in zip(node.args.kwonlyargs, list(node.args.kw_defaults), strict=False):
        params.append(
            {
                "name": kwarg.arg,
                "annotation": _annotation_to_str(kwarg.annotation),
                "default": _literal_value(default),
            }
        )
    if not params:
        return None
    return {"params": params}


def _build_import_context(tree: ast.AST) -> ImportContext:
    alias_to_lib = _collect_import_aliases(tree)
    fastapi_targets: set[str] = set()
    flask_targets: set[str] = set()
    flask_blueprints: set[str] = set()
    typer_targets: set[str] = set()
    click_groups: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = _assignment_target(node)
        if target is None or not isinstance(node.value, ast.Call):
            continue
        lib, callee, _ = _resolve_call_target(node.value.func, alias_to_lib)
        if lib == "fastapi" and callee in {"FastAPI", "APIRouter"}:
            fastapi_targets.add(target)
        elif lib == "flask" and callee in {"Flask", "Blueprint"}:
            if callee == "Flask":
                flask_targets.add(target)
            else:
                flask_blueprints.add(target)
        elif lib == "typer" and callee == "Typer":
            typer_targets.add(target)
        elif lib == "click" and callee in {"Group", "group"}:
            click_groups.add(target)

    return ImportContext(
        alias_to_lib=alias_to_lib,
        fastapi_targets=fastapi_targets,
        flask_targets=flask_targets,
        flask_blueprints=flask_blueprints,
        typer_targets=typer_targets,
        click_groups=click_groups,
    )


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


class _EntryPointVisitor(ast.NodeVisitor):
    def __init__(
        self,
        settings: DetectorSettings,
        rel_path: str,
        module: str,
        ctx: ImportContext,
    ) -> None:
        self.settings = settings
        self.rel_path = rel_path
        self.module = module
        self.ctx = ctx
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
            if (
                self.settings.detect_fastapi
                and lib == "fastapi"
                and attr in {"get", "post", "put", "delete", "patch", "options", "head"}
                and base in self.ctx.fastapi_targets
            ):
                self.candidates.append(
                    EntryPointCandidate(
                        kind="http",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        http_method=attr.upper() if attr is not None else None,
                        route_path=_extract_route_path(decorator),
                        status_codes=_extract_status_codes(decorator),
                        auth_required=auth_required,
                        evidence={"decorator": _decorator_to_str(decorator)},
                    )
                )
            elif (
                self.settings.detect_flask
                and lib == "flask"
                and attr == "route"
                and (base in self.ctx.flask_targets or base in self.ctx.flask_blueprints)
            ):
                methods = _extract_methods(decorator)
                http_method = methods[0] if methods else "GET"
                self.candidates.append(
                    EntryPointCandidate(
                        kind="http",
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        http_method=http_method,
                        route_path=_extract_route_path(decorator),
                        status_codes=_extract_status_codes(decorator),
                        auth_required=auth_required,
                        evidence={"decorator": _decorator_to_str(decorator)},
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
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=explicit_name or command_name,
                        arguments_schema=click_args,
                        evidence={"decorator": _decorator_to_str(decorator)},
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
                        rel_path=self.rel_path,
                        module=self.module,
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 0) or 0),
                        end_lineno=getattr(node, "end_lineno", None),
                        command_name=explicit_name or command_name,
                        arguments_schema=typer_schema,
                        evidence={"decorator": _decorator_to_str(decorator)},
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
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    lineno=int(getattr(node, "lineno", 0) or 0),
                    end_lineno=getattr(node, "end_lineno", None),
                    schedule=schedule,
                    trigger=trigger,
                    evidence={"decorator": _decorator_to_str(decorator)},
                )
            )
