"""Infer lightweight contracts (nullability, guards, raises) for functions."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.utilities.ast import literal_int, literal_value, safe_unparse
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression
from codeintel.storage.query_results import records_from_relation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

NULLABLE_MARKERS = ("optional", "none", "|none")
SINGLE_COMPARATOR = 1
MIN_ISINSTANCE_ARGS = 2


@dataclass(frozen=True)
class ConditionContext:
    """Shared context for extracting preconditions."""

    params: set[str]
    rel_path: str
    line: int | None
    limit: int


@dataclass(frozen=True)
class _RowInputs:
    """Inputs required to build contract rows."""

    repo: str
    commit: str
    max_conditions_per_func: int
    goids: set[int]
    ast_by_goid: dict[int, FunctionAst]
    doc_map: dict[tuple[str, str], dict[str, object]]
    type_map: dict[int, dict[str, object]]
    now: datetime


def build_function_contracts_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    function_ast_map: dict[int, FunctionAst] | None = None,
    catalog: FunctionCatalogProvider | None = None,
    max_conditions_per_func: int = 64,
) -> list[dict[str, object]]:
    """
    Build contract rows for `analytics.function_contracts` without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRelationSaver.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    function_ast_map
        Mapping of GOID to parsed function AST (from AstProvider).
    catalog
        Function catalog provider (from CatalogProvider).
    max_conditions_per_func
        Maximum number of preconditions/postconditions/raises per function.

    Returns
    -------
    list[dict[str, object]]
        Contract rows ready for persistence.
    """
    ast_by_goid = function_ast_map or {}

    if catalog is not None:
        all_goids = {span.goid for span in catalog.catalog().function_spans}
    else:
        all_goids = set()

    doc_map = _load_docstrings(gateway, repo=snapshot.repo, commit=snapshot.commit)
    type_map = _load_function_types(gateway, repo=snapshot.repo, commit=snapshot.commit)

    return _build_rows(
        _RowInputs(
            repo=snapshot.repo,
            commit=snapshot.commit,
            max_conditions_per_func=max_conditions_per_func,
            goids=all_goids,
            ast_by_goid=ast_by_goid,
            doc_map=doc_map,
            type_map=type_map,
            now=datetime.now(tz=UTC),
        )
    )


def _build_rows(inputs: _RowInputs) -> list[dict[str, object]]:
    """Build contract rows from ASTs, docs, and types.

    Returns
    -------
    list[dict[str, object]]
        Contract rows ready for persistence.
    """
    rows: list[dict[str, object]] = []
    for goid in inputs.goids:
        info = inputs.ast_by_goid.get(goid)
        doc_key = (info.rel_path, info.qualname) if info is not None else None
        doc = inputs.doc_map.get(doc_key) if doc_key is not None else None
        type_info = inputs.type_map.get(goid)

        if info is None:
            rows.append(
                {
                    "repo": inputs.repo,
                    "commit": inputs.commit,
                    "function_goid_h128": goid,
                    "preconditions_json": [],
                    "postconditions_json": [],
                    "raises_json": [],
                    "param_nullability_json": {},
                    "return_nullability": None,
                    "contract_confidence": 0.0,
                    "created_at": inputs.now,
                }
            )
            continue

        contracts = _analyze_function(
            info,
            doc=doc,
            type_info=type_info,
            max_conditions=inputs.max_conditions_per_func,
        )
        rows.append(
            {
                "repo": inputs.repo,
                "commit": inputs.commit,
                "function_goid_h128": goid,
                "preconditions_json": contracts["preconditions"],
                "postconditions_json": contracts["postconditions"],
                "raises_json": contracts["raises"],
                "param_nullability_json": contracts["param_nullability"],
                "return_nullability": contracts["return_nullability"],
                "contract_confidence": contracts["confidence"],
                "created_at": inputs.now,
            }
        )
    return rows


def _load_docstrings(
    gateway: StorageGateway, *, repo: str, commit: str
) -> dict[tuple[str, str], dict[str, object]]:
    predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    relation = (
        gateway.relation_from_table_key("core.docstrings")
        .filter(predicate)
        .select("rel_path", "qualname", "params", "returns")
    )
    rows = _normalize_records(records_from_relation(relation))
    mapping: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        rel_path = row["rel_path"]
        qualname = row["qualname"]
        params = row["params"]
        returns = row["returns"]
        mapping[str(rel_path), str(qualname)] = {
            "params": _coerce_json(params) or [],
            "returns": _coerce_json(returns),
        }
    return mapping


def _load_function_types(
    gateway: StorageGateway, *, repo: str, commit: str
) -> dict[int, dict[str, object]]:
    predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    relation = (
        gateway.relation_from_table_key("analytics.function_types")
        .filter(predicate)
        .select("function_goid_h128", "return_type", "param_types")
    )
    rows = _normalize_records(records_from_relation(relation))
    mapping: dict[int, dict[str, object]] = {}
    for row in rows:
        goid = normalize_decimal_id(row["function_goid_h128"])
        if goid is None:
            continue
        mapping[goid] = {
            "return_type": str(row["return_type"]) if row["return_type"] is not None else None,
            "param_types": _coerce_json(row["param_types"]) or {},
        }
    return mapping


def _normalize_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    return [{str(key): value for key, value in record.items()} for record in records]


def _coerce_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _analyze_function(
    func: FunctionAst,
    *,
    doc: dict[str, object] | None,
    type_info: dict[str, object] | None,
    max_conditions: int,
) -> dict[str, object]:
    param_nullability = _infer_param_nullability(func, doc, type_info)
    return_nullability = _infer_return_nullability(func, doc, type_info)

    visitor = _ContractVisitor(
        params=set(param_nullability.keys()),
        rel_path=func.rel_path,
        limit=max_conditions,
    )
    visitor.visit(func.node)

    postconditions: list[dict[str, object]] = []
    if return_nullability == "non_null":
        postconditions.append(
            {
                "kind": "non_null",
                "param": "return",
                "source": "type_hint",
                "rel_path": func.rel_path,
                "line": func.end_line,
            }
        )
    if _returns_bool(type_info, func.qualname):
        postconditions.append(
            {
                "kind": "returns_bool_predicate",
                "param": "return",
                "source": "type_hint",
                "rel_path": func.rel_path,
                "line": func.end_line,
            }
        )

    confidence = _contract_confidence(
        has_types=type_info is not None,
        has_doc=doc is not None,
        has_guards=bool(visitor.preconditions or visitor.raises),
    )

    return {
        "preconditions": visitor.preconditions[:max_conditions],
        "postconditions": postconditions[:max_conditions],
        "raises": visitor.raises[:max_conditions],
        "param_nullability": param_nullability,
        "return_nullability": return_nullability,
        "confidence": confidence,
    }


def _infer_param_nullability(
    func: FunctionAst,
    doc: dict[str, object] | None,
    type_info: dict[str, object] | None,
) -> dict[str, str]:
    param_types = (type_info or {}).get("param_types")
    annotations: dict[str, object] = param_types if isinstance(param_types, dict) else {}
    annotation_map = {
        name: str(ann) if ann is not None else None for name, ann in annotations.items()
    }
    doc_params = {}
    if doc is not None:
        raw_params = doc.get("params", [])
        if isinstance(raw_params, list):
            for entry in raw_params:
                if isinstance(entry, dict):
                    doc_params[entry.get("name")] = str(entry.get("desc") or "")

    nullability: dict[str, str] = {}
    for arg in _param_names(func.node):
        ann = annotation_map.get(arg)
        if ann:
            nullability[arg] = "nullable" if _is_nullable_type(ann) else "non_null"
        else:
            nullability[arg] = "unknown"

        doc_hint = doc_params.get(arg, "") or ""
        if "optional" in doc_hint.lower() or "may be none" in doc_hint.lower():
            nullability[arg] = "nullable"
        elif "required" in doc_hint.lower() or "must not be none" in doc_hint.lower():
            nullability[arg] = "non_null"
    return nullability


def _infer_return_nullability(
    func: FunctionAst,
    doc: dict[str, object] | None,
    type_info: dict[str, object] | None,
) -> str | None:
    return_type = None
    if type_info is not None:
        return_type = type_info.get("return_type")
        if return_type is not None:
            return_type = str(return_type)
    if return_type:
        return "nullable" if _is_nullable_type(return_type) else "non_null"

    doc_return = None
    if doc is not None:
        doc_return = doc.get("returns")
        if isinstance(doc_return, dict):
            desc = str(doc_return.get("desc") or "").lower()
            if "none" in desc or "null" in desc:
                return "nullable"
    if _has_explicit_none_return(func.node):
        return "nullable"
    return None


def _is_nullable_type(type_str: str) -> bool:
    normalized = type_str.replace(" ", "").lower()
    return any(marker in normalized for marker in NULLABLE_MARKERS)


def _returns_bool(type_info: dict[str, object] | None, qualname: str) -> bool:
    if type_info is None:
        return False
    return_type = type_info.get("return_type")
    if return_type is None:
        return False
    return_str = str(return_type).lower()
    if "bool" in return_str:
        return True
    name = qualname.rsplit(".", maxsplit=1)[-1]
    return name.startswith(("is_", "has_", "should_"))


def _param_names(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    names: list[str] = []
    args = node.args
    params = list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        params.append(args.vararg)
    if args.kwarg is not None:
        params.append(args.kwarg)
    for param in params:
        if param.arg in {"self", "cls"}:
            continue
        names.append(param.arg)
    return names


def _has_explicit_none_return(node: ast.AST) -> bool:
    class _ReturnVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.has_none = False

        def visit_Return(self, node: ast.Return) -> None:
            if node.value is None or (
                isinstance(node.value, ast.Constant) and node.value.value is None
            ):
                self.has_none = True
            self.generic_visit(node)

    visitor = _ReturnVisitor()
    visitor.visit(node)
    return visitor.has_none


class _ContractVisitor(ast.NodeVisitor):
    """Extract preconditions and raises from a function AST."""

    def __init__(self, *, params: set[str], rel_path: str, limit: int) -> None:
        self.params = params
        self.rel_path = rel_path
        self.limit = limit
        self.preconditions: list[dict[str, object]] = []
        self.raises: list[dict[str, object]] = []

    def visit_Assert(self, node: ast.Assert) -> None:
        ctx = ConditionContext(
            params=self.params,
            rel_path=self.rel_path,
            line=getattr(node, "lineno", None),
            limit=self.limit,
        )
        self.preconditions.extend(
            _preconditions_from_test(
                node.test,
                negate=False,
                ctx=ctx,
            )
        )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if any(isinstance(stmt, ast.Raise) for stmt in node.body):
            ctx = ConditionContext(
                params=self.params,
                rel_path=self.rel_path,
                line=getattr(node, "lineno", None),
                limit=self.limit,
            )
            self.preconditions.extend(
                _preconditions_from_test(
                    node.test,
                    negate=True,
                    ctx=ctx,
                )
            )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = _extract_exception_name(node.exc)
        self.raises.append(
            {
                "exception": exc or "Exception",
                "source": "raise",
                "message_snippet": _extract_message_snippet(node.exc),
                "rel_path": self.rel_path,
                "line": getattr(node, "lineno", None),
            }
        )
        self.generic_visit(node)


def _preconditions_from_test(
    test: ast.AST,
    *,
    negate: bool,
    ctx: ConditionContext,
) -> list[dict[str, object]]:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _preconditions_from_test(test.operand, negate=not negate, ctx=ctx)
    if isinstance(test, ast.Compare):
        condition = _precondition_from_compare(test, negate=negate, ctx=ctx)
        return [condition] if condition is not None else []
    if _is_isinstance_call(test) and isinstance(test, ast.Call):
        condition = _precondition_from_isinstance(test, negate=negate, ctx=ctx)
        return [condition] if condition is not None else []
    if isinstance(test, ast.Name):
        return _truthy_condition(test.id, negate=negate, ctx=ctx)
    if isinstance(test, ast.Attribute):
        return _truthy_condition(test.attr, negate=negate, ctx=ctx)
    return []


def _truthy_condition(
    param: str, *, negate: bool, ctx: ConditionContext
) -> list[dict[str, object]]:
    if param not in ctx.params:
        return []
    return [
        {
            "kind": "truthy" if not negate else "falsey",
            "param": param,
            "source": "assert" if not negate else "guard",
            "rel_path": ctx.rel_path,
            "line": ctx.line,
        }
    ]


def _is_isinstance_call(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_isinstance_call(node.operand)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) >= MIN_ISINSTANCE_ARGS
    )


def _precondition_from_compare(
    compare: ast.Compare,
    *,
    negate: bool,
    ctx: ConditionContext,
) -> dict[str, object] | None:
    if len(compare.ops) != SINGLE_COMPARATOR or len(compare.comparators) != SINGLE_COMPARATOR:
        return None
    left = compare.left
    op = compare.ops[0]
    right = compare.comparators[0]

    len_condition = _len_compare_condition(left, op, right, negate=negate, ctx=ctx)
    if len_condition is not None:
        return len_condition

    param = _param_name(left)
    if param is None or param not in ctx.params:
        return None

    none_condition = _none_compare_condition(op, right, negate=negate, ctx=ctx, param=param)
    if none_condition is not None:
        return none_condition

    return _numeric_compare_condition(op, right, negate=negate, ctx=ctx, param=param)


def _len_compare_condition(
    left: ast.AST,
    op: ast.cmpop,
    right: ast.AST,
    *,
    negate: bool,
    ctx: ConditionContext,
) -> dict[str, object] | None:
    if not _is_len_call(left):
        return None
    target = left.args[0] if isinstance(left, ast.Call) and left.args else None
    param = _param_name(target)
    if param is None or param not in ctx.params:
        return None
    threshold = _constant_int(right)
    if threshold is None:
        return None
    if isinstance(op, (ast.Gt, ast.GtE)) and not negate:
        return _len_condition(param, threshold, ctx.rel_path, ctx.line, source="assert")
    if isinstance(op, (ast.Eq, ast.LtE, ast.Lt)) and negate:
        return _len_condition(param, threshold, ctx.rel_path, ctx.line, source="guard")
    return None


def _none_compare_condition(
    op: ast.cmpop,
    right: ast.AST,
    *,
    negate: bool,
    ctx: ConditionContext,
    param: str,
) -> dict[str, object] | None:
    if not _is_none(right):
        return None
    is_non_null_check = isinstance(op, (ast.IsNot, ast.NotEq)) and not negate
    is_guard = isinstance(op, (ast.Is, ast.Eq)) and negate
    if not (is_non_null_check or is_guard):
        return None
    return {
        "kind": "non_null",
        "param": param,
        "source": "assert" if not negate else "guard",
        "rel_path": ctx.rel_path,
        "line": ctx.line,
    }


def _numeric_compare_condition(
    op: ast.cmpop,
    right: ast.AST,
    *,
    negate: bool,
    ctx: ConditionContext,
    param: str,
) -> dict[str, object] | None:
    if not isinstance(right, ast.Constant) or not isinstance(right.value, (int, float)):
        return None
    value = right.value
    if isinstance(op, (ast.GtE, ast.Gt)) and not negate:
        return {
            "kind": "ge",
            "param": param,
            "value": value,
            "source": "assert",
            "rel_path": ctx.rel_path,
            "line": ctx.line,
        }
    if isinstance(op, (ast.Lt, ast.LtE)) and negate:
        return {
            "kind": "ge",
            "param": param,
            "value": value,
            "source": "guard",
            "rel_path": ctx.rel_path,
            "line": ctx.line,
        }
    return None


def _precondition_from_isinstance(
    call: ast.Call, *, negate: bool, ctx: ConditionContext
) -> dict[str, object] | None:
    if len(call.args) < MIN_ISINSTANCE_ARGS:
        return None
    target = call.args[0]
    param = _param_name(target)
    if param is None:
        return None
    type_expr = call.args[1]
    type_name = _annotation_str(type_expr)
    if type_name is None:
        return None
    if negate:
        return None
    return {
        "kind": "instance_of",
        "param": param,
        "value": type_name,
        "source": "assert",
        "rel_path": ctx.rel_path,
        "line": ctx.line,
    }


def _param_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_len_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"


def _annotation_str(node: ast.AST) -> str | None:
    return safe_unparse(node) or None


def _constant_int(node: ast.AST) -> int | None:
    return literal_int(node)


def _len_condition(
    param: str, threshold: int, rel_path: str, line: int | None, *, source: str
) -> dict[str, object]:
    return {
        "kind": "len_gt",
        "param": param,
        "value": max(0, threshold),
        "source": source,
        "rel_path": rel_path,
        "line": line,
    }


def _is_none(node: ast.AST) -> bool:
    return literal_value(node) is None


def _extract_exception_name(exc: ast.AST | None) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        base = _extract_exception_name(exc.value)
        if base:
            return f"{base}.{exc.attr}"
        return exc.attr
    if isinstance(exc, ast.Call):
        return _extract_exception_name(exc.func)
    return None


def _extract_message_snippet(exc: ast.AST | None) -> str | None:
    if not isinstance(exc, ast.Call):
        return None
    if not exc.args:
        return None
    first_arg = exc.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


def _contract_confidence(*, has_types: bool, has_doc: bool, has_guards: bool) -> float:
    confidence = 0.0
    if has_types:
        confidence += 0.3
    if has_doc:
        confidence += 0.3
    if has_guards:
        confidence += 0.3
    return min(confidence, 1.0)
