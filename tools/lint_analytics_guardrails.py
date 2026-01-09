"""Combined guardrails for analytics helper patterns."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ast_grep_py import SgNode, SgRoot

from tools.lint_file_utils import find_literal_candidates, list_python_files

_ANALYTICS_ROOT = "src/codeintel/build/analytics"
_ALLOWLIST_ITER_ROWS: dict[str, frozenset[str]] = {
    "src/codeintel/build/analytics/cfg_dfg/cfg_core.py": frozenset({"load_cfg_blocks"}),
    "src/codeintel/build/analytics/cfg_dfg/dfg_core.py": frozenset({"load_dfg_edges"}),
    "src/codeintel/build/analytics/cfg_dfg/helpers.py": frozenset({"load_function_metadata"}),
    "src/codeintel/build/analytics/compute/data_models/usage.py": frozenset({"_rows_for_snapshot"}),
    "src/codeintel/build/analytics/compute/dependencies/compute.py": frozenset(
        {"_rows_for_snapshot"}
    ),
    "src/codeintel/build/analytics/data_models/core.py": frozenset({"_rows_for_snapshot"}),
    "src/codeintel/build/analytics/entrypoints/core.py": frozenset({"_rows_for_snapshot"}),
    "src/codeintel/build/analytics/functions/function_contracts.py": frozenset(
        {"_doc_map_from_frame", "_type_map_from_frame"}
    ),
    "src/codeintel/build/analytics/functions/function_effects.py": frozenset(
        {
            "_call_graph_from_frames",
            "_ensure_call_graph_nodes",
            "_unresolved_call_counts_from_frame",
        }
    ),
    "src/codeintel/build/analytics/functions/metrics.py": frozenset({"_load_goids_from_frame"}),
    "src/codeintel/build/analytics/graphs/config_data_flow.py": frozenset(
        {"_config_reference_rows_from_tabular", "_entrypoint_rows_from_tabular"}
    ),
    "src/codeintel/build/analytics/graphs/config_graph_metrics.py": frozenset(
        {"_rows_from_tabular"}
    ),
    "src/codeintel/build/analytics/graphs/config_references.py": frozenset(
        {"_config_entries_from_table", "_modules_by_path_from_table"}
    ),
    "src/codeintel/build/analytics/py_cpg_quality_report.py": frozenset(
        {
            "_anchor_rate",
            "_count_from_table",
            "_inspect_anchor_ids",
            "_reader_rows",
            "_symbol_edge_counts",
        }
    ),
    "src/codeintel/build/analytics/scip_diagnostics_rollups.py": frozenset(
        {"_aggregate_rollup_rows"}
    ),
    "src/codeintel/build/analytics/semantic_roles/core.py": frozenset(
        {
            "_contracts_from_frame",
            "_effects_from_frame",
            "_function_rows_from_frame",
            "_graph_metrics_from_frame",
            "_module_meta_from_frame",
        }
    ),
    "src/codeintel/build/analytics/subsystems/affinity.py": frozenset(
        {
            "_config_edge_tuples",
            "_import_edge_tuples",
            "_symbol_edge_tuples",
            "load_modules_from_frame",
        }
    ),
    "src/codeintel/build/analytics/subsystems/cache.py": frozenset(
        {"build_subsystem_profile_cache_rows"}
    ),
    "src/codeintel/build/analytics/utilities/catalogs.py": frozenset({"_iter_rows_from_source"}),
    "src/codeintel/build/analytics/utilities/datasets.py": frozenset({"_validated_records"}),
}
_ALLOWLIST_NO_DECODER: frozenset[str] = frozenset(
    {
        "src/codeintel/build/analytics/cfg_dfg/helpers.py",
    }
)
_DECODER_PREFIXES: tuple[str, ...] = ("_list_values", "_flatten_", "_normalize_")
_FINALIZE_CALLS: frozenset[str] = frozenset(
    {
        "finalize_analytics_table",
        "finalize_analytics_result",
        "finalize_table",
        "_finalize_rows_for_parquet",
    }
)

_ITER_ROWS_MESSAGE = (
    "iter_rows usage in analytics module without allowlist entry; "
    "use Plan.aggregate or add to allowlist for AST/graph boundary."
)
_FINALIZE_MESSAGE = (
    "analytics write_dataset call without finalize_analytics_* in the same function."
)
_ROWSET_ORDER_MESSAGE = "Plan.aggregate(list) without order_by in rowset helper."
_ROWSET_DECODER_MESSAGE = "List-aggregate rowset missing list-decoding helper."
_LIST_LITERAL_PATTERN = re.compile(r"(['\"])list\1")


@dataclass(frozen=True)
class Violation:
    """Single guardrail violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


@dataclass(frozen=True)
class AnalyticsScan:
    """Container for analytics guardrail findings."""

    iter_rows: list[Violation]
    finalize_writes: list[Violation]
    rowset_guardrails: list[Violation]


def _candidate_paths(root: Path) -> set[Path]:
    analytics_candidates = find_literal_candidates(
        root,
        patterns=("iter_rows", "aggregate"),
        include_globs=(f"{_ANALYTICS_ROOT}/**/*.py",),
    )
    finalize_candidates = find_literal_candidates(
        root,
        patterns=("write_dataset",),
        include_globs=("src/**/*.py",),
    )
    return analytics_candidates | finalize_candidates


def _iter_python_files(root: Path) -> Iterable[Path]:
    yield from list_python_files(root, ("src",))


def _parse_root(path: Path) -> SgRoot | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return SgRoot(source, "python")
    except (RuntimeError, ValueError):
        return None


def _rel_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _literal_string(text: str) -> str | None:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _function_defs(root: SgRoot) -> list[SgNode]:
    tree = root.root()
    functions: list[SgNode] = []
    for child in tree.children():
        if child.kind() == "function_definition":
            functions.append(child)
            continue
        if child.kind() == "decorated_definition":
            match = child.find(pattern="def $NAME($$$ARGS): $$$BODY")
            if match is not None:
                functions.append(match)
    return functions


def _function_name(node: SgNode) -> str | None:
    name_node = node.field("name")
    if name_node is None:
        return None
    name = name_node.text()
    return name if isinstance(name, str) else None


def _function_has_call(node: SgNode, name: str) -> bool:
    if node.find(pattern=f"{name}($$$ARGS)") is not None:
        return True
    return node.find(pattern=f"$OBJ.{name}($$$ARGS)") is not None


def _find_calls(node: SgNode, name: str) -> list[SgNode]:
    direct = node.find_all(pattern=f"{name}($$$ARGS)")
    attr = node.find_all(pattern=f"$OBJ.{name}($$$ARGS)")
    return list(direct) + list(attr)


def _call_has_analytics_table_key(call_node: SgNode) -> bool:
    matches = call_node.find_all(pattern="table_key=$KEY")
    for match in matches:
        key_node = match.get_match("KEY")
        if key_node is None:
            continue
        key_value = _literal_string(key_node.text())
        if key_value is not None and key_value.startswith("analytics."):
            return True
    return False


def _has_decoder_helper(functions: list[SgNode]) -> bool:
    for node in functions:
        name = _function_name(node)
        if name is None:
            continue
        if any(name.startswith(prefix) for prefix in _DECODER_PREFIXES):
            return True
    return False


def _has_list_aggregate(node: SgNode) -> bool:
    calls = node.find_all(pattern="$OBJ.aggregate($$$ARGS)")
    return any(_LIST_LITERAL_PATTERN.search(call.text()) for call in calls)


def _has_order_by(node: SgNode) -> bool:
    return node.find(pattern="$OBJ.order_by($$$ARGS)") is not None


def _iter_rows_violations(
    *,
    path: Path,
    rel: str,
    functions: list[SgNode],
) -> list[Violation]:
    violations: list[Violation] = []
    allowlist = _ALLOWLIST_ITER_ROWS.get(rel, frozenset())
    for node in functions:
        if not _function_has_call(node, "iter_rows"):
            continue
        name = _function_name(node)
        if name is None or name in allowlist:
            continue
        violations.append(
            Violation(
                path=path,
                lineno=node.range().start.line + 1,
                message=_ITER_ROWS_MESSAGE,
            )
        )
    return violations


def _rowset_guardrail_violations(
    *,
    path: Path,
    rel: str,
    functions: list[SgNode],
) -> list[Violation]:
    violations: list[Violation] = []
    has_list_aggregate = False
    for node in functions:
        if not _has_list_aggregate(node):
            continue
        has_list_aggregate = True
        if _has_order_by(node):
            continue
        violations.append(
            Violation(
                path=path,
                lineno=node.range().start.line + 1,
                message=_ROWSET_ORDER_MESSAGE,
            )
        )
    if (
        has_list_aggregate
        and rel not in _ALLOWLIST_NO_DECODER
        and not _has_decoder_helper(functions)
    ):
        violations.append(
            Violation(
                path=path,
                lineno=1,
                message=_ROWSET_DECODER_MESSAGE,
            )
        )
    return violations


def _finalize_write_violations(*, path: Path, functions: list[SgNode]) -> list[Violation]:
    violations: list[Violation] = []
    for node in functions:
        write_calls = _find_calls(node, "write_dataset")
        if not write_calls:
            continue
        has_finalize = any(_function_has_call(node, name) for name in _FINALIZE_CALLS)
        if has_finalize:
            continue
        for call in write_calls:
            if not _call_has_analytics_table_key(call):
                continue
            violations.append(
                Violation(
                    path=path,
                    lineno=call.range().start.line + 1,
                    message=_FINALIZE_MESSAGE,
                )
            )
    return violations


def scan_analytics(repo_root: Path) -> AnalyticsScan:
    """Run analytics guardrails in a single pass.

    Parameters
    ----------
    repo_root
        Repository root for path resolution.

    Returns
    -------
    AnalyticsScan
        Aggregated findings for analytics guardrails.
    """
    candidate_paths = _candidate_paths(repo_root)
    if not candidate_paths:
        return AnalyticsScan(iter_rows=[], finalize_writes=[], rowset_guardrails=[])
    iter_rows_violations: list[Violation] = []
    finalize_violations: list[Violation] = []
    rowset_violations: list[Violation] = []
    for path in _iter_python_files(repo_root):
        if path not in candidate_paths:
            continue
        parsed_root = _parse_root(path)
        if parsed_root is None:
            continue
        rel = _rel_path(path, root=repo_root)
        functions = _function_defs(parsed_root)
        is_analytics_file = rel.startswith(f"{_ANALYTICS_ROOT}/")
        if is_analytics_file:
            iter_rows_violations.extend(
                _iter_rows_violations(path=path, rel=rel, functions=functions)
            )
            rowset_violations.extend(
                _rowset_guardrail_violations(path=path, rel=rel, functions=functions)
            )
        finalize_violations.extend(
            _finalize_write_violations(path=path, functions=functions)
        )
    return AnalyticsScan(
        iter_rows=iter_rows_violations,
        finalize_writes=finalize_violations,
        rowset_guardrails=rowset_violations,
    )


def _emit_violations(violations: list[Violation], *, root: Path) -> None:
    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    sys.stderr.write("\n".join(output_lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run analytics guardrails with shared scanning.

    Returns
    -------
    int
        Exit code (0 on success, 1 on violations).
    """
    args = list(argv) if argv is not None else []
    repo_root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    findings = scan_analytics(repo_root)
    had_violations = False
    if findings.rowset_guardrails:
        _emit_violations(findings.rowset_guardrails, root=repo_root)
        sys.stderr.write(
            f"{len(findings.rowset_guardrails)} analytics rowset guardrail violation(s).\n"
        )
        had_violations = True
    if findings.finalize_writes:
        _emit_violations(findings.finalize_writes, root=repo_root)
        sys.stderr.write(
            f"{len(findings.finalize_writes)} analytics finalize write violation(s).\n"
        )
        had_violations = True
    if findings.iter_rows:
        _emit_violations(findings.iter_rows, root=repo_root)
        sys.stderr.write(
            f"{len(findings.iter_rows)} analytics iter_rows guardrail violation(s).\n"
        )
        had_violations = True
    return 1 if had_violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
