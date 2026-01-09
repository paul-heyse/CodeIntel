"""Guardrail for iter_rows usage in analytics modules."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCAN_ROOTS: tuple[str, ...] = ("src/codeintel/build/analytics",)
_ALLOWLIST: dict[str, frozenset[str]] = {
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


@dataclass(frozen=True)
class Violation:
    """Single guardrail violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


def _iter_python_files(root: Path) -> Iterable[Path]:
    for rel_root in _SCAN_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_uses_iter_rows(node: ast.FunctionDef) -> bool:
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and _call_name(call) == "iter_rows":
            return True
    return False


def _allowlisted(path: Path, func_name: str, *, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    allowed = _ALLOWLIST.get(rel)
    if not allowed:
        return False
    return func_name in allowed


def _lint_file(path: Path, *, root: Path) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not isinstance(tree, ast.Module):
        return []
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _function_uses_iter_rows(node):
            continue
        if _allowlisted(path, node.name, root=root):
            continue
        violations.append(
            Violation(
                path=path,
                lineno=node.lineno,
                message=(
                    "iter_rows usage in analytics module without allowlist entry; "
                    "use Plan.aggregate or add to allowlist for AST/graph boundary."
                ),
            )
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analytics iter_rows guardrail.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    violations: list[Violation] = []
    for path in _iter_python_files(root):
        violations.extend(_lint_file(path, root=root))

    if not violations:
        return 0

    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    output_lines.append(f"{len(violations)} analytics iter_rows guardrail violation(s).")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
