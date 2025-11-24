"""LibCST-based call edge collection."""

from __future__ import annotations

import libcst as cst
from libcst import MetadataWrapper, metadata

from codeintel.graphs.call_context import EdgeResolutionContext
from codeintel.graphs.call_resolution import build_evidence, resolve_callee, resolve_via_scip
from codeintel.models.rows import CallGraphEdgeRow

FUNCTION_NODE_TYPES = (cst.FunctionDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef))


class _FileCallGraphVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, context: EdgeResolutionContext) -> None:
        self.rel_path = rel_path
        self.context = context
        self.current_function_goid: int | None = None
        self.edges: list[CallGraphEdgeRow] = []

    def _pos(self, node: cst.CSTNode) -> tuple[metadata.CodePosition, metadata.CodePosition] | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        return pos.start, pos.end

    def visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, FUNCTION_NODE_TYPES):
            span = self._pos(node)
            if span is None:
                return True
            start, end = span
            self.current_function_goid = self.context.function_index.lookup(
                self.rel_path, start.line, end.line
            )
            return True

        if isinstance(node, cst.Call):
            self._handle_call(node)
        return True

    def leave(self, node: cst.CSTNode) -> None:
        if isinstance(node, FUNCTION_NODE_TYPES):
            self.current_function_goid = None

    def _handle_call(self, node: cst.Call) -> None:
        if self.current_function_goid is None:
            spans = self.context.function_index.spans_for_path(self.rel_path)
            if spans:
                self.current_function_goid = spans[0].goid
        if self.current_function_goid is None:
            return

        span = self._pos(node)
        if span is None:
            return
        start, _end = span

        callee_name, attr_chain = _extract_callee(node.func)
        resolution = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if resolution.callee_goid is None and scip_paths:
            resolution = resolve_via_scip(scip_paths, self.context.def_goids_by_path)

        evidence = build_evidence(callee_name, attr_chain, resolution, scip_paths)
        self.edges.append(
            CallGraphEdgeRow(
                repo=self.context.repo,
                commit=self.context.commit,
                caller_goid_h128=self.current_function_goid,
                callee_goid_h128=resolution.callee_goid,
                callsite_path=self.rel_path,
                callsite_line=start.line,
                callsite_col=start.column,
                language="python",
                kind="direct" if resolution.callee_goid is not None else "unresolved",
                resolved_via=resolution.resolved_via,
                confidence=resolution.confidence,
                evidence_json=evidence,
            )
        )


def _extract_callee(expr: cst.CSTNode) -> tuple[str, list[str]]:
    if isinstance(expr, cst.Name):
        return expr.value, [expr.value]
    if isinstance(expr, cst.Attribute):
        names: list[str] = []
        attr_node = expr
        while isinstance(attr_node, cst.Attribute):
            names.append(attr_node.attr.value)
            base = attr_node.value
            if isinstance(base, cst.Attribute):
                attr_node = base
                continue
            if isinstance(base, cst.Name):
                names.append(base.value)
            break
        names.reverse()
        return names[-1], names
    return "", []


def collect_edges_cst(
    rel_path: str,
    module: cst.Module,
    context: EdgeResolutionContext,
) -> list[CallGraphEdgeRow]:
    """
    Collect call edges via LibCST for a single module.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected edges for the file.
    """
    visitor = _FileCallGraphVisitor(rel_path=rel_path, context=context)
    wrapper = MetadataWrapper(module)
    wrapper.resolve(metadata.PositionProvider)
    wrapper.visit(visitor)
    return visitor.edges
