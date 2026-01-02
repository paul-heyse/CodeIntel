from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

try:
    import libcst as cst  # type: ignore
    from libcst.metadata import MetadataWrapper, ByteSpanPositionProvider  # type: ignore
except Exception as e:  # pragma: no cover
    cst = None
    MetadataWrapper = None
    ByteSpanPositionProvider = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class DefRec:
    path: str                 # repo-relative posix
    kind: str                 # function|class|method
    name: str
    qname: str
    start_byte: int
    end_byte: int
    container_qname: Optional[str]


class _DefCollector(cst.CSTVisitor):  # type: ignore[misc]
    METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

    def __init__(self, module_qname: str):
        self.module_qname = module_qname
        self.stack: List[Tuple[str, str]] = []  # (kind,name)
        # (kind,name,qname,start,end,container_qname)
        self._defs: List[Tuple[str, str, str, int, int, Optional[str]]] = []

    def _current_container_qname(self) -> Optional[str]:
        if not self.stack:
            return self.module_qname
        parts = [self.module_qname] + [n for _, n in self.stack]
        return ".".join(parts)

    def _push(self, kind: str, name: str) -> None:
        self.stack.append((kind, name))

    def _pop(self) -> None:
        self.stack.pop()

    def _make_qname(self, name: str) -> str:
        parts = [self.module_qname] + [n for _, n in self.stack] + [name]
        return ".".join([p for p in parts if p])

    def visit_ClassDef(self, node: "cst.ClassDef") -> Optional[bool]:
        span = self.get_metadata(ByteSpanPositionProvider, node)
        qname = self._make_qname(node.name.value)
        container = self._current_container_qname()
        self._defs.append(("class", node.name.value, qname, int(span.start), int(span.end), container))
        self._push("class", node.name.value)
        return True

    def leave_ClassDef(self, node: "cst.ClassDef") -> None:
        self._pop()

    def visit_FunctionDef(self, node: "cst.FunctionDef") -> Optional[bool]:
        span = self.get_metadata(ByteSpanPositionProvider, node)
        kind = "method" if any(k == "class" for k, _ in self.stack) else "function"
        qname = self._make_qname(node.name.value)
        container = self._current_container_qname()
        self._defs.append((kind, node.name.value, qname, int(span.start), int(span.end), container))
        self._push("function", node.name.value)
        return True

    def leave_FunctionDef(self, node: "cst.FunctionDef") -> None:
        self._pop()

    @property
    def defs(self) -> List[Tuple[str, str, str, int, int, Optional[str]]]:
        return self._defs


def path_to_module_qname(rel_path: str) -> str:
    """Convert repo-relative python path to a module qualifier.

    Examples:
      pkg/foo.py -> pkg.foo
      pkg/__init__.py -> pkg
    """
    p = PurePosixPath(rel_path)
    if p.suffix == ".py":
        p = p.with_suffix("")
    parts = list(p.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@dataclass
class DefIndex:
    module_qname: str
    path: str
    defs: List[DefRec]
    name_map: Dict[str, List[DefRec]]

    def enclosing_def(self, byte_offset: int) -> Optional[DefRec]:
        best: Optional[DefRec] = None
        for d in self.defs:
            if d.start_byte <= byte_offset < d.end_byte:
                if best is None or (d.end_byte - d.start_byte) < (best.end_byte - best.start_byte):
                    best = d
        return best

    def by_name(self, name: str) -> List[DefRec]:
        return self.name_map.get(name, [])


def build_def_index(rel_path: str, src_bytes: bytes) -> DefIndex:
    if cst is None:
        raise RuntimeError(f"libcst is not importable. Install libcst. Import error: {_IMPORT_ERROR}")

    module_qname = path_to_module_qname(rel_path)
    module = cst.parse_module(src_bytes)
    wrapper = MetadataWrapper(module)
    collector = _DefCollector(module_qname)
    wrapper.visit(collector)

    defs: List[DefRec] = []
    for kind, name, qname, start, end, container in collector.defs:
        defs.append(DefRec(
            path=rel_path,
            kind=kind,
            name=name,
            qname=qname,
            start_byte=start,
            end_byte=end,
            container_qname=container,
        ))

    name_map: Dict[str, List[DefRec]] = {}
    for d in defs:
        name_map.setdefault(d.name, []).append(d)

    for k in name_map:
        name_map[k].sort(key=lambda r: (r.start_byte, r.end_byte, r.qname))
    defs.sort(key=lambda r: (r.start_byte, r.end_byte, r.qname))
    return DefIndex(module_qname=module_qname, path=rel_path, defs=defs, name_map=name_map)
