"""Inspect extraction step with port injection."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import InspectExtractOptions
from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key
from codeintel.ingestion.compute.base import BaseExtractStep

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

LOG = logging.getLogger(__name__)

PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_MEMBERS_TABLE_KEY = "core.py_inspect_members_static"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_ANNOTATIONS_TABLE_KEY = "core.py_inspect_annotations_kv"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"


@dataclass(frozen=True)
class InspectExtractResult:
    """Result bundle for inspect extraction."""

    result: ExecutionResult
    object_rows: ColumnarRows = field(default_factory=dict)
    member_rows: ColumnarRows = field(default_factory=dict)
    unwrap_rows: ColumnarRows = field(default_factory=dict)
    signature_rows: ColumnarRows = field(default_factory=dict)
    signature_param_rows: ColumnarRows = field(default_factory=dict)
    annotation_rows: ColumnarRows = field(default_factory=dict)
    source_rows: ColumnarRows = field(default_factory=dict)
    object_row_count: int = 0
    member_row_count: int = 0
    unwrap_row_count: int = 0
    signature_row_count: int = 0
    signature_param_row_count: int = 0
    annotation_row_count: int = 0
    source_row_count: int = 0


def _stable_id(*parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _ok_status() -> dict[str, object]:
    return {"ok": True}


def _error_status(exc: Exception) -> dict[str, object]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
    }


def _truncate_repr(value: object, limit: int = 240) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return text[:limit]


def _value_ref(value: object) -> dict[str, object]:
    try:
        repr_text = repr(value)
    except Exception:
        repr_text = "<unreprable>"
    repr_bytes = repr_text.encode("utf-8", errors="replace")
    return {
        "kind": type(value).__name__,
        "type_qualname": type(value).__qualname__,
        "repr_trunc": _truncate_repr(repr_text),
        "repr_len": len(repr_text),
        "repr_sha256": hashlib.sha256(repr_bytes).digest(),
        "is_callable": callable(value),
        "is_descriptor": inspect.isdatadescriptor(value)
        or inspect.ismemberdescriptor(value)
        or inspect.isgetsetdescriptor(value),
        "is_builtin": inspect.isbuiltin(value),
    }


def _object_kind(value: object) -> str:
    if inspect.ismodule(value):
        return "module"
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value):
        return "function"
    if inspect.ismethod(value):
        return "method"
    if inspect.isbuiltin(value):
        return "builtin"
    if inspect.isroutine(value):
        return "routine"
    return type(value).__name__


def _object_module_name(value: object) -> str | None:
    if inspect.ismodule(value):
        return getattr(value, "__name__", None)
    module_name = getattr(value, "__module__", None)
    return module_name if isinstance(module_name, str) else None


def _object_qualname(value: object) -> str | None:
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else None


def _object_name(value: object) -> str | None:
    name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else None


def _object_id(value: object, kind: str) -> str:
    module_name = _object_module_name(value)
    qualname = _object_qualname(value)
    return _stable_id("py_inspect_obj", module_name, qualname, kind)


def _has_signature_override(value: object) -> bool:
    return hasattr(value, "__signature__")


def _has_wrapped(value: object) -> bool:
    return getattr(value, "__wrapped__", None) is not None


def _annotation_payload(
    annotations: dict[str, object],
    *,
    repo: str,
    commit: str,
    mode: str,
    object_id: str,
    eval_str: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in annotations.items():
        if not isinstance(key, str):
            continue
        rows.append(
            {
                "repo": repo,
                "commit": commit,
                "mode": mode,
                "object_id": object_id,
                "eval_str": eval_str,
                "key": key,
                "value": _value_ref(value),
                "status": _ok_status(),
            }
        )
    return rows


def _signature_rows(
    value: object,
    *,
    repo: str,
    commit: str,
    mode: str,
    object_id: str,
    follow_wrapped: bool,
    eval_str: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    signature_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    signature_id = _stable_id("py_inspect_sig", object_id, follow_wrapped, eval_str)
    try:
        signature = inspect.signature(value, follow_wrapped=follow_wrapped)
    except (TypeError, ValueError) as exc:
        signature_rows.append(
            {
                "repo": repo,
                "commit": commit,
                "mode": mode,
                "signature_id": signature_id,
                "object_id": object_id,
                "variant": "primary",
                "follow_wrapped": follow_wrapped,
                "eval_str": eval_str,
                "effective_object_id": object_id,
                "sig_text": None,
                "sig_format": "inspect",
                "return_annotation": None,
                "has_varargs": None,
                "has_varkw": None,
                "status": _error_status(exc),
            }
        )
        return signature_rows, param_rows

    has_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values()
    )
    has_varkw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    return_annotation = None
    if signature.return_annotation is not inspect.Signature.empty:
        return_annotation = _value_ref(signature.return_annotation)
    signature_rows.append(
        {
            "repo": repo,
            "commit": commit,
            "mode": mode,
            "signature_id": signature_id,
            "object_id": object_id,
            "variant": "primary",
            "follow_wrapped": follow_wrapped,
            "eval_str": eval_str,
            "effective_object_id": object_id,
            "sig_text": str(signature),
            "sig_format": "inspect",
            "return_annotation": return_annotation,
            "has_varargs": has_varargs,
            "has_varkw": has_varkw,
            "status": _ok_status(),
        }
    )
    for idx, param in enumerate(signature.parameters.values()):
        default_present = param.default is not inspect.Parameter.empty
        annotation_present = param.annotation is not inspect.Parameter.empty
        param_rows.append(
            {
                "repo": repo,
                "commit": commit,
                "mode": mode,
                "signature_id": signature_id,
                "param_index": idx,
                "name": param.name,
                "kind": param.kind.name,
                "default_present": default_present,
                "default_value": _value_ref(param.default) if default_present else None,
                "annotation_present": annotation_present,
                "annotation_value": _value_ref(param.annotation) if annotation_present else None,
                "status": _ok_status(),
            }
        )
    return signature_rows, param_rows


def _unwrap_hops(
    value: object,
    *,
    repo: str,
    commit: str,
    mode: str,
    root_object_id: str,
    follow_wrapped: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = value
    stop_reason: str | None = None
    max_hops = 20
    for hop in range(max_hops):
        obj_id = _object_id(current, _object_kind(current))
        has_wrapped = _has_wrapped(current)
        rows.append(
            {
                "repo": repo,
                "commit": commit,
                "mode": mode,
                "root_object_id": root_object_id,
                "hop": hop,
                "object_id": obj_id,
                "has_wrapped": has_wrapped,
                "has_signature_override": _has_signature_override(current),
                "stop_reason": stop_reason,
                "status": _ok_status(),
            }
        )
        if not follow_wrapped:
            stop_reason = "follow_wrapped_disabled"
            break
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            stop_reason = "no_wrapped"
            break
        current = wrapped
    else:
        stop_reason = "max_hops"
    if rows:
        rows[-1]["stop_reason"] = stop_reason
    return rows


def _source_row(
    value: object,
    *,
    repo: str,
    commit: str,
    mode: str,
    object_id: str,
) -> dict[str, object] | None:
    try:
        source_lines, start_line = inspect.getsourcelines(value)
        file_name = inspect.getsourcefile(value)
    except (OSError, TypeError) as exc:
        return {
            "repo": repo,
            "commit": commit,
            "mode": mode,
            "object_id": object_id,
            "file_name": None,
            "start_line": None,
            "line_count": None,
            "source_sha256": None,
            "source_preview": None,
            "status": _error_status(exc),
        }
    source_text = "".join(source_lines)
    source_bytes = source_text.encode("utf-8", errors="replace")
    preview = source_text[:200]
    return {
        "repo": repo,
        "commit": commit,
        "mode": mode,
        "object_id": object_id,
        "file_name": file_name,
        "start_line": max(start_line - 1, 0),
        "line_count": len(source_lines),
        "source_sha256": hashlib.sha256(source_bytes).digest(),
        "source_preview": preview,
        "status": _ok_status(),
    }


@contextlib.contextmanager
def _sys_path_prefix(root: Path) -> Iterable[None]:
    root_str = str(root)
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(root_str)


def _repo_root_for_module(module: ModuleRecord) -> Path:
    root = module.file_path
    for _ in Path(module.rel_path).parts:
        root = root.parent
    return root


class InspectExtractStep(BaseExtractStep):
    """Inspect extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: InspectExtractOptions | None = None,
    ) -> None:
        super().__init__(discovery=discovery)
        self._options = options or InspectExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> InspectExtractResult:
        if not self._options.enable:
            return InspectExtractResult(
                result=ExecutionResult.skip("Inspect extraction disabled by options")
            )
        allowlist = set(self._options.module_allowlist)
        if not allowlist:
            return InspectExtractResult(
                result=ExecutionResult.skip("Inspect extraction disabled (no allowlist)")
            )
        try:
            object_buffer = columnar_buffer_for_table_key(PY_INSPECT_OBJECTS_TABLE_KEY)
            member_buffer = columnar_buffer_for_table_key(PY_INSPECT_MEMBERS_TABLE_KEY)
            unwrap_buffer = columnar_buffer_for_table_key(PY_INSPECT_UNWRAP_TABLE_KEY)
            signature_buffer = columnar_buffer_for_table_key(PY_INSPECT_SIGNATURES_TABLE_KEY)
            signature_param_buffer = columnar_buffer_for_table_key(
                PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY
            )
            annotation_buffer = columnar_buffer_for_table_key(PY_INSPECT_ANNOTATIONS_TABLE_KEY)
            source_buffer = columnar_buffer_for_table_key(PY_INSPECT_SOURCE_TABLE_KEY)
        except (KeyError, RuntimeError) as exc:
            return InspectExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []
        seen_objects: set[str] = set()
        mode = "allowlist"
        object_limit = self._options.max_objects
        object_count = 0

        for module in modules:
            if object_count >= object_limit:
                break
            if module.module_name not in allowlist:
                continue
            repo_root = _repo_root_for_module(module)
            with _sys_path_prefix(repo_root):
                try:
                    loaded_module = importlib.import_module(module.module_name)
                except ImportError as exc:
                    message = f"Inspect import failed for {module.module_name}: {exc}"
                    warnings.append(message)
                    LOG.warning("%s", message)
                    continue

            module_kind = _object_kind(loaded_module)
            module_object_id = _object_id(loaded_module, module_kind)
            if module_object_id not in seen_objects:
                seen_objects.add(module_object_id)
                object_buffer.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "mode": mode,
                        "object_id": module_object_id,
                        "object_addr": id(loaded_module),
                        "kind": module_kind,
                        "module_name": _object_module_name(loaded_module),
                        "qualname": _object_qualname(loaded_module),
                        "name": _object_name(loaded_module),
                        "type_qualname": type(loaded_module).__qualname__,
                        "is_builtin": inspect.isbuiltin(loaded_module),
                        "is_callable": callable(loaded_module),
                        "is_descriptor": inspect.isdatadescriptor(loaded_module),
                        "has_wrapped": _has_wrapped(loaded_module),
                        "has_signature_override": _has_signature_override(loaded_module),
                        "has_annotations": bool(getattr(loaded_module, "__annotations__", None)),
                        "status": _ok_status(),
                    }
                )
                object_count += 1
                if object_count >= object_limit:
                    warnings.append("Inspect object limit reached")
                    break

            try:
                members = inspect.getmembers_static(loaded_module)
            except (AttributeError, TypeError) as exc:
                warnings.append(f"Inspect members failed for {module.module_name}: {exc}")
                continue
            for attr_name, value in members:
                if object_count >= object_limit:
                    break
                value_kind = _object_kind(value)
                value_object_id = None
                if inspect.isroutine(value) or inspect.isclass(value) or inspect.ismodule(value):
                    value_object_id = _object_id(value, value_kind)
                    if value_object_id not in seen_objects:
                        seen_objects.add(value_object_id)
                        object_buffer.append(
                            {
                                "repo": repo,
                                "commit": commit,
                                "mode": mode,
                                "object_id": value_object_id,
                                "object_addr": id(value),
                                "kind": value_kind,
                                "module_name": _object_module_name(value),
                                "qualname": _object_qualname(value),
                                "name": _object_name(value),
                                "type_qualname": type(value).__qualname__,
                                "is_builtin": inspect.isbuiltin(value),
                                "is_callable": callable(value),
                                "is_descriptor": inspect.isdatadescriptor(value),
                                "has_wrapped": _has_wrapped(value),
                                "has_signature_override": _has_signature_override(value),
                                "has_annotations": bool(getattr(value, "__annotations__", None)),
                                "status": _ok_status(),
                            }
                        )
                        object_count += 1
                member_buffer.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "mode": mode,
                        "owner_object_id": module_object_id,
                        "owner_kind": module_kind,
                        "attr_name": attr_name,
                        "value_kind": value_kind,
                        "value_object_id": value_object_id,
                        "value_ref": _value_ref(value),
                        "desc_kind": type(value).__name__,
                        "desc_is_data": inspect.isdatadescriptor(value),
                        "desc_is_methoddesc": inspect.ismethoddescriptor(value),
                        "desc_is_getset": inspect.isgetsetdescriptor(value),
                        "desc_is_member": inspect.ismemberdescriptor(value),
                        "status": _ok_status(),
                    }
                )
                if not callable(value) or value_object_id is None:
                    continue
                unwrap_rows = _unwrap_hops(
                    value,
                    repo=repo,
                    commit=commit,
                    mode=mode,
                    root_object_id=value_object_id,
                    follow_wrapped=self._options.follow_wrapped,
                )
                for row in unwrap_rows:
                    unwrap_buffer.append(row)
                sig_rows, param_rows = _signature_rows(
                    value,
                    repo=repo,
                    commit=commit,
                    mode=mode,
                    object_id=value_object_id,
                    follow_wrapped=self._options.follow_wrapped,
                    eval_str=self._options.eval_str,
                )
                for row in sig_rows:
                    signature_buffer.append(row)
                for row in param_rows:
                    signature_param_buffer.append(row)
                try:
                    annotations = inspect.get_annotations(value, eval_str=self._options.eval_str)
                except (TypeError, ValueError, NameError) as exc:
                    warnings.append(f"Inspect annotations failed for {attr_name}: {exc}")
                    annotations = {}
                for row in _annotation_payload(
                    annotations,
                    repo=repo,
                    commit=commit,
                    mode=mode,
                    object_id=value_object_id,
                    eval_str=self._options.eval_str,
                ):
                    annotation_buffer.append(row)
                source_row = _source_row(
                    value,
                    repo=repo,
                    commit=commit,
                    mode=mode,
                    object_id=value_object_id,
                )
                if source_row is not None:
                    source_buffer.append(source_row)

        return InspectExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            object_rows=object_buffer.data,
            member_rows=member_buffer.data,
            unwrap_rows=unwrap_buffer.data,
            signature_rows=signature_buffer.data,
            signature_param_rows=signature_param_buffer.data,
            annotation_rows=annotation_buffer.data,
            source_rows=source_buffer.data,
            object_row_count=object_buffer.row_count,
            member_row_count=member_buffer.row_count,
            unwrap_row_count=unwrap_buffer.row_count,
            signature_row_count=signature_buffer.row_count,
            signature_param_row_count=signature_param_buffer.row_count,
            annotation_row_count=annotation_buffer.row_count,
            source_row_count=source_buffer.row_count,
        )


__all__ = ["InspectExtractResult", "InspectExtractStep"]
