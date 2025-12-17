"""File artifact data saver for Hamilton materialization.

This module implements a Hamilton ``DataSaver`` that persists file artifacts
(bytes, strings, or existing files) using atomic write semantics. It is used by
targets that produce non-tabular outputs (exports, indexes, reports) and want
DAG-visible I/O.
"""

from __future__ import annotations

import os
import tempfile
import time
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.outputs import expected_artifacts
from codeintel.build.hamilton.run_records import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.tracking.asset_tracking import AssetRecord

SaveStatus = Literal["succeeded", "skipped", "failed"]

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)


@dataclass(frozen=True)
class FileArtifactSaver(DataSaver):
    """Persist a file artifact for a specific snapshot.

    This adapter:
    - Computes the target input hash (manifest key) from the graph + env.
    - Applies manifest-based skip (authoritative for artifact writes).
    - Writes bytes to a contract-resolved output path using atomic rename.
    - Records the artifact in the asset catalog for observability.
    - Returns a metadata dict (as required by Hamilton's DataSaver API).
    """

    env: BuildEnv
    graph: TargetGraph
    target_name: str
    artifact_name: str

    _hamilton_runtime_types = (BuildEnv, TargetGraph)

    @classmethod
    def name(cls) -> str:
        """Return a stable name for this saver adapter.

        Returns
        -------
        str
            Adapter name used by Hamilton for saver metadata.
        """
        return "codeintel.file_artifact"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Types that this saver can write as a file artifact.
        """
        return [bytes, str, Path]

    @classmethod
    def applies_to(cls, type_: type) -> bool:
        """Return True when this saver can handle the Hamilton node output type.

        Parameters
        ----------
        type_
            Hamilton node output type.

        Returns
        -------
        bool
            True when the saver can persist the output type.
        """
        if type_ in {bytes, str, Path}:
            return True

        origin = get_origin(type_)
        if origin in {types.UnionType, typing.Union}:
            bases = {get_origin(arg) or arg for arg in get_args(type_)}
            if bases.issubset({bytes, str, Path, type(None)}):
                return True

        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, Any]:
        """Save the provided artifact content and return metadata.

        Parameters
        ----------
        data
            Artifact payload. Supported types are bytes, str (encoded as UTF-8),
            or Path (reads bytes from the referenced file).

        Returns
        -------
        dict[str, Any]
            Metadata describing the write, including status and input hash.
        """
        start = time.perf_counter()
        input_hash: str | None = None
        result: dict[str, Any] | None = None

        try:
            target = self.graph.get(self.target_name)
            if target is None:
                result = _failed(
                    artifact_name=self.artifact_name,
                    duration_ms=_duration_ms(start),
                    input_hash="",
                    error=f"Target not found in graph: {self.target_name}",
                )
            else:
                input_hash = compute_input_hash(
                    target=target,
                    snapshot=self.env.snapshot,
                    gateway=self.env.gateway,
                    options_hash=None,
                    manifests=self.env.manifest_index,
                )

                if should_skip_native_target(self.env, target, input_hash):
                    resolved = _resolve_artifact_path(
                        self.env,
                        self.graph,
                        self.target_name,
                        self.artifact_name,
                    )
                    result = _skipped(
                        artifact_name=self.artifact_name,
                        duration_ms=_duration_ms(start),
                        input_hash=input_hash,
                        path=str(resolved) if resolved is not None else None,
                    )
                elif data is None:
                    result = _skipped(
                        artifact_name=self.artifact_name,
                        duration_ms=_duration_ms(start),
                        input_hash=input_hash,
                        path=None,
                    )
                else:
                    output_path = _resolve_artifact_path(
                        self.env, self.graph, self.target_name, self.artifact_name
                    )
                    if output_path is None:
                        return _failed(
                            artifact_name=self.artifact_name,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            error=f"Artifact path could not be resolved: {self.artifact_name}",
                        )

                    # Validate contract if strict mode is enabled
                    ContractEnforcer.validate_artifact_write(self.artifact_name)

                    if isinstance(data, Path) and _same_path(data, output_path):
                        size_bytes = output_path.stat().st_size
                        _record_asset(
                            env=self.env,
                            record=AssetRecord(
                                asset_key=self.artifact_name,
                                asset_type="artifact",
                                repo=self.env.snapshot.repo,
                                commit=self.env.snapshot.commit,
                                owner_target=self.target_name,
                                file_size_bytes=size_bytes,
                                input_hash=input_hash,
                                metadata={
                                    "path": str(output_path),
                                    "size_bytes": size_bytes,
                                },
                            ),
                        )
                        result = _succeeded(
                            artifact_name=self.artifact_name,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            path=str(output_path),
                            size_bytes=size_bytes,
                        )
                    else:
                        content_bytes = _coerce_bytes(data)
                        _atomic_write(output_path, content_bytes)
                        _record_asset(
                            env=self.env,
                            record=AssetRecord(
                                asset_key=self.artifact_name,
                                asset_type="artifact",
                                repo=self.env.snapshot.repo,
                                commit=self.env.snapshot.commit,
                                owner_target=self.target_name,
                                file_size_bytes=len(content_bytes),
                                input_hash=input_hash,
                                metadata={
                                    "path": str(output_path),
                                    "size_bytes": len(content_bytes),
                                },
                            ),
                        )

                        result = _succeeded(
                            artifact_name=self.artifact_name,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            path=str(output_path),
                            size_bytes=len(content_bytes),
                        )

        except _RECOVERABLE_EXCEPTIONS as exc:
            result = _failed(
                artifact_name=self.artifact_name,
                duration_ms=_duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            return _failed(
                artifact_name=self.artifact_name,
                duration_ms=_duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown artifact materialization failure",
            )

        return result


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _succeeded(
    *,
    artifact_name: str,
    duration_ms: float,
    input_hash: str,
    path: str,
    size_bytes: int,
) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "artifact_name": artifact_name,
        "path": path,
        "size_bytes": size_bytes,
        "duration_ms": duration_ms,
        "input_hash": input_hash,
        "error": None,
    }


def _skipped(
    *,
    artifact_name: str,
    duration_ms: float,
    input_hash: str,
    path: str | None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "artifact_name": artifact_name,
        "path": path,
        "size_bytes": None,
        "duration_ms": duration_ms,
        "input_hash": input_hash,
        "error": None,
    }


def _failed(
    *,
    artifact_name: str,
    duration_ms: float,
    input_hash: str,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "artifact_name": artifact_name,
        "path": None,
        "size_bytes": None,
        "duration_ms": duration_ms,
        "input_hash": input_hash,
        "error": error,
    }


def _resolve_artifact_path(
    env: BuildEnv, graph: TargetGraph, target_name: str, artifact_name: str
) -> Path | None:
    target = graph.get(target_name)
    if target is None:
        return None

    artifacts = expected_artifacts(
        target,
        env.snapshot,
        path_formatter={
            "build_dir": str(env.paths.build_dir),
            "scip_dir": str(env.paths.scip_dir),
            "export_dir": str(env.paths.document_output_dir),
            "repo_root": str(env.snapshot.repo_root),
        },
    )
    for art in artifacts:
        if art.name == artifact_name:
            if art.path is None:
                return None
            return Path(art.path)
    return None


def _coerce_bytes(data: object) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, Path):
        return data.read_bytes()
    msg = f"Unsupported artifact payload type: {type(data).__name__}"
    raise TypeError(msg)


def _atomic_write(output_path: Path, content: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )

    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
        temp_path.rename(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _record_asset(
    *,
    env: BuildEnv,
    record: AssetRecord,
) -> None:
    env.gateway.assets.record_asset(record)


__all__ = [
    "FileArtifactSaver",
    "SaveStatus",
]
