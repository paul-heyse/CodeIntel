"""File artifact data saver for Hamilton materialization.

This module implements a Hamilton ``DataSaver`` that persists file artifacts
(bytes, strings, or existing files) using atomic write semantics. It is used by
targets that produce non-tabular outputs (exports, indexes, reports) and want
DAG-visible I/O.
"""

from __future__ import annotations

import os
import tempfile
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, get_args, get_origin

import pyarrow as pa
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.hamilton.materializers.path_templates import (
    default_formatter,
    format_path_template,
)
from codeintel.core.execution.materialization import (
    failed_artifact_result,
    succeeded_artifact_result,
)
from codeintel.storage.duckdb_types import DuckDBRelation

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class ArtifactWritePlan:
    """Deferred artifact write plan executed by FileArtifactSaver.

    This allows compute nodes to remain I/O free while still enabling
    streaming writers (bounded memory) for large artifacts.

    Parameters
    ----------
    write_to
        Callable that writes the artifact payload to the provided path and returns
        the number of bytes written.
    """

    write_to: Callable[[Path], int]


@dataclass(frozen=True)
class FileArtifactSaver(DataSaver):
    """Persist a file artifact for a specific snapshot.

    This adapter:
    - Resolves target metadata from the DAG catalog.
    - Writes bytes to a contract-resolved output path using atomic rename.
    - Returns metadata convertible to a MaterializationResult describing the write outcome.
    """

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    artifact_name: str
    path_template: str | None = None
    output_role: Literal["contract", "internal"] | None = None

    _hamilton_runtime_types = (BuildEnv, DagCatalog)

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
        return [
            ArtifactWritePlan,
            bytes,
            str,
            Path,
            DuckDBRelation,
            pa.Table,
            pa.RecordBatchReader,
        ]

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
        if type_ in {bytes, str, Path, DuckDBRelation, pa.Table, pa.RecordBatchReader}:
            return True
        if type_ is ArtifactWritePlan:
            return True

        origin = get_origin(type_)
        if origin in {types.UnionType, typing.Union}:
            bases = {get_origin(arg) or arg for arg in get_args(type_)}
            if bases.issubset(
                {
                    ArtifactWritePlan,
                    bytes,
                    str,
                    Path,
                    DuckDBRelation,
                    pa.Table,
                    pa.RecordBatchReader,
                    type(None),
                }
            ):
                return True

        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, object]:
        """Save the provided artifact content and return metadata.

        Parameters
        ----------
        data
            Artifact payload. Supported types are bytes, str (encoded as UTF-8),
            Path (reads bytes from the referenced file), DuckDB relations, or
            Arrow tables/readers.

        Returns
        -------
        dict[str, object]
            Metadata describing the write and materialization outcome.
        """
        start = perf_counter()
        input_hash: str | None = None
        result: MaterializationResult | None = None

        try:
            prepared = resolve_materialization_context(
                env=self.env,
                catalog=self.catalog,
                target_name=self.target_name,
            )
            if isinstance(prepared, MaterializationContextError):
                result = failed_artifact_result(
                    artifact_name=self.artifact_name,
                    duration_ms=duration_ms(start),
                    input_hash=prepared.input_hash or "",
                    error=prepared.message,
                )
            else:
                context = prepared
                input_hash = context.input_hash
                if data is None:
                    result = failed_artifact_result(
                        artifact_name=self.artifact_name,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        error="Expected artifact payload but received None",
                    )
                else:
                    output_path = _resolve_artifact_path(
                        self.env,
                        self.target_name,
                        self.artifact_name,
                        path_template=self.path_template,
                    )
                    if output_path is None:
                        result = failed_artifact_result(
                            artifact_name=self.artifact_name,
                            duration_ms=duration_ms(start),
                            input_hash=input_hash or "",
                            error=f"Artifact path could not be resolved: {self.artifact_name}",
                        )
                    else:
                        # Validate contract if strict mode is enabled
                        ContractEnforcer.validate_artifact_write(self.artifact_name)

                        if isinstance(data, ArtifactWritePlan):
                            size_bytes = _atomic_write_via_plan(output_path, data)
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=size_bytes,
                            )
                        elif isinstance(data, DuckDBRelation):
                            size_bytes = _write_relation_artifact(output_path, data)
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=size_bytes,
                            )
                        elif isinstance(data, pa.RecordBatchReader):
                            size_bytes = _write_arrow_reader(output_path, data)
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=size_bytes,
                            )
                        elif isinstance(data, pa.Table):
                            size_bytes = _write_arrow_reader(output_path, data.to_reader())
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=size_bytes,
                            )
                        elif isinstance(data, Path) and _same_path(data, output_path):
                            size_bytes = output_path.stat().st_size
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=size_bytes,
                            )
                        else:
                            content_bytes = _coerce_bytes(data)
                            _atomic_write(output_path, content_bytes)
                            result = succeeded_artifact_result(
                                artifact_name=self.artifact_name,
                                duration_ms=duration_ms(start),
                                input_hash=input_hash or "",
                                path=str(output_path),
                                size_bytes=len(content_bytes),
                            )

        except _RECOVERABLE_EXCEPTIONS as exc:
            result = failed_artifact_result(
                artifact_name=self.artifact_name,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            result = failed_artifact_result(
                artifact_name=self.artifact_name,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown artifact materialization failure",
            )

        return result.to_mapping()


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _resolve_artifact_path(
    env: BuildEnv,
    target_name: str,
    artifact_name: str,
    *,
    path_template: str | None,
) -> Path | None:
    if not path_template:
        msg = (
            f"Missing artifact path_template for {target_name}.{artifact_name} "
            "on a contract output node."
        )
        raise ValueError(msg)
    return _resolve_artifact_path_from_template(env, path_template)


def _resolve_artifact_path_from_template(env: BuildEnv, template: str) -> Path:
    fmt = default_formatter(
        build_dir=str(env.paths.build_dir),
        scip_dir=str(env.paths.scip_dir),
        export_dir=str(env.paths.document_output_dir),
        repo_root=str(env.snapshot.repo_root),
    )
    return Path(format_path_template(template, formatter=fmt))


def _coerce_bytes(data: object) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, Path):
        return data.read_bytes()
    msg = f"Unsupported artifact payload type: {type(data).__name__}"
    raise TypeError(msg)


def _default_ipc_write_options() -> pa.ipc.IpcWriteOptions:
    return pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V5,
        compression="zstd",
        use_threads=True,
        unify_dictionaries=True,
    )


def _write_arrow_reader(output_path: Path, reader: pa.RecordBatchReader) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as sink:
        with pa.ipc.new_stream(sink, reader.schema, options=_default_ipc_write_options()) as writer:
            for batch in reader:
                writer.write_batch(batch)
    return output_path.stat().st_size


def _write_relation_artifact(output_path: Path, relation: DuckDBRelation) -> int:
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        relation.to_parquet(str(output_path))
        return output_path.stat().st_size
    if suffix in {".arrow", ".ipc"}:
        return _write_arrow_reader(output_path, relation.fetch_arrow_reader())
    msg = f"Unsupported relation artifact extension: {output_path.suffix}"
    raise ValueError(msg)


def _atomic_write_via_plan(output_path: Path, plan: ArtifactWritePlan) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        size_bytes = plan.write_to(tmp_path)
        tmp_path.rename(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        return size_bytes


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


__all__ = ["ArtifactWritePlan", "FileArtifactSaver"]
