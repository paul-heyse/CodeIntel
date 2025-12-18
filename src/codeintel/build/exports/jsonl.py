"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from codeintel.build.exports.common import MAX_EXPORT_LIMIT, build_export_relation
from codeintel.build.exports.engine import export_all_datasets
from codeintel.build.exports.engine import export_jsonl_for_table as _engine_export_jsonl_for_table

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from codeintel.build.exports.common import ExportCallOptions
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class _SupportsIsoformat(Protocol):
    def isoformat(self) -> str: ...


def _default_serializer(obj: object) -> object:
    """Serialize objects for JSON output.

    Parameters
    ----------
    obj
        Object to serialize.

    Returns
    -------
    object
        JSON-serializable representation.

    Raises
    ------
    TypeError
        If object is not serializable.
    """
    if isinstance(obj, _SupportsIsoformat):
        return obj.isoformat()
    message = f"Type {type(obj)} is not JSON serializable"
    raise TypeError(message)


def export_jsonl_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
    *,
    serializer: Callable[[object], object] | None = None,
) -> None:
    """Export a single DuckDB table to JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    table_name
        Fully qualified table name (schema.table) to export.
    output_path
        Destination path for the JSONL file.
    serializer
        Custom JSON serializer for complex types.

    Raises
    ------
    ValueError
        If the requested table is not registered in the dataset mapping.
    """
    dataset_mapping = gateway.datasets.mapping
    if table_name not in dataset_mapping.values():
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    if serializer is not None:
        msg = "export_jsonl_for_table does not support custom serializer in engine mode"
        raise ValueError(msg)
    _engine_export_jsonl_for_table(gateway, table_name, output_path)


def export_dataset_to_jsonl(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Export a dataset resolved through the dataset registry to JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    dataset_name
        Logical dataset name to export (e.g., ``function_profile``).
    output_dir
        Destination directory for the JSONL file.

    Returns
    -------
    Path
        Path to the written JSONL file.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    dataset_mapping = gateway.datasets.mapping
    jsonl_mapping = gateway.datasets.jsonl_mapping or {}
    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    table_name = dataset_mapping[dataset_name]
    filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
    output_path = output_dir / filename
    export_jsonl_for_table(gateway, table_name, output_path)
    return output_path


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> list[Path]:
    """Export configured datasets to JSONL files under `Document Output/`.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where JSONL artifacts are written.
    options
        Export options controlling dataset selection and validation.

    Returns
    -------
    list[Path]
        List of written file paths.
    """
    return export_all_datasets(
        gateway,
        document_output_dir,
        fmt="jsonl",
        options=options,
    )


def export_repo_map_json(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    format_output: Literal["json", "jsonl"] = "json",
) -> Path:
    """Export the repo_map table as JSON or JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where the export artifact is written.
    format_output
        Output format; "json" produces a single JSON array, "jsonl"
        produces newline-delimited JSON records.

    Returns
    -------
    Path
        Path to the written file.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)
    table_name = "core.repo_map"
    if format_output == "json":
        output_path = document_output_dir / "repo_map.json"
        rel = build_export_relation(gateway, table_name, MAX_EXPORT_LIMIT, 0)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("[")
            first = True
            reader = rel.fetch_record_batch(10_000)
            for batch in reader:
                payload = batch.to_pydict()
                columns = list(payload.keys())
                for idx in range(batch.num_rows):
                    record = {name: payload[name][idx] for name in columns}
                    if first:
                        first = False
                    else:
                        handle.write(",")
                    handle.write(json.dumps(record, default=_default_serializer))
            handle.write("]\n")
    else:
        output_path = document_output_dir / "repo_map.jsonl"
        export_jsonl_for_table(gateway, table_name, output_path)
    return output_path


__all__ = [
    "export_all_jsonl",
    "export_dataset_to_jsonl",
    "export_jsonl_for_table",
    "export_repo_map_json",
]
