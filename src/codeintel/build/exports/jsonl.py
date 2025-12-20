"""JSON/JSONL exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from codeintel.build.exports.common import MAX_EXPORT_LIMIT, build_export_relation
from codeintel.build.exports.engine import export_all_datasets
from codeintel.build.exports.engine import export_jsonl_for_table as _engine_export_jsonl_for_table
from codeintel.build.exports.writers import default_json_serializer
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from codeintel.build.exports.common import ExportCallOptions
    from codeintel.storage.gateway import StorageGateway


def export_jsonl_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
    *,
    settings: ExportAuditSettings,
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
    settings
        Export audit settings.
    serializer
        Custom JSON serializer for complex types.

    Raises
    ------
    ValueError
        If the requested table is not registered in the dataset mapping.
    """
    registry = gateway.datasets
    if table_name not in registry.by_table_key:
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    if serializer is not None:
        msg = "export_jsonl_for_table does not support custom serializer in engine mode"
        raise ValueError(msg)
    _engine_export_jsonl_for_table(gateway, table_name, output_path, settings)


def export_dataset_to_jsonl(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
    *,
    settings: ExportAuditSettings,
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
    settings
        Export audit settings.

    Returns
    -------
    Path
        Path to the written JSONL file.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    registry = gateway.datasets
    jsonl_mapping = registry.jsonl_datasets
    try:
        table_name = registry.resolve_table_key(dataset_name)
    except KeyError as exc:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message) from exc
    filename = jsonl_mapping.get(table_name, f"{dataset_name}.jsonl")
    output_path = output_dir / filename
    export_jsonl_for_table(gateway, table_name, output_path, settings=settings)
    return output_path


def export_all_jsonl(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    settings: ExportAuditSettings,
    options: ExportCallOptions | None = None,
) -> list[Path]:
    """Export configured datasets to JSONL files under `Document Output/`.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where JSONL artifacts are written.
    settings
        Export audit settings.
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
        settings=settings,
        options=options,
    )


def export_repo_map_json(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    settings: ExportAuditSettings,
    format_output: Literal["json", "jsonl"] = "json",
) -> Path:
    """Export the repo_map table as JSON or JSONL.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where the export artifact is written.
    settings
        Export audit settings.
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
            reader = rel.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
            for batch in reader:
                payload = batch.to_pydict()
                columns = list(payload.keys())
                for idx in range(batch.num_rows):
                    record = {name: payload[name][idx] for name in columns}
                    if first:
                        first = False
                    else:
                        handle.write(",")
                    handle.write(json.dumps(record, default=default_json_serializer))
            handle.write("]\n")
    else:
        output_path = document_output_dir / "repo_map.jsonl"
        export_jsonl_for_table(gateway, table_name, output_path, settings=settings)
    return output_path


__all__ = [
    "export_all_jsonl",
    "export_dataset_to_jsonl",
    "export_jsonl_for_table",
    "export_repo_map_json",
]
