"""Helpers to emit dataset-to-filename manifests for Document Output exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def write_dataset_manifest(
    output_dir: Path,
    dataset_mapping: Mapping[str, str],
    *,
    jsonl_mapping: Mapping[str, str],
    parquet_mapping: Mapping[str, str],
    selected: list[str] | None = None,
) -> Path:
    """
    Write a manifest mapping dataset names to export filenames.

    Parameters
    ----------
    output_dir :
        Document Output directory where the manifest will be written.
    dataset_mapping :
        Registry mapping dataset name -> fully qualified table/view name.
    jsonl_mapping :
        Mapping of table -> JSONL filename for datasets with JSON exports.
    parquet_mapping :
        Mapping of table -> Parquet filename for datasets with Parquet exports.
    selected :
        Optional subset of dataset names requested for export.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected) if selected is not None else None
    entries: list[dict[str, object]] = []

    for name, table in sorted(dataset_mapping.items()):
        entry: dict[str, object] = {"name": name, "table": table}
        if table in jsonl_mapping:
            entry["jsonl"] = jsonl_mapping[table]
        if table in parquet_mapping:
            entry["parquet"] = parquet_mapping[table]
        if selected_set is not None:
            entry["selected"] = name in selected_set
        entries.append(entry)

    manifest = {"datasets": entries}
    path = output_dir / "datasets_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
