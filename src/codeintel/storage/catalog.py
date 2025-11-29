"""Dataset catalog generation helpers for docs/CI artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from codeintel.pipeline.export.manifest import compute_file_hash
from codeintel.serving.http.datasets import describe_dataset
from codeintel.storage.datasets import Dataset, DatasetRegistry
from codeintel.storage.gateway import DuckDBConnection
from codeintel.storage.repositories.base import fetch_all_dicts


@dataclass(frozen=True)
class CatalogEntry:
    """Rendered catalog entry for a dataset."""

    name: str
    table: str
    description: str
    schema_columns: list[dict[str, object]]
    sample_rows: list[dict[str, object]]
    json_schema_id: str | None
    json_schema_digest: str | None
    jsonl_filename: str | None
    parquet_filename: str | None
    capabilities: dict[str, bool]
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    schema_version: str | None
    stable_id: str | None
    validation_profile: str | None
    upstream_dependencies: list[str]


def _schema_columns(dataset: Dataset) -> list[dict[str, object]]:
    """
    Serialize schema columns for a dataset.

    Returns
    -------
    list[dict[str, object]]
        Column descriptors including name, type, and nullability.
    """
    if dataset.schema is None:
        return []
    return [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in dataset.schema.columns
    ]


def _slug(name: str) -> str:
    """
    Return a simple slug for anchors.

    Returns
    -------
    str
        Lowercase anchor-friendly string.
    """
    return name.lower().replace(" ", "-")


def _handle_sampling_failure(
    *,
    dataset_name: str,
    strict: bool,
    warn: Callable[[str], None] | None,
    exc: Exception,
) -> list[dict[str, object]]:
    message = f"Failed to sample rows for {dataset_name}: {exc}"
    if strict:
        raise RuntimeError(message) from exc
    if warn is not None:
        warn(message)
    return []


def _sample_rows(
    con: DuckDBConnection | None,
    dataset: Dataset,
    *,
    limit: int,
    strict: bool,
    warn: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """
    Collect sample rows via metadata.dataset_rows when available.

    Returns
    -------
    list[dict[str, object]]
        Sample rows, empty on errors or when sampling is disabled.

    Raises
    ------
    RuntimeError
        When sampling is strict and the dataset_rows macro is unavailable or sampling fails.
    """
    limited = max(0, limit)
    if con is None or limited == 0:
        return []
    if strict:
        try:
            available = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.table_functions
                WHERE table_function_name = 'dataset_rows'
                """
            ).fetchone()
        except Exception as exc:
            message = f"Failed to check sampling macro availability for {dataset.name}: {exc}"
            raise RuntimeError(message) from exc
        if available is None or int(available[0]) == 0:
            message = (
                f"dataset_rows macro unavailable for sampling; skipping samples for {dataset.name}"
            )
            raise RuntimeError(message)
        return fetch_all_dicts(
            con,
            "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
            [dataset.table_key, limited, 0],
        )
    try:
        available = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.table_functions
            WHERE table_function_name = 'dataset_rows'
            """
        ).fetchone()
        if available is None or int(available[0]) == 0:
            message = (
                f"dataset_rows macro unavailable for sampling; skipping samples for {dataset.name}"
            )
            if warn is not None:
                warn(message)
            return []
        return fetch_all_dicts(
            con,
            "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
            [dataset.table_key, limited, 0],
        )
    except Exception as exc:  # noqa: BLE001 - surface empty sample on errors
        return _handle_sampling_failure(
            dataset_name=dataset.name,
            strict=strict,
            warn=warn,
            exc=exc,
        )


def _schema_digest(dataset: Dataset) -> str | None:
    """
    Compute a JSON Schema digest when the schema file exists.

    Returns
    -------
    str | None
        Hex digest for the JSON Schema file if present.
    """
    if dataset.json_schema_id is None:
        return None
    path = Path("src/codeintel/config/schemas/export") / f"{dataset.json_schema_id}.json"
    if not path.exists():
        return None
    return compute_file_hash(path)


def build_catalog(
    registry: DatasetRegistry,
    *,
    con: DuckDBConnection | None,
    sample_rows: int = 3,
    sample_rows_strict: bool = False,
    warn: Callable[[str], None] | None = None,
) -> list[CatalogEntry]:
    """
    Build an in-memory catalog from a DatasetRegistry.

    Parameters
    ----------
    registry
        Dataset registry loaded from metadata.datasets.
    con
        Optional DuckDB connection used for sampling rows.
    sample_rows
        Number of rows to sample per dataset (0 to skip).
    sample_rows_strict
        When True, raise on sampling failures instead of silently skipping.
    warn
        Optional warning callback to surface sampling issues.

    Returns
    -------
    list[CatalogEntry]
        Catalog entries sorted by dataset name.
    """
    return [
        CatalogEntry(
            name=name,
            table=ds.table_key,
            description=describe_dataset(ds.name, ds.table_key),
            schema_columns=_schema_columns(ds),
            sample_rows=_sample_rows(
                con,
                ds,
                limit=sample_rows,
                strict=sample_rows_strict,
                warn=warn,
            ),
            json_schema_id=ds.json_schema_id,
            json_schema_digest=_schema_digest(ds),
            jsonl_filename=ds.jsonl_filename,
            parquet_filename=ds.parquet_filename,
            capabilities=ds.capabilities(),
            owner=ds.owner,
            freshness_sla=ds.freshness_sla,
            retention_policy=ds.retention_policy,
            schema_version=ds.schema_version,
            stable_id=ds.stable_id,
            validation_profile=ds.validation_profile,
            upstream_dependencies=list(ds.upstream_dependencies),
        )
        for name, ds in sorted(registry.by_name.items())
    ]


def write_markdown_catalog(output_dir: Path, entries: Iterable[CatalogEntry]) -> Path:
    """
    Render a Markdown catalog and return the written path.

    Returns
    -------
    Path
        Path to the generated Markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "catalog.md"
    rendered = list(entries)
    lines: list[str] = [
        "# Dataset Catalog",
        "",
        "Generated from the dataset registry with sample rows and capabilities.",
        "",
        "## Datasets",
        "",
        *[f"- [{entry.name}](#{_slug(entry.name)})" for entry in rendered],
        "",
    ]
    for entry in rendered:
        lines.extend(
            [
                f"## {entry.name}",
                "",
                f"- Table: `{entry.table}`",
                f"- Description: {entry.description}",
                f"- Owner: {entry.owner or 'n/a'}",
                f"- Freshness: {entry.freshness_sla or 'n/a'}",
                f"- Retention: {entry.retention_policy or 'n/a'}",
                f"- Schema version: {entry.schema_version or 'n/a'}",
                f"- Stable ID: {entry.stable_id or 'n/a'}",
                f"- Validation profile: {entry.validation_profile or 'strict'}",
                f"- JSONL filename: {entry.jsonl_filename or 'n/a'}",
                f"- Parquet filename: {entry.parquet_filename or 'n/a'}",
                f"- JSON Schema: {entry.json_schema_id or 'n/a'}",
                f"- Schema digest: {entry.json_schema_digest or 'n/a'}",
                f"- Capabilities: {json.dumps(entry.capabilities, sort_keys=True)}",
                f"- Upstream deps: {', '.join(entry.upstream_dependencies) if entry.upstream_dependencies else 'none'}",
                "",
                "### Columns",
            ]
        )
        if entry.schema_columns:
            lines.extend(
                [
                    "| name | type | nullable |",
                    "| --- | --- | --- |",
                    *[
                        f"| {col['name']} | {col['type']} | {'yes' if col['nullable'] else 'no'} |"
                        for col in entry.schema_columns
                    ],
                ]
            )
        else:
            lines.append("_No static schema (view or unknown)._")
        lines.append("")
        lines.append("### Sample rows")
        if entry.sample_rows:
            headers = list(entry.sample_rows[0].keys())
            header_row = "| " + " | ".join(headers) + " |"
            separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = [
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
                for row in entry.sample_rows
            ]
            lines.extend([header_row, separator_row, *data_rows])
        else:
            lines.append("_No sample rows available._")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_html_catalog(output_dir: Path, entries: Iterable[CatalogEntry]) -> Path:
    """
    Render an HTML catalog and return the written path.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = list(entries)
    path = output_dir / "catalog.html"
    parts: list[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Dataset Catalog</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:0 auto;padding:24px;}table{border-collapse:collapse;width:100%;margin:12px 0;}th,td{border:1px solid #ddd;padding:8px;}th{background:#f3f3f3;text-align:left;}code{background:#f7f7f7;padding:2px 4px;border-radius:4px;}</style>",
        "</head><body>",
        "<h1>Dataset Catalog</h1>",
        "<p>Generated from the dataset registry with sample rows and capabilities.</p>",
    ]
    parts.append("<nav><ul>")
    parts.extend(f"<li><a href='#{_slug(entry.name)}'>{entry.name}</a></li>" for entry in rendered)
    parts.append("</ul></nav>")
    for entry in rendered:
        slug = _slug(entry.name)
        parts.append(f"<h2 id='{slug}'>{entry.name}</h2>")
        parts.append("<ul>")
        parts.append(f"<li><strong>Table</strong>: <code>{entry.table}</code></li>")
        parts.append(f"<li><strong>Description</strong>: {entry.description}</li>")
        parts.append(f"<li><strong>Owner</strong>: {entry.owner or 'n/a'}</li>")
        parts.append(f"<li><strong>Freshness</strong>: {entry.freshness_sla or 'n/a'}</li>")
        parts.append(f"<li><strong>Retention</strong>: {entry.retention_policy or 'n/a'}</li>")
        parts.append(f"<li><strong>Schema version</strong>: {entry.schema_version or 'n/a'}</li>")
        parts.append(f"<li><strong>Stable ID</strong>: {entry.stable_id or 'n/a'}</li>")
        parts.append(
            f"<li><strong>Validation profile</strong>: {entry.validation_profile or 'strict'}</li>"
        )
        parts.append(f"<li><strong>JSONL filename</strong>: {entry.jsonl_filename or 'n/a'}</li>")
        parts.append(
            f"<li><strong>Parquet filename</strong>: {entry.parquet_filename or 'n/a'}</li>"
        )
        parts.append(f"<li><strong>JSON Schema</strong>: {entry.json_schema_id or 'n/a'}</li>")
        parts.append(
            f"<li><strong>Schema digest</strong>: {entry.json_schema_digest or 'n/a'}</li>"
        )
        parts.append(
            f"<li><strong>Capabilities</strong>: {json.dumps(entry.capabilities, sort_keys=True)}</li>"
        )
        deps = ", ".join(entry.upstream_dependencies) if entry.upstream_dependencies else "none"
        parts.append(f"<li><strong>Upstream deps</strong>: {deps}</li>")
        parts.append("</ul>")

        parts.append("<h3>Columns</h3>")
        if entry.schema_columns:
            parts.extend(
                [
                    "<table><thead><tr><th>name</th><th>type</th><th>nullable</th></tr></thead>",
                    "<tbody>",
                    *[
                        f"<tr><td>{col['name']}</td><td>{col['type']}</td>"
                        f"<td>{'yes' if col['nullable'] else 'no'}</td></tr>"
                        for col in entry.schema_columns
                    ],
                    "</tbody></table>",
                ]
            )
        else:
            parts.append("<p><em>No static schema (view or unknown).</em></p>")

        parts.append("<h3>Sample rows</h3>")
        if entry.sample_rows:
            headers = list(entry.sample_rows[0].keys())
            header_cells = "".join(f"<th>{h}</th>" for h in headers)
            parts.extend(
                [
                    "<table><thead><tr>" + header_cells + "</tr></thead>",
                    "<tbody>",
                    *[
                        "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
                        for row in entry.sample_rows
                    ],
                    "</tbody></table>",
                ]
            )
        else:
            parts.append("<p><em>No sample rows available.</em></p>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
