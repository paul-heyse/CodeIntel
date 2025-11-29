"""Catalog utilities for graph metric plugins."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from codeintel.analytics.graphs.plugins import (
    GraphMetricPluginMetadata,
    graph_metric_plugin_metadata,
    list_graph_metric_plugins,
)


def _get_plugins_map(catalog: dict[str, object]) -> dict[str, dict[str, Any]]:
    plugins_raw = catalog.get("plugins", {})
    if isinstance(plugins_raw, dict):
        return cast("dict[str, dict[str, Any]]", plugins_raw)
    return {}


def build_plugin_catalog() -> dict[str, object]:
    """
    Build a JSON-serializable catalog of graph metric plugins.

    Returns
    -------
    dict[str, object]
        Mapping with plugin metadata keyed by name.
    """
    metadata: dict[str, object] = {}
    for plugin in list_graph_metric_plugins():
        meta: GraphMetricPluginMetadata = graph_metric_plugin_metadata(plugin)
        metadata[plugin.name] = {
            "name": meta.name,
            "description": meta.description,
            "stage": meta.stage,
            "severity": meta.severity,
            "enabled_by_default": meta.enabled_by_default,
            "depends_on": list(meta.depends_on),
            "provides": list(meta.provides),
            "requires": list(meta.requires),
            "resource_hints": (
                {
                    "max_runtime_ms": meta.resource_hints.max_runtime_ms,
                    "memory_mb_hint": meta.resource_hints.memory_mb_hint,
                }
                if meta.resource_hints is not None
                else None
            ),
            "options_model": meta.options_model.__name__ if meta.options_model else None,
            "options_default": meta.options_default,
            "version_hash": meta.version_hash,
            "contract_checkers": len(meta.contract_checkers),
            "scope_aware": meta.scope_aware,
            "supported_scopes": list(meta.supported_scopes),
            "requires_isolation": meta.requires_isolation,
            "isolation_kind": meta.isolation_kind,
            "config_schema_ref": meta.config_schema_ref,
            "row_count_tables": list(meta.row_count_tables),
            "cache_populates": list(meta.cache_populates),
            "cache_consumes": list(meta.cache_consumes),
        }
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "plugins": metadata,
        "count": len(metadata),
    }


def write_plugin_catalog(path: Path) -> None:
    """
    Write the plugin catalog to a file path as JSON.

    Parameters
    ----------
    path:
        Destination path for the catalog JSON.
    """
    catalog = build_plugin_catalog()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def _format_seq(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def render_plugin_catalog_markdown(catalog: dict[str, object]) -> str:
    """
    Render a Markdown summary of the plugin catalog.

    Parameters
    ----------
    catalog:
        Catalog payload produced by build_plugin_catalog.

    Returns
    -------
    str
        Markdown string.
    """
    plugins = _get_plugins_map(catalog)
    generated_at = str(catalog.get("generated_at", ""))
    lines: list[str] = []
    lines.append("# Graph Plugin Catalog")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append(f"Plugin count: {len(plugins)}")
    lines.append("")
    lines.append(
        "| Name | Stage | Severity | Enabled | Isolation | Scope-aware | Depends | Provides | "
        "Requires |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for name in sorted(plugins):
        meta = plugins[name]
        enabled = "yes" if meta["enabled_by_default"] else "no"
        isolation_kind = meta.get("isolation_kind") or (
            "yes" if meta.get("requires_isolation") else "no"
        )
        scope_flag = "yes" if meta.get("scope_aware") else "no"
        lines.append(
            "| {name} | {stage} | {severity} | {enabled} | {isolation} | {scope} | {depends} | "
            "{provides} | {requires} |".format(
                name=name,
                stage=meta["stage"],
                severity=meta["severity"],
                enabled=enabled,
                isolation=isolation_kind,
                scope=scope_flag,
                depends=_format_seq(meta.get("depends_on", [])),
                provides=_format_seq(meta.get("provides", [])),
                requires=_format_seq(meta.get("requires", [])),
            )
        )
    lines.append("")
    lines.append("## Plugin Details")
    for name in sorted(plugins):
        meta = plugins[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Description: {meta.get('description')}")
        lines.append(f"- Stage: {meta['stage']} (severity: {meta['severity']})")
        lines.append(f"- Enabled by default: {'yes' if meta['enabled_by_default'] else 'no'}")
        lines.append(
            f"- Isolation: {'yes' if meta['requires_isolation'] else 'no'} "
            f"({meta.get('isolation_kind') or 'none'})"
        )
        lines.append(
            f"- Scope-aware: {'yes' if meta.get('scope_aware') else 'no'} "
            f"(supports: {_format_seq(meta.get('supported_scopes', []))})"
        )
        lines.append(
            f"- Depends on: {_format_seq(meta.get('depends_on', []))}; "
            f"Provides: {_format_seq(meta.get('provides', []))}; "
            f"Requires data: {_format_seq(meta.get('requires', []))}"
        )
        lines.append(
            f"- Resource hints: {meta.get('resource_hints') or 'none'}; "
            f"Row count tables: {_format_seq(meta.get('row_count_tables', []))}"
        )
        lines.append(
            f"- Options model: {meta.get('options_model') or 'none'} "
            f"(default: {meta.get('options_default')})"
        )
        lines.append(
            f"- Config schema ref: {meta.get('config_schema_ref') or 'none'}; "
            f"Contracts: {meta.get('contract_checkers')}"
        )
        lines.append(f"- Version hash: {meta.get('version_hash') or 'n/a'}")
        lines.append("")
    lines.append("## Plan Output Examples")
    lines.append(
        "The CLI and MCP plan surfaces return ordered plugins, skip reasons, dependency graph, "
        "and enriched metadata. Example:"
    )
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "plan_id": "example-plan-id",
                "ordered_plugins": ["core_graph_metrics", "graph_metrics_modules_ext"],
                "skipped_plugins": [{"name": "graph_stats", "reason": "disabled"}],
                "dep_graph": {"core_graph_metrics": [], "graph_metrics_modules_ext": []},
                "plugin_metadata": {
                    "core_graph_metrics": {
                        "stage": "core",
                        "severity": "fatal",
                        "requires_isolation": False,
                        "isolation_kind": None,
                        "scope_aware": False,
                        "supported_scopes": [],
                        "depends_on": [],
                        "provides": [],
                        "requires": [],
                    }
                },
            },
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Manifest excerpts include correlation/run IDs, scope, isolation flags, and row counts:"
    )
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "run_id": "example-run-id",
                "scope": {"paths": ["src/"], "modules": [], "time_window": None},
                "plugins": [
                    {
                        "name": "core_graph_metrics",
                        "status": "ok",
                        "requires_isolation": False,
                        "row_counts": {"analytics.graph_metrics_functions": 42},
                        "contracts": [],
                    }
                ],
            },
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("To regenerate this catalog:")
    lines.append("")
    lines.append(
        "- Build JSON/Markdown/HTML: `uv run python scripts/render_graph_plugin_catalog.py`"
    )
    lines.append(
        "- Inspect plan ordering: `uv run codeintel graph plugins --plan --json` "
        "(shows dep graph, isolation, scope fields)."
    )
    lines.append(
        "- Inspect manifest snippets: run graph metrics to produce "
        "`build/graph-metrics/manifest.json`."
    )
    return "\n".join(lines)


def write_plugin_catalog_markdown(path: Path, catalog: dict[str, object] | None = None) -> None:
    """
    Write the plugin catalog as Markdown.

    Parameters
    ----------
    path:
        Destination path for the Markdown document.
    catalog:
        Optional pre-built catalog; build_plugin_catalog is used when omitted.
    """
    payload = build_plugin_catalog() if catalog is None else catalog
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_plugin_catalog_markdown(payload), encoding="utf-8")


def render_plugin_catalog_html(catalog: dict[str, object]) -> str:
    """
    Render a simple HTML view of the plugin catalog.

    Parameters
    ----------
    catalog:
        Catalog payload produced by build_plugin_catalog.

    Returns
    -------
    str
        HTML document string.
    """
    plugins = _get_plugins_map(catalog)
    generated_at = str(catalog.get("generated_at", ""))
    lines: list[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html>")
    lines.append("<head>")
    lines.append('<meta charset="utf-8" />')
    lines.append("<title>Graph Plugin Catalog</title>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("<h1>Graph Plugin Catalog</h1>")
    lines.append(f"<p>Generated at: {generated_at} | Plugin count: {len(plugins)}</p>")
    lines.append("<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">")
    lines.append(
        "<tr><th>Name</th><th>Stage</th><th>Severity</th><th>Enabled</th>"
        "<th>Isolation</th><th>Scope-aware</th><th>Depends</th><th>Provides</th>"
        "<th>Requires</th></tr>"
    )
    for name in sorted(plugins):
        meta = plugins[name]
        enabled = "yes" if meta["enabled_by_default"] else "no"
        isolation_kind = meta.get("isolation_kind") or (
            "yes" if meta.get("requires_isolation") else "no"
        )
        scope_flag = "yes" if meta.get("scope_aware") else "no"
        lines.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{meta['stage']}</td>"
            f"<td>{meta['severity']}</td>"
            f"<td>{enabled}</td>"
            f"<td>{isolation_kind}</td>"
            f"<td>{scope_flag}</td>"
            f"<td>{_format_seq(meta.get('depends_on', []))}</td>"
            f"<td>{_format_seq(meta.get('provides', []))}</td>"
            f"<td>{_format_seq(meta.get('requires', []))}</td>"
            "</tr>"
        )
    lines.append("</table>")
    lines.append("<h2>Plugin Details</h2>")
    for name in sorted(plugins):
        meta = plugins[name]
        lines.append(f"<h3>{name}</h3>")
        lines.append("<ul>")
        lines.append(f"<li>Description: {meta.get('description')}</li>")
        lines.append(
            "<li>"
            f"Stage: {meta['stage']} (severity: {meta['severity']}); "
            f"Enabled by default: {'yes' if meta['enabled_by_default'] else 'no'}"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Isolation: {'yes' if meta.get('requires_isolation') else 'no'} "
            f"({meta.get('isolation_kind') or 'none'})"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Scope-aware: {'yes' if meta.get('scope_aware') else 'no'} "
            f"(supports: {_format_seq(meta.get('supported_scopes', []))})"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Depends on: {_format_seq(meta.get('depends_on', []))}; "
            f"Provides: {_format_seq(meta.get('provides', []))}; "
            f"Requires data: {_format_seq(meta.get('requires', []))}"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Resource hints: {meta.get('resource_hints') or 'none'}; "
            f"Row count tables: {_format_seq(meta.get('row_count_tables', []))}"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Options model: {meta.get('options_model') or 'none'} "
            f"(default: {meta.get('options_default')})"
            "</li>"
        )
        lines.append(
            "<li>"
            f"Config schema ref: {meta.get('config_schema_ref') or 'none'}; "
            f"Contracts: {meta.get('contract_checkers')}"
            "</li>"
        )
        lines.append(f"<li>Version hash: {meta.get('version_hash') or 'n/a'}</li>")
        lines.append("</ul>")
    lines.append("<h2>Plan Output Examples</h2>")
    lines.append(
        "<p>Plan responses include ordered plugins, skip reasons, dependency graph, and "
        "enriched metadata.</p>"
    )
    lines.append("<pre>")
    lines.append(
        json.dumps(
            {
                "plan_id": "example-plan-id",
                "ordered_plugins": ["core_graph_metrics", "graph_metrics_modules_ext"],
                "skipped_plugins": [{"name": "graph_stats", "reason": "disabled"}],
                "dep_graph": {"core_graph_metrics": [], "graph_metrics_modules_ext": []},
                "plugin_metadata": {
                    "core_graph_metrics": {
                        "stage": "core",
                        "severity": "fatal",
                        "requires_isolation": False,
                        "isolation_kind": None,
                        "scope_aware": False,
                        "supported_scopes": [],
                        "depends_on": [],
                        "provides": [],
                        "requires": [],
                    }
                },
            },
            indent=2,
        )
    )
    lines.append("</pre>")
    lines.append("<p>Manifest excerpts carry run/scope IDs, isolation flags, and row counts:</p>")
    lines.append("<pre>")
    lines.append(
        json.dumps(
            {
                "run_id": "example-run-id",
                "scope": {"paths": ["src/"], "modules": [], "time_window": None},
                "plugins": [
                    {
                        "name": "core_graph_metrics",
                        "status": "ok",
                        "requires_isolation": False,
                        "row_counts": {"analytics.graph_metrics_functions": 42},
                        "contracts": [],
                    }
                ],
            },
            indent=2,
        )
    )
    lines.append("</pre>")
    lines.append("<p>")
    lines.append(
        "Regenerate: uv run python scripts/render_graph_plugin_catalog.py "
        "(JSON + Markdown + HTML)."
    )
    lines.append("</p>")
    lines.append("</body>")
    lines.append("</html>")
    return "\n".join(lines)


def write_plugin_catalog_html(path: Path, catalog: dict[str, object] | None = None) -> None:
    """
    Write the plugin catalog as HTML.

    Parameters
    ----------
    path:
        Destination path for the HTML document.
    catalog:
        Optional pre-built catalog; build_plugin_catalog is used when omitted.
    """
    payload = build_plugin_catalog() if catalog is None else catalog
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_plugin_catalog_html(payload), encoding="utf-8")
