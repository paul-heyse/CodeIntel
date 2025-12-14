"""Plugin catalog generation using the build registry.

This module provides functions for generating documentation catalogs
from the unified build registry (codeintel.build.unified_registry).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def _compute_version_hash(name: str, version: str) -> str:
    """Compute a hash of version-relevant metadata.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version string.

    Returns
    -------
    str
        Hash string (8 characters).
    """
    raw = f"{name}:{version}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def build_plugin_catalog() -> dict[str, Any]:
    """Build a JSON-serializable catalog of all registered plugins.

    Uses the build registry as the single source of truth for plugin
    discovery and metadata.

    Returns
    -------
    dict[str, Any]
        Catalog dict with 'plugins' key containing plugin metadata.
    """
    try:
        # Lazy import to avoid circular dependency at module load time
        from codeintel.build.unified_registry import get_unified_registry  # noqa: PLC0415
    except ImportError:
        log.warning("Build plugin registry not available")
        return {"plugins": {}, "count": 0}

    registry = get_unified_registry().get_all_plugins()
    plugins: dict[str, dict[str, Any]] = {}

    for target_name, plugin_class in registry.items():
        try:
            plugin = plugin_class()
        except (TypeError, ValueError, AttributeError) as exc:
            log.debug("Failed to instantiate plugin %s: %s", target_name, exc)
            continue

        plugin_name = getattr(plugin, "plugin_name", target_name)
        plugin_version = getattr(plugin, "plugin_version", "1.0.0")
        plugin_description = getattr(plugin, "plugin_description", "")

        plugins[plugin_name] = {
            "name": plugin_name,
            "description": plugin_description,
            "version": plugin_version,
            "version_hash": _compute_version_hash(plugin_name, plugin_version),
            "target": target_name,
            "stage": "build",
            "enabled_by_default": True,
            "depends_on": [],
            "provides": [],
            "requires": [],
        }

    return {"plugins": plugins, "count": len(plugins)}


def render_plugin_catalog_markdown(catalog: dict[str, Any] | None = None) -> str:
    """Render the plugin catalog as Markdown.

    Parameters
    ----------
    catalog
        Pre-built catalog dict. If None, builds a new one.

    Returns
    -------
    str
        Markdown-formatted catalog documentation.
    """
    if catalog is None:
        catalog = build_plugin_catalog()
    plugins = catalog.get("plugins", {})

    lines: list[str] = [
        "# Plugin Catalog",
        "",
        f"Total plugins: {catalog.get('count', 0)}",
        "",
    ]

    # Group plugins by stage
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for plugin_meta in plugins.values():
        stage = plugin_meta.get("stage", "other")
        by_stage.setdefault(stage, []).append(plugin_meta)

    for stage in sorted(by_stage.keys()):
        lines.append(f"## Stage: {stage}")
        lines.append("")

        for plugin_meta in sorted(by_stage[stage], key=lambda p: p["name"]):
            lines.append(f"### {plugin_meta['name']}")
            lines.append("")
            lines.append(plugin_meta.get("description", "No description") or "No description")
            lines.append("")
            lines.append(f"- **Version**: {plugin_meta.get('version', 'unknown')}")
            lines.append(f"- **Target**: {plugin_meta.get('target', 'unknown')}")
            lines.append(
                f"- **Enabled by default**: {plugin_meta.get('enabled_by_default', False)}"
            )

            if plugin_meta.get("depends_on"):
                deps = ", ".join(plugin_meta["depends_on"])
                lines.append(f"- **Depends on**: {deps}")

            if plugin_meta.get("provides"):
                provides = ", ".join(plugin_meta["provides"])
                lines.append(f"- **Provides**: {provides}")

            if plugin_meta.get("requires"):
                requires = ", ".join(plugin_meta["requires"])
                lines.append(f"- **Requires**: {requires}")

            lines.append("")

    # Add Plan Output Examples section
    lines.append("## Plan Output Examples")
    lines.append("")
    lines.append("Example pipeline plan output showing plugin execution order:")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps({"run_id": "example-run", "plugins": list(plugins.keys())[:3]}, indent=2)
    )
    lines.append("```")
    lines.append("")

    # Add Manifest excerpts section
    lines.append("## Manifest excerpts")
    lines.append("")
    lines.append("Example manifest entry for plugin configuration:")
    lines.append("")
    if plugins:
        first_name = next(iter(plugins))
        first_meta = plugins[first_name]
        lines.append("```yaml")
        lines.append(f"plugin: {first_name}")
        lines.append(f"target: {first_meta.get('target', 'unknown')}")
        lines.append(f"enabled: {first_meta.get('enabled_by_default', False)}")
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_plugin_catalog(path: Path) -> None:
    """Write the plugin catalog to a JSON file.

    Parameters
    ----------
    path
        Output file path.
    """
    catalog = build_plugin_catalog()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    log.info("Wrote plugin catalog to %s", path)


def write_plugin_catalog_markdown(path: Path, catalog: dict[str, Any] | None = None) -> None:
    """Write the plugin catalog to a Markdown file.

    Parameters
    ----------
    path
        Output file path.
    catalog
        Pre-built catalog dict. If None, builds a new one.
    """
    markdown = render_plugin_catalog_markdown(catalog)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    log.info("Wrote plugin catalog markdown to %s", path)


def write_plugin_catalog_html(path: Path, catalog: dict[str, Any] | None = None) -> None:
    """Write the plugin catalog to an HTML file.

    Parameters
    ----------
    path
        Output file path.
    catalog
        Pre-built catalog dict. If None, builds a new one.
    """
    markdown = render_plugin_catalog_markdown(catalog)
    # Simple HTML wrapper
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Plugin Catalog</title>
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        h3 {{ color: #666; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        ul {{ margin-left: 20px; }}
    </style>
</head>
<body>
<pre>{markdown}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    log.info("Wrote plugin catalog HTML to %s", path)


__all__ = [
    "build_plugin_catalog",
    "render_plugin_catalog_markdown",
    "write_plugin_catalog",
    "write_plugin_catalog_html",
    "write_plugin_catalog_markdown",
]
