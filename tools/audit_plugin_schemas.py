"""Audit plugin output tables against TABLE_SCHEMAS."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codeintel.config.datasets import TABLE_SCHEMAS

DIVIDER = "=" * 70
SECTION_DIVIDER = "-" * 70
ORPHAN_TABLE_DISPLAY_THRESHOLD = 5


@dataclass(frozen=True)
class PluginAuditState:
    """Computed state for the audit report."""

    plugin_tables: dict[str, list[str]]
    artifacts: list[tuple[str, str]]
    missing_schemas: list[tuple[str, list[str]]]
    orphan_tables_by_schema: dict[str, list[str]]
    plugin_count: int


def _collect_plugin_tables() -> tuple[dict[str, list[str]], list[tuple[str, str]], int]:
    register_class_based_plugins()
    registry = get_ingest_registry()
    plugin_tables: dict[str, list[str]] = {}
    artifacts: list[tuple[str, str]] = []
    for plugin in registry.all_plugins():
        meta = plugin.metadata
        for table_key in meta.output_tables:
            if "." not in table_key or table_key.startswith("index."):
                artifacts.append((table_key, meta.name))
                continue
            plugin_tables.setdefault(table_key, []).append(meta.name)
    return plugin_tables, artifacts, len(list(registry.all_plugins()))


def _find_missing_schemas(plugin_tables: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    return [
        (table_key, plugins)
        for table_key, plugins in sorted(plugin_tables.items())
        if table_key not in TABLE_SCHEMAS
    ]


def _find_orphan_tables(plugin_tables: dict[str, list[str]]) -> dict[str, list[str]]:
    used_tables = set(plugin_tables.keys())
    orphans: dict[str, list[str]] = {}
    for table_key in sorted(TABLE_SCHEMAS.keys()):
        if table_key.startswith(("metadata.", "build.", "docs.")):
            continue
        if table_key in used_tables:
            continue
        schema = table_key.split(".", maxsplit=1)[0]
        orphans.setdefault(schema, []).append(table_key)
    return orphans


def _build_state() -> PluginAuditState:
    plugin_tables, artifacts, plugin_count = _collect_plugin_tables()
    missing_schemas = _find_missing_schemas(plugin_tables)
    orphan_tables_by_schema = _find_orphan_tables(plugin_tables)
    return PluginAuditState(
        plugin_tables=plugin_tables,
        artifacts=artifacts,
        missing_schemas=missing_schemas,
        orphan_tables_by_schema=orphan_tables_by_schema,
        plugin_count=plugin_count,
    )


def _format_missing_schemas(missing: list[tuple[str, list[str]]]) -> list[str]:
    lines = [
        "1. MISSING SCHEMAS (declared by plugins, not in TABLE_SCHEMAS)",
        SECTION_DIVIDER,
    ]
    if not missing:
        lines.append("  ✅ All plugin tables have schemas defined")
        lines.append("")
        return lines

    for table_key, plugins in missing:
        lines.append(f"  ❌ {table_key}")
        lines.append(f"     Declared by: {', '.join(plugins)}")
    lines.append("")
    return lines


def _format_artifacts(artifacts: list[tuple[str, str]]) -> list[str]:
    lines = [
        "2. ARTIFACTS IN output_tables (should use output_artifacts)",
        SECTION_DIVIDER,
    ]
    if not artifacts:
        lines.append("  ✅ No artifacts found in output_tables")
        lines.append("")
        return lines

    for artifact, plugin in artifacts:
        lines.append(f"  ⚠️  {artifact}")
        lines.append(f"     Declared by: {plugin}")
    lines.append("")
    return lines


def _format_orphans(orphan_tables_by_schema: dict[str, list[str]]) -> list[str]:
    lines = [
        "3. ORPHAN SCHEMAS (in TABLE_SCHEMAS, no plugin writes to them)",
        SECTION_DIVIDER,
    ]
    if not orphan_tables_by_schema:
        lines.append("  ✅ All schemas are used by plugins")
        lines.append("")
        return lines

    for schema, tables in sorted(orphan_tables_by_schema.items()):
        if len(tables) > ORPHAN_TABLE_DISPLAY_THRESHOLD:
            lines.append(f"  {schema}.*: {len(tables)} tables (not from ingestion)")
            continue
        lines.extend(f"  {table}" for table in sorted(tables))
    lines.append("")
    return lines


def _format_summary(state: PluginAuditState) -> list[str]:
    lines = [
        DIVIDER,
        "SUMMARY",
        DIVIDER,
        f"  Plugins registered: {state.plugin_count}",
        f"  Tables declared: {len(state.plugin_tables)}",
        f"  Missing schemas: {len(state.missing_schemas)}",
        f"  Artifacts misclassified: {len(state.artifacts)}",
        "",
    ]
    if state.missing_schemas or state.artifacts:
        lines.append("⚠️  Issues found - see above for details")
    else:
        lines.append("✅ All checks passed")
    return lines


def _format_report(state: PluginAuditState) -> str:
    lines = [
        DIVIDER,
        "PLUGIN SCHEMA AUDIT",
        DIVIDER,
        "",
    ]
    lines.extend(_format_missing_schemas(state.missing_schemas))
    lines.extend(_format_artifacts(state.artifacts))
    lines.extend(_format_orphans(state.orphan_tables_by_schema))
    lines.extend(_format_summary(state))
    return "\n".join(lines)


def main() -> int:
    """Run the plugin schema audit.

    Returns
    -------
    int
        Exit code: 0 when clean, 1 when issues are detected.
    """
    state = _build_state()
    sys.stdout.write(_format_report(state))
    sys.stdout.write("\n")
    if state.missing_schemas or state.artifacts:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
