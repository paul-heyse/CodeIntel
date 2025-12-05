#!/usr/bin/env python3
"""Audit plugin output tables against TABLE_SCHEMAS.

This script identifies:
1. Tables declared by plugins but missing from TABLE_SCHEMAS
2. Tables in TABLE_SCHEMAS not used by any plugin
3. Artifact-like declarations in output_tables (e.g., "index.scip")

Usage:
    uv run python tools/audit_plugin_schemas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codeintel.ingestion.plugins.registry import (
    get_ingest_registry,
    register_class_based_plugins,
)

from codeintel.config.datasets import TABLE_SCHEMAS


def main() -> int:
    """Run the plugin schema audit."""
    print("=" * 70)
    print("PLUGIN SCHEMA AUDIT")
    print("=" * 70)
    print()

    # Register plugins
    register_class_based_plugins()
    registry = get_ingest_registry()

    # Collect all declared output tables
    plugin_tables: dict[str, list[str]] = {}  # table_key -> [plugin_names]
    artifacts: list[tuple[str, str]] = []  # (artifact, plugin_name)

    for plugin in registry.all_plugins():
        meta = plugin.metadata
        for table_key in meta.output_tables:
            # Check if this looks like an artifact
            if "." not in table_key or table_key.startswith("index."):
                artifacts.append((table_key, meta.name))
                continue

            plugin_tables.setdefault(table_key, []).append(meta.name)

    # 1. Tables declared but missing from schemas
    print("1. MISSING SCHEMAS (declared by plugins, not in TABLE_SCHEMAS)")
    print("-" * 70)
    missing = []
    for table_key, plugins in sorted(plugin_tables.items()):
        if table_key not in TABLE_SCHEMAS:
            missing.append((table_key, plugins))
            print(f"  ❌ {table_key}")
            print(f"     Declared by: {', '.join(plugins)}")
    if not missing:
        print("  ✅ All plugin tables have schemas defined")
    print()

    # 2. Artifact declarations in output_tables
    print("2. ARTIFACTS IN output_tables (should use output_artifacts)")
    print("-" * 70)
    if artifacts:
        for artifact, plugin in artifacts:
            print(f"  ⚠️  {artifact}")
            print(f"     Declared by: {plugin}")
    else:
        print("  ✅ No artifacts found in output_tables")
    print()

    # 3. Tables with schemas but no plugin
    print("3. ORPHAN SCHEMAS (in TABLE_SCHEMAS, no plugin writes to them)")
    print("-" * 70)
    used_tables = set(plugin_tables.keys())
    orphans = []
    for table_key in sorted(TABLE_SCHEMAS.keys()):
        # Skip views and metadata tables
        if table_key.startswith(("metadata.", "build.", "docs.")):
            continue
        if table_key not in used_tables:
            orphans.append(table_key)

    if orphans:
        # Group by schema
        by_schema: dict[str, list[str]] = {}
        for t in orphans:
            schema = t.split(".")[0]
            by_schema.setdefault(schema, []).append(t)

        for schema, tables in sorted(by_schema.items()):
            if len(tables) > 5:
                print(f"  {schema}.*: {len(tables)} tables (not from ingestion)")
            else:
                for t in tables:
                    print(f"  {t}")
    else:
        print("  ✅ All schemas are used by plugins")
    print()

    # 4. Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Plugins registered: {len(list(registry.all_plugins()))}")
    print(f"  Tables declared: {len(plugin_tables)}")
    print(f"  Missing schemas: {len(missing)}")
    print(f"  Artifacts misclassified: {len(artifacts)}")
    print()

    # Return error code if issues found
    if missing or artifacts:
        print("⚠️  Issues found - see above for details")
        return 1

    print("✅ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
