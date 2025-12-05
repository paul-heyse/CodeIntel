#!/usr/bin/env python3
"""Analyze usage of config package exports across the codebase.

This script identifies potentially unused configuration types, dataclasses,
and other exports from the codeintel.config package by searching for their
usage across the entire codebase.

Usage
-----
    uv run python -m tools.analyze_config_usage [--json] [--threshold N]

Options
-------
--json
    Output results as JSON instead of formatted text.
--threshold N
    Flag items with fewer than N external uses (default: 1).
--include-tests
    Include test files in usage counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportInfo:
    """Information about a single export from config."""

    name: str
    source_file: str
    kind: str  # "class", "function", "constant", "type_alias"
    external_uses: int = 0
    test_uses: int = 0
    internal_uses: int = 0  # Within config package
    used_in_files: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    total_exports: int
    unused_count: int
    low_usage_count: int
    exports: list[ExportInfo]


def get_config_exports() -> dict[str, list[str]]:
    """Extract all exports from config package modules.

    Returns
    -------
    dict[str, list[str]]
        Mapping of source file to list of exported names.
    """
    config_dir = Path("src/codeintel/config")
    exports: dict[str, list[str]] = {}

    # Common names to skip (properties, methods, etc.)
    skip_names = {
        "repo", "commit", "repo_root", "build_dir", "snapshot", "paths",
        "default", "analytics", "graphs", "ingestion", "profiles",
        "code", "config", "from_args", "from_layout", "from_primitives",
    }

    # Parse each Python file in config
    for py_file in config_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        file_exports: list[str] = []

        for node in ast.walk(tree):
            # Get __all__ if defined - this is the authoritative source
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    name = elt.value
                                    if name not in skip_names and len(name) > 3:
                                        file_exports.append(name)

        # If no __all__, fall back to class definitions
        if not file_exports:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_") and node.name not in skip_names:
                        file_exports.append(node.name)

        if file_exports:
            exports[str(py_file)] = file_exports

    # Also check datasets subpackage
    datasets_dir = config_dir / "datasets"
    if datasets_dir.exists():
        for py_file in datasets_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            file_exports: list[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(
                                        elt.value, str
                                    ):
                                        name = elt.value
                                        if name not in skip_names and len(name) > 3:
                                            file_exports.append(name)

            # If no __all__, get class definitions
            if not file_exports:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not node.name.startswith("_") and node.name not in skip_names:
                            file_exports.append(node.name)

            if file_exports:
                exports[str(py_file)] = file_exports

        # Check rows subpackage
        rows_dir = datasets_dir / "rows"
        if rows_dir.exists():
            for py_file in rows_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    tree = ast.parse(py_file.read_text())
                except SyntaxError:
                    continue

                file_exports: list[str] = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not node.name.startswith("_") and node.name not in skip_names:
                            file_exports.append(node.name)

                if file_exports:
                    exports[str(py_file)] = file_exports

    return exports


def classify_export(name: str, source_file: str) -> str:
    """Classify an export by its type.

    Parameters
    ----------
    name
        Name of the export.
    source_file
        Source file path.

    Returns
    -------
    str
        Classification: "class", "function", "constant", "type_alias".
    """
    if name.endswith("Config") or name.endswith("Row") or name.endswith("Schema"):
        return "class"
    if name.endswith("StepConfig"):
        return "class"
    if name[0].isupper() and "_" not in name:
        return "class"
    if name.startswith("resolve_") or name.startswith("build_"):
        return "function"
    if name.isupper():
        return "constant"
    return "unknown"


def count_usages(name: str, include_tests: bool = False) -> tuple[int, int, int, list[str]]:
    """Count usages of a name across the codebase.

    Parameters
    ----------
    name
        Name to search for.
    include_tests
        Whether to include test files in external count.

    Returns
    -------
    tuple[int, int, int, list[str]]
        (external_uses, test_uses, internal_uses, files_list).
    """
    # Use ripgrep for fast searching - search for the name as a word
    try:
        result = subprocess.run(
            [
                "rg",
                "-l",  # Files only
                "--type=py",
                "-w",  # Word boundary
                name,
                "src/",
                "tests/",
            ],
            check=False, capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
    except FileNotFoundError:
        # Fallback to grep if rg not available
        result = subprocess.run(
            [
                "grep",
                "-rlw",  # Word boundary
                "--include=*.py",
                name,
                "src/",
                "tests/",
            ],
            check=False, capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        files = [f for f in result.stdout.strip().split("\n") if f]

    external_uses = 0
    test_uses = 0
    internal_uses = 0
    external_files: list[str] = []

    for f in files:
        if not f:
            continue
        # Config package files (including datasets subpackage)
        if "src/codeintel/config/" in f:
            internal_uses += 1
        elif f.startswith("tests/"):
            test_uses += 1
            if include_tests:
                external_uses += 1
                external_files.append(f)
        else:
            external_uses += 1
            external_files.append(f)

    return external_uses, test_uses, internal_uses, external_files


def analyze_config_usage(
    threshold: int = 1,
    include_tests: bool = False,
) -> AnalysisResult:
    """Analyze usage of all config exports.

    Parameters
    ----------
    threshold
        Flag items with fewer than this many external uses.
    include_tests
        Whether to include test files in external count.

    Returns
    -------
    AnalysisResult
        Complete analysis results.
    """
    exports_by_file = get_config_exports()
    all_exports: list[ExportInfo] = []
    seen_names: set[str] = set()

    for source_file, names in exports_by_file.items():
        for name in names:
            # Deduplicate - same name from different files
            if name in seen_names:
                continue
            seen_names.add(name)

            kind = classify_export(name, source_file)
            external, test, internal, files = count_usages(name, include_tests)

            info = ExportInfo(
                name=name,
                source_file=source_file,
                kind=kind,
                external_uses=external,
                test_uses=test,
                internal_uses=internal,
                used_in_files=files[:5],  # Limit to 5 files
            )
            all_exports.append(info)

    # Sort by external uses (ascending) so unused items are first
    all_exports.sort(key=lambda x: (x.external_uses, x.name))

    unused = sum(1 for e in all_exports if e.external_uses == 0)
    low_usage = sum(1 for e in all_exports if 0 < e.external_uses < threshold)

    return AnalysisResult(
        total_exports=len(all_exports),
        unused_count=unused,
        low_usage_count=low_usage,
        exports=all_exports,
    )


def format_text_report(result: AnalysisResult, threshold: int) -> str:
    """Format analysis result as text.

    Parameters
    ----------
    result
        Analysis result.
    threshold
        Threshold used for low-usage flagging.

    Returns
    -------
    str
        Formatted text report.
    """
    lines = [
        "=" * 80,
        "CONFIG PACKAGE USAGE ANALYSIS",
        "=" * 80,
        "",
        f"Total exports analyzed: {result.total_exports}",
        f"Potentially unused (0 external uses): {result.unused_count}",
        f"Low usage (<{threshold} external uses): {result.low_usage_count}",
        "",
        "-" * 80,
        "POTENTIALLY UNUSED EXPORTS (0 external uses)",
        "-" * 80,
        "",
    ]

    unused = [e for e in result.exports if e.external_uses == 0]
    if unused:
        for export in unused:
            lines.append(f"  ❌ {export.name}")
            lines.append(f"     Source: {export.source_file}")
            lines.append(f"     Kind: {export.kind}")
            lines.append(f"     Internal uses: {export.internal_uses}, Test uses: {export.test_uses}")
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    lines.extend([
        "-" * 80,
        f"LOW USAGE EXPORTS (1-{threshold-1} external uses)" if threshold > 1 else "LOW USAGE EXPORTS",
        "-" * 80,
        "",
    ])

    low_usage = [e for e in result.exports if 0 < e.external_uses < threshold]
    if low_usage:
        for export in low_usage:
            lines.append(f"  ⚠️  {export.name} ({export.external_uses} uses)")
            lines.append(f"     Source: {export.source_file}")
            if export.used_in_files:
                lines.append(f"     Used in: {', '.join(export.used_in_files[:3])}")
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    lines.extend([
        "-" * 80,
        "WELL-USED EXPORTS (by usage count)",
        "-" * 80,
        "",
    ])

    well_used = [e for e in result.exports if e.external_uses >= threshold]
    well_used.sort(key=lambda x: -x.external_uses)  # Descending
    for export in well_used[:20]:  # Top 20
        lines.append(f"  ✅ {export.name}: {export.external_uses} external uses")

    if len(well_used) > 20:
        lines.append(f"  ... and {len(well_used) - 20} more")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> int:
    """Run the analysis.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Analyze usage of config package exports."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Flag items with fewer than N external uses (default: 1)",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in external usage counts",
    )
    args = parser.parse_args()

    print("Analyzing config package usage...", file=sys.stderr)
    result = analyze_config_usage(
        threshold=args.threshold,
        include_tests=args.include_tests,
    )

    if args.json:
        # Convert to JSON-serializable dict
        data = {
            "total_exports": result.total_exports,
            "unused_count": result.unused_count,
            "low_usage_count": result.low_usage_count,
            "exports": [
                {
                    "name": e.name,
                    "source_file": e.source_file,
                    "kind": e.kind,
                    "external_uses": e.external_uses,
                    "test_uses": e.test_uses,
                    "internal_uses": e.internal_uses,
                    "used_in_files": e.used_in_files,
                }
                for e in result.exports
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(format_text_report(result, args.threshold))

    # Return non-zero if there are unused exports
    return 1 if result.unused_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
