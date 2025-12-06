"""Analyze usage of config package exports across the codebase."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path("src/codeintel/config")
DATASETS_DIR = CONFIG_DIR / "datasets"
ROWS_DIR = DATASETS_DIR / "rows"
SEARCH_ROOTS: tuple[Path, ...] = (Path("src"), Path("tests"))
MIN_EXPORT_NAME_LENGTH = 4
MAX_USAGE_FILE_COUNT = 5
WELL_USED_REPORT_LIMIT = 20
WORD_BOUNDARY_TEMPLATE = r"\b{}\b"
SKIP_NAMES: set[str] = {
    "repo",
    "commit",
    "repo_root",
    "build_dir",
    "snapshot",
    "paths",
    "default",
    "analytics",
    "graphs",
    "ingestion",
    "profiles",
    "code",
    "config",
    "from_args",
    "from_layout",
    "from_primitives",
}


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


def _parse_python_file(py_file: Path) -> ast.Module | None:
    try:
        return ast.parse(py_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _extract_all_exports(tree: ast.AST) -> list[str]:
    exports: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple))):
            continue
        is_all_assignment = any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        )
        if not is_all_assignment:
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                export_name = element.value
                if export_name not in SKIP_NAMES and len(export_name) >= MIN_EXPORT_NAME_LENGTH:
                    exports.append(export_name)
    return exports


def _extract_class_exports(tree: ast.AST) -> list[str]:
    return [
        node.name
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ClassDef)
            and not node.name.startswith("_")
            and node.name not in SKIP_NAMES
        )
    ]


def _collect_exports_from_file(py_file: Path) -> list[str]:
    tree = _parse_python_file(py_file)
    if tree is None:
        return []

    explicit_exports = _extract_all_exports(tree)
    if explicit_exports:
        return explicit_exports

    return _extract_class_exports(tree)


def _iter_config_modules() -> list[Path]:
    return [
        py_file
        for directory in (CONFIG_DIR, DATASETS_DIR, ROWS_DIR)
        if directory.exists()
        for py_file in directory.glob("*.py")
        if not py_file.name.startswith("_")
    ]


def get_config_exports() -> dict[str, list[str]]:
    """Extract all exports from config package modules.

    Returns
    -------
    dict[str, list[str]]
        Mapping of source file to list of exported names.
    """
    exports: dict[str, list[str]] = {}
    for py_file in _iter_config_modules():
        file_exports = _collect_exports_from_file(py_file)
        if file_exports:
            exports[str(py_file)] = file_exports
    return exports


def classify_export(name: str) -> str:
    """Classify an export by its type.

    Parameters
    ----------
    name
        Name of the export.

    Returns
    -------
    str
        Classification: "class", "function", "constant", "type_alias".
    """
    if name.endswith(("Config", "Row", "Schema")):
        return "class"
    if name[0].isupper() and "_" not in name:
        return "class"
    if name.startswith(("resolve_", "build_")):
        return "function"
    if name.isupper():
        return "constant"
    return "unknown"


def _file_contains_pattern(path: Path, pattern: re.Pattern[str]) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(pattern.search(content))


def count_usages(
    name: str,
    *,
    include_tests: bool = False,
) -> tuple[int, int, int, list[str]]:
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
    pattern = re.compile(WORD_BOUNDARY_TEMPLATE.format(re.escape(name)))
    external_uses = 0
    test_uses = 0
    internal_uses = 0
    external_files: list[str] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not _file_contains_pattern(path, pattern):
                continue

            if path.is_relative_to(CONFIG_DIR):
                internal_uses += 1
                continue

            if path.is_relative_to(Path("tests")):
                test_uses += 1
                if include_tests:
                    if len(external_files) < MAX_USAGE_FILE_COUNT:
                        external_files.append(path.as_posix())
                    external_uses += 1
                continue

            if len(external_files) < MAX_USAGE_FILE_COUNT:
                external_files.append(path.as_posix())
            external_uses += 1

    return external_uses, test_uses, internal_uses, external_files


def analyze_config_usage(
    threshold: int = 1,
    *,
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

            kind = classify_export(name)
            external, test, internal, files = count_usages(name, include_tests=include_tests)

            info = ExportInfo(
                name=name,
                source_file=source_file,
                kind=kind,
                external_uses=external,
                test_uses=test,
                internal_uses=internal,
                used_in_files=files[:MAX_USAGE_FILE_COUNT],
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
            lines.append(
                f"     Internal uses: {export.internal_uses}, Test uses: {export.test_uses}"
            )
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    lines.extend(
        [
            "-" * 80,
            f"LOW USAGE EXPORTS (1-{threshold - 1} external uses)"
            if threshold > 1
            else "LOW USAGE EXPORTS",
            "-" * 80,
            "",
        ]
    )

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

    lines.extend(
        [
            "-" * 80,
            "WELL-USED EXPORTS (by usage count)",
            "-" * 80,
            "",
        ]
    )

    well_used = [e for e in result.exports if e.external_uses >= threshold]
    well_used.sort(key=lambda x: -x.external_uses)  # Descending
    lines.extend(
        f"  ✅ {export.name}: {export.external_uses} external uses"
        for export in well_used[:WELL_USED_REPORT_LIMIT]
    )

    if len(well_used) > WELL_USED_REPORT_LIMIT:
        remaining = len(well_used) - WELL_USED_REPORT_LIMIT
        lines.append(f"  ... and {remaining} more")

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
    parser = argparse.ArgumentParser(description="Analyze usage of config package exports.")
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

    sys.stderr.write("Analyzing config package usage...\n")
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
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text_report(result, args.threshold))
        sys.stdout.write("\n")

    # Return non-zero if there are unused exports
    return 1 if result.unused_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
