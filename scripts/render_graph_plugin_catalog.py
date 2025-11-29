"""Render graph plugin catalog artifacts (JSON + Markdown + HTML)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
for path in (REPO_SRC, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codeintel.analytics.graphs.catalog import (  # noqa: E402
    build_plugin_catalog,
    write_plugin_catalog,
    write_plugin_catalog_html,
    write_plugin_catalog_markdown,
)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for catalog rendering.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate graph plugin catalog artifacts (JSON + Markdown + HTML)."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("build/plugin_catalog/catalog.json"),
        help="Path to write catalog JSON (default: build/plugin_catalog/catalog.json).",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=Path("docs/plugin_catalog.md"),
        help="Path to write Markdown catalog (default: docs/plugin_catalog.md).",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=Path("docs/plugin_catalog.html"),
        help="Path to write HTML catalog (default: docs/plugin_catalog.html).",
    )
    return parser.parse_args()


def main() -> int:
    """
    Generate catalog artifacts.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = parse_args()
    catalog = build_plugin_catalog()
    write_plugin_catalog(args.json_path)
    write_plugin_catalog_markdown(args.markdown_path, catalog)
    write_plugin_catalog_html(args.html_path, catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
