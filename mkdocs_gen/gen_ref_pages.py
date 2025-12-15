"""Generate API reference pages and navigation for codeintel modules.

This script is executed by the mkdocs-gen-files plugin during ``mkdocs build``.
It walks all Python modules under ``src/`` and generates virtual Markdown files
under ``reference/`` with mkdocstrings directives for automatic API documentation.

The script also builds a literate navigation file (``reference/SUMMARY.md``) that
mkdocs-literate-nav uses to structure the Code Reference section.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import mkdocs_gen_files

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Protocol

    class _Nav(Protocol):
        def __setitem__(self, key: tuple[str, ...], value: str) -> None: ...
        def build_literate_nav(self) -> Iterable[str]: ...


def _get_nav() -> _Nav:
    nav_type = getattr(mkdocs_gen_files, "Nav", None)
    if nav_type is None:
        message = "mkdocs_gen_files.Nav is not available"
        raise RuntimeError(message)
    return cast("_Nav", nav_type())


def generate_api_reference() -> None:
    """Generate API reference pages for all Python modules under src/.

    Walk the source tree, create virtual Markdown pages with mkdocstrings
    directives, and build literate navigation for the reference section.

    The function handles special cases:
    - ``__init__.py`` files map to ``index.md`` for section-index support
    - ``__main__.py`` files are skipped (script entrypoints)
    - Edit paths are mapped back to the real source files
    """
    root = Path(__file__).resolve().parent.parent
    src_root = root / "src"

    nav = _get_nav()

    for path in sorted(src_root.rglob("*.py")):
        module_path = path.relative_to(src_root).with_suffix("")
        doc_path = path.relative_to(src_root).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                continue
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"# `{ident}`\n\n")
            fd.write(f"::: {ident}\n")
            fd.write("    options:\n")
            fd.write("      heading_level: 2\n")
            fd.write("      show_root_heading: false\n")
            fd.write("      show_source: true\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


# Called unconditionally when the script is executed by mkdocs-gen-files plugin
generate_api_reference()
