"""Pytest fixtures for MkDocs generation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_docs_root(tmp_path: Path) -> Path:
    """Create a sample docs root with test markdown files.

    Creates a directory structure mimicking mkdocs-build/docs/ with sample
    markdown files for testing the combined markdown generator.

    Returns
    -------
    Path
        Path to the temporary docs root directory.
    """
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    # Create index.md
    (docs_root / "index.md").write_text(
        "# Welcome\n\nThis is the index page.\n\n## Getting Started\n\nStart here.\n",
        encoding="utf-8",
    )

    # Create architecture directory
    arch_dir = docs_root / "architecture"
    arch_dir.mkdir()

    # Create overview.md
    (arch_dir / "overview.md").write_text(
        "# Architecture Overview\n\n## Subsystems\n\nDescription of subsystems.\n\n"
        "### Analytics\n\nAnalytics subsystem.\n",
        encoding="utf-8",
    )

    # Create layering.md
    (arch_dir / "layering.md").write_text(
        "# Layering\n\n## Layer Rules\n\nImport rules go here.\n",
        encoding="utf-8",
    )

    return docs_root


@pytest.fixture
def sample_docs_with_code_fence(tmp_path: Path) -> Path:
    """Create docs with code fences containing heading-like content.

    Tests that headings inside code blocks are properly ignored.

    Returns
    -------
    Path
        Path to the temporary docs root directory.
    """
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    content = """# Real Heading

Some text here.

```python
# This is a comment, not a heading
def example():
    pass
```

## Another Real Heading

More text.

~~~markdown
# Heading inside tilde fence
~~~

### Final Heading
"""
    (docs_root / "test.md").write_text(content, encoding="utf-8")

    return docs_root


@pytest.fixture
def sample_source_package(tmp_path: Path) -> Path:
    """Create a sample Python package for testing API reference generation.

    Returns
    -------
    Path
        Path to the src directory containing the package.
    """
    src_root = tmp_path / "src"
    src_root.mkdir()

    # Create a sample package
    pkg_dir = src_root / "sample_pkg"
    pkg_dir.mkdir()

    # __init__.py
    (pkg_dir / "__init__.py").write_text(
        '"""Sample package for testing."""\n',
        encoding="utf-8",
    )

    # module.py
    (pkg_dir / "module.py").write_text(
        '"""Sample module.\n\nThis module does things.\n"""\n\n'
        "def hello() -> str:\n"
        '    """Return greeting."""\n'
        '    return "hello"\n',
        encoding="utf-8",
    )

    # subpackage
    sub_dir = pkg_dir / "sub"
    sub_dir.mkdir()

    (sub_dir / "__init__.py").write_text(
        '"""Subpackage."""\n',
        encoding="utf-8",
    )

    (sub_dir / "utils.py").write_text(
        '"""Utility functions."""\n\n'
        "def helper() -> int:\n"
        '    """Return a number."""\n'
        "    return 42\n",
        encoding="utf-8",
    )

    # __main__.py (should be skipped)
    (pkg_dir / "__main__.py").write_text(
        '"""Main entry point."""\nif __name__ == "__main__":\n    pass\n',
        encoding="utf-8",
    )

    return src_root
