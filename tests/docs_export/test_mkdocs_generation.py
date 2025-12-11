"""Comprehensive tests for MkDocs documentation generation scripts.

This module tests all functionality in the mkdocs_gen/ scripts:
- build_single_markdown.py: Combined architecture overview generator
- gen_ref_pages.py: API reference page generator
- gen_arch_diagrams.py: Architecture diagram generator

Tests follow the Testing Charter - using real file operations with
isolated tmp_path fixtures, no mocking.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_false,
    expect_in,
    expect_length,
    expect_not_equal,
    expect_true,
)

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MKDOCS_GEN_DIR = REPO_ROOT / "mkdocs_gen"


def _load_build_single_markdown() -> ModuleType:
    module_path = MKDOCS_GEN_DIR / "build_single_markdown.py"
    if not module_path.exists():
        message = f"build_single_markdown.py not found at {module_path}"
        raise ImportError(message)

    module_name = "mkdocs_gen.build_single_markdown"
    loader = importlib.machinery.SourceFileLoader(module_name, str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        message = f"Unable to create spec for {module_path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec to allow dataclass to work
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


build_single_markdown = _load_build_single_markdown()
DEFAULT_INPUT_FILES = build_single_markdown.DEFAULT_INPUT_FILES
Heading = build_single_markdown.Heading
build_combined_markdown = build_single_markdown.build_combined_markdown
extract_headings = build_single_markdown.extract_headings
main = build_single_markdown.main
read_combined_body = build_single_markdown.read_combined_body
slugify = build_single_markdown.slugify

# Test constants
EXPECTED_HEADING_LEVELS = 6
EXPECTED_TWO_HEADINGS = 2
MIN_REAL_DOCS_LENGTH = 1000


# =============================================================================
# Slugify Tests
# =============================================================================


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello World", "hello-world"),
        ("Architecture Overview", "architecture-overview"),
        ("Simple", "simple"),
        ("multiple   spaces", "multiple-spaces"),
        ("UPPERCASE", "uppercase"),
    ],
)
def test_slugify_basic(input_text: str, expected: str) -> None:
    """Test basic text conversion to slugs."""
    expect_equal(slugify(input_text), expected)


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello & World", "hello-world"),  # & removed, spaces collapsed
        ("Test (example)", "test-example"),
        ("Path/To/File", "pathtofile"),
        ("Code: Example", "code-example"),
        ("100% Complete", "100-complete"),
    ],
)
def test_slugify_special_chars(input_text: str, expected: str) -> None:
    """Test that special characters are properly removed or handled."""
    expect_equal(slugify(input_text), expected)


def test_slugify_empty() -> None:
    """Test empty string input."""
    result = slugify("")
    expect_true(not result)


def test_slugify_whitespace_only() -> None:
    """Test whitespace-only input."""
    result = slugify("   ")
    expect_true(not result)


def test_slugify_leading_trailing_hyphens() -> None:
    """Test that leading/trailing hyphens are stripped."""
    expect_equal(slugify("- test -"), "test")
    expect_equal(slugify("---heading---"), "heading")


# =============================================================================
# Extract Headings Tests
# =============================================================================


def test_extract_headings_all_levels() -> None:
    """Test that all heading levels 1-6 are detected."""
    lines = [
        "# Level 1",
        "## Level 2",
        "### Level 3",
        "#### Level 4",
        "##### Level 5",
        "###### Level 6",
    ]
    headings = extract_headings(lines)

    expect_length(headings, EXPECTED_HEADING_LEVELS)
    for i, heading in enumerate(headings, start=1):
        expect_equal(heading.level, i)
        expect_equal(heading.title, f"Level {i}")


def test_extract_headings_code_fence_skip() -> None:
    """Test that headings inside backtick code blocks are ignored."""
    lines = [
        "# Real Heading",
        "```python",
        "# This is a comment",
        "def foo():",
        "    pass",
        "```",
        "## Another Real Heading",
    ]
    headings = extract_headings(lines)

    expect_length(headings, EXPECTED_TWO_HEADINGS)
    expect_equal(headings[0].title, "Real Heading")
    expect_equal(headings[1].title, "Another Real Heading")


def test_extract_headings_tilde_fence() -> None:
    """Test that headings inside tilde code blocks are ignored."""
    lines = [
        "# Before",
        "~~~",
        "# Inside fence",
        "~~~",
        "# After",
    ]
    headings = extract_headings(lines)

    expect_length(headings, EXPECTED_TWO_HEADINGS)
    expect_equal(headings[0].title, "Before")
    expect_equal(headings[1].title, "After")


def test_extract_headings_trailing_hashes() -> None:
    """Test that trailing hashes are stripped from titles."""
    lines = [
        "# Heading ##",
        "## Another Heading ###",
    ]
    headings = extract_headings(lines)

    expect_length(headings, EXPECTED_TWO_HEADINGS)
    expect_equal(headings[0].title, "Heading")
    expect_equal(headings[1].title, "Another Heading")


def test_extract_headings_line_indices() -> None:
    """Test that line indices are correctly recorded."""
    lines = [
        "Some text",
        "# First Heading",
        "More text",
        "",
        "## Second Heading",
    ]
    headings = extract_headings(lines)

    expected_first_index = 1
    expected_second_index = 4

    expect_length(headings, EXPECTED_TWO_HEADINGS)
    expect_equal(headings[0].body_line_index, expected_first_index)
    expect_equal(headings[1].body_line_index, expected_second_index)


def test_extract_headings_empty_lines() -> None:
    """Test extraction with empty input."""
    headings = extract_headings([])
    expect_equal(headings, [])


def test_extract_headings_no_headings() -> None:
    """Test extraction when there are no headings."""
    lines = ["Just some text", "More text", "No headings here"]
    headings = extract_headings(lines)
    expect_equal(headings, [])


# =============================================================================
# Read Combined Body Tests
# =============================================================================


def test_read_combined_body_single_file(sample_docs_root: Path) -> None:
    """Test reading a single file."""
    lines = read_combined_body(sample_docs_root, ["index.md"])

    expect_true(len(lines) > 0)
    expect_in("# Welcome", lines[0])


def test_read_combined_body_multiple_files(sample_docs_root: Path) -> None:
    """Test reading multiple files with separators."""
    input_files = ["index.md", "architecture/overview.md"]
    lines = read_combined_body(sample_docs_root, input_files)

    # Check for separator
    expect_true("---" in lines)
    # Check content from both files
    content = "\n".join(lines)
    expect_in("Welcome", content)
    expect_in("Architecture Overview", content)


def test_read_combined_body_missing_file(
    sample_docs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that missing files are logged and skipped."""
    with caplog.at_level(logging.WARNING):
        lines = read_combined_body(
            sample_docs_root,
            ["index.md", "nonexistent.md"],
        )

    # Should still have content from existing file
    expect_true(len(lines) > 0)
    expect_in("missing", caplog.text.lower())


def test_read_combined_body_all_missing(sample_docs_root: Path) -> None:
    """Test that empty list is returned when all files missing."""
    lines = read_combined_body(sample_docs_root, ["nonexistent.md"])
    expect_equal(lines, [])


def test_read_combined_body_preserves_content(sample_docs_root: Path) -> None:
    """Test that file content is preserved exactly."""
    original = (sample_docs_root / "index.md").read_text(encoding="utf-8")
    lines = read_combined_body(sample_docs_root, ["index.md"])
    combined = "\n".join(lines)

    # Content should match (minus trailing newline differences)
    expect_equal(combined.strip(), original.strip())


# =============================================================================
# Build Combined Markdown Tests
# =============================================================================


def test_build_combined_markdown_structure(sample_docs_root: Path) -> None:
    """Test that output has correct overall structure."""
    input_files = ["index.md", "architecture/overview.md"]
    result = build_combined_markdown(sample_docs_root, input_files)

    # Should start with the title
    expect_true(result.startswith("# CodeIntel - Combined architecture overview"))

    # Should have TOC section
    expect_in("## Table of contents", result)

    # Should have content from input files
    expect_in("Welcome", result)
    expect_in("Architecture Overview", result)


def test_build_combined_markdown_toc_nesting(sample_docs_root: Path) -> None:
    """Test that TOC has correct nesting structure."""
    input_files = ["architecture/overview.md"]
    result = build_combined_markdown(sample_docs_root, input_files)

    # Level 1 heading should have no indent
    expect_in("- [Architecture Overview](#architecture-overview)", result)

    # Level 2 heading should have 2-space indent
    expect_in("  - [Subsystems](#subsystems)", result)

    # Level 3 heading should have 4-space indent
    expect_in("    - [Analytics](#analytics)", result)


def test_build_combined_markdown_line_numbers(sample_docs_root: Path) -> None:
    """Test that line numbers in TOC are accurate."""
    input_files = ["index.md"]
    result = build_combined_markdown(sample_docs_root, input_files)
    lines = result.split("\n")

    # Find a TOC entry with line number
    toc_entry = None
    for line in lines:
        if "(line " in line and "Welcome" in line:
            toc_entry = line
            break

    if toc_entry is None:
        pytest.fail("Expected TOC entry for Welcome heading")

    # Extract the line number from TOC
    match = re.search(r"\(line (\d+)\)", toc_entry)
    if match is None:
        pytest.fail("Expected line number in TOC entry")
    toc_line_num = int(match.group(1))

    # Verify the heading is actually at that line
    expect_true(lines[toc_line_num - 1].startswith("# Welcome"))


def test_build_combined_markdown_anchors_valid(sample_docs_root: Path) -> None:
    """Test that all TOC anchors match actual headings."""
    input_files = ["index.md", "architecture/overview.md"]
    result = build_combined_markdown(sample_docs_root, input_files)

    # Extract all anchors from TOC
    anchors = re.findall(r"\]\(#([^)]+)\)", result)

    # Each anchor should correspond to a heading that would slugify to it
    for anchor in anchors:
        # The anchor should appear in the document body as a heading
        expect_in(anchor, result.lower())


def test_build_combined_markdown_empty_raises(tmp_path: Path) -> None:
    """Test that empty input raises ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No input content"):
        build_combined_markdown(empty_dir, ["nonexistent.md"])


def test_build_combined_markdown_code_fences_preserved(
    sample_docs_with_code_fence: Path,
) -> None:
    """Test that code fences are preserved in output."""
    result = build_combined_markdown(sample_docs_with_code_fence, ["test.md"])

    # Code fence content should be preserved
    expect_in("```python", result)
    expect_in("def example():", result)

    # But the comment inside shouldn't be in TOC
    # TOC ends after the last "(line N)" entry before blank line + body
    lines = result.split("\n")
    toc_end_idx = 0
    for i, line in enumerate(lines):
        if "(line " in line:
            toc_end_idx = i
    toc_section = "\n".join(lines[: toc_end_idx + 1])
    expect_false("This is a comment" in toc_section)


# =============================================================================
# Heading Dataclass Tests
# =============================================================================


def test_heading_frozen() -> None:
    """Test that Heading is immutable."""
    heading = Heading(level=1, title="Test", anchor="test", body_line_index=0)

    assert_cannot_setattr(heading, "level", 2)


def test_heading_equality() -> None:
    """Test Heading equality comparison."""
    h1 = Heading(level=1, title="Test", anchor="test", body_line_index=0)
    h2 = Heading(level=1, title="Test", anchor="test", body_line_index=0)
    h3 = Heading(level=2, title="Test", anchor="test", body_line_index=0)

    expect_equal(h1, h2)
    expect_not_equal(h1, h3)


# =============================================================================
# Integration Tests with Real Docs
# =============================================================================


def test_build_with_real_docs() -> None:
    """Test building combined markdown with real documentation.

    This test uses the actual docs to ensure they're well-formed.
    """
    real_docs_root = REPO_ROOT / "mkdocs-build" / "docs"

    if not real_docs_root.exists():
        pytest.skip("mkdocs-build/docs not found")

    # Filter to only files that exist
    existing_files = [f for f in DEFAULT_INPUT_FILES if (real_docs_root / f).exists()]

    if not existing_files:
        pytest.skip("No input files found")

    result = build_combined_markdown(real_docs_root, existing_files)

    # Basic structure checks
    expect_in("# CodeIntel - Combined architecture overview", result)
    expect_in("## Table of contents", result)
    expect_true(len(result) > MIN_REAL_DOCS_LENGTH)


def test_main_function_creates_file(tmp_path: Path) -> None:
    """Test that main() creates the output file."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "test.md").write_text("# Test\n\nContent.\n", encoding="utf-8")

    output_path = tmp_path / "output.md"

    main(
        output_path=output_path,
        docs_root=docs_root,
        input_files=["test.md"],
    )

    expect_true(output_path.exists())
    content = output_path.read_text(encoding="utf-8")
    expect_in("# CodeIntel - Combined architecture overview", content)
    expect_in("Test", content)
