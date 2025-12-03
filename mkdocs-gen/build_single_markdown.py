"""Build a single combined Markdown overview document for CodeIntel.

This script concatenates selected architecture documentation files into a single
Markdown document suitable for uploading to LLMs or other tools that benefit from
a unified context document.

Features
--------
- Reads selected Markdown files from ``mkdocs-build/docs/``
- Concatenates them into one document, separated by horizontal rules
- Scans all headings (#, ##, ###, ...) outside code fences
- Converts internal file links to anchor links where possible
- Removes broken links to content not in the combined document
- Prepends a Table of Contents with:
    - Nested bullets according to heading level
    - Markdown anchors like [Title](#title)
    - Final line numbers: (line N) where N is in the final combined file
- Writes the result to ``CodeIntel_architecture_overview.md`` at repo root

Usage
-----
From repo root::

    python mkdocs-gen/build_single_markdown.py

Or via Makefile::

    make docs-summary
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default configuration
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_ROOT = DEFAULT_REPO_ROOT / "mkdocs-build" / "docs"
DEFAULT_OUTPUT_PATH = DEFAULT_REPO_ROOT / "CodeIntel_architecture_overview.md"

DEFAULT_INPUT_FILES: tuple[str, ...] = (
    "index.md",
    "architecture/overview.md",
    "architecture/layering.md",
    "architecture/datasets-and-snapshots.md",
    "architecture/ingestion.md",
    "architecture/analytics.md",
    "architecture/graphs.md",
    "architecture/storage.md",
    "architecture/serving.md",
    "architecture/pipeline.md",
    "architecture/runtime.md",
)


@dataclass(frozen=True)
class Heading:
    """Represents a parsed Markdown heading.

    Attributes
    ----------
    level
        Heading level (1 for #, 2 for ##, etc.).
    title
        Cleaned heading text without leading/trailing hashes.
    anchor
        GitHub-style slugified anchor for linking.
    body_line_index
        0-based line index in the body before TOC is prepended.
    """

    level: int
    title: str
    anchor: str
    body_line_index: int


def slugify(text: str) -> str:
    """Convert heading text to a GitHub-style slug for anchors.

    Parameters
    ----------
    text
        The heading text to convert.

    Returns
    -------
    str
        Lowercase, hyphenated slug with special characters removed.

    Examples
    --------
    >>> slugify("Hello World")
    'hello-world'
    >>> slugify("Architecture & Design")
    'architecture--design'
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def read_combined_body(
    docs_root: Path,
    input_files: Sequence[str],
) -> list[str]:
    """Read and concatenate all input markdown files into one list of lines.

    Parameters
    ----------
    docs_root
        Root directory containing the documentation files.
    input_files
        Sequence of file paths relative to docs_root.

    Returns
    -------
    list[str]
        Combined lines from all input files, separated by horizontal rules.
    """
    combined: list[str] = []

    for rel_path in input_files:
        path = docs_root / rel_path
        if not path.exists():
            log.warning("Missing file %s, skipping", path)
            continue

        if combined:
            combined.append("")
            combined.append("---")
            combined.append("")

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        combined.extend(lines)

    return combined


def extract_headings(body_lines: Sequence[str]) -> list[Heading]:
    """Scan body lines for Markdown headings outside of fenced code blocks.

    Parameters
    ----------
    body_lines
        Lines of markdown text to scan.

    Returns
    -------
    list[Heading]
        List of detected headings with their positions and metadata.
    """
    headings: list[Heading] = []
    in_code_block = False

    fence_re = re.compile(r"^(```|~~~)")
    heading_re = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

    for idx, line in enumerate(body_lines):
        stripped = line.strip()

        if fence_re.match(stripped):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        match = heading_re.match(line)
        if not match:
            continue

        hashes = match.group(1)
        raw_title = match.group(2)
        title = re.sub(r"\s+#*$", "", raw_title).strip()
        level = len(hashes)
        anchor = slugify(title)

        headings.append(
            Heading(
                level=level,
                title=title,
                anchor=anchor,
                body_line_index=idx,
            )
        )

    return headings


def _resolve_link(
    text: str,
    url: str,
    valid_anchors: set[str],
) -> str:
    """Resolve a single markdown link to self-contained form.

    Parameters
    ----------
    text
        The link text.
    url
        The link URL/path.
    valid_anchors
        Set of valid anchor names.

    Returns
    -------
    str
        Resolved link or plain text if no valid anchor found.
    """
    # Keep external URLs
    if url.startswith(("http://", "https://", "mailto:")):
        return f"[{text}]({url})"

    # Handle internal anchor links
    if url.startswith("#"):
        anchor = url[1:]
        if anchor in valid_anchors:
            return f"[{text}]({url})"
        return text  # Just plain text, no extra bold

    # Handle file links - extract or derive anchor
    if "#" in url:
        _, anchor = url.rsplit("#", 1)
    else:
        anchor = slugify(text)

    # Check for matching anchor
    if anchor in valid_anchors:
        return f"[{text}](#{anchor})"

    # Try slugified link text
    text_anchor = slugify(text)
    if text_anchor in valid_anchors:
        return f"[{text}](#{text_anchor})"

    # No valid anchor - return plain text (surrounding formatting preserved)
    return text


def fix_links(
    body_lines: list[str],
    valid_anchors: set[str],
) -> list[str]:
    """Fix or remove links in the document body.

    Transforms links to be self-contained within the combined document:
    - Internal file links become anchor links if a matching heading exists
    - MkDocs cross-reference links are converted to plain text
    - External URLs (http/https) are preserved
    - Valid internal anchors are preserved
    - Image links to local files are converted to text descriptions

    Parameters
    ----------
    body_lines
        Lines of the document body to process.
    valid_anchors
        Set of valid anchor names from document headings.

    Returns
    -------
    list[str]
        Processed lines with fixed links.
    """
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    image_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    crossref_re = re.compile(r"\[([^\]]+)\]\[([^\]]*)\]")
    fence_re = re.compile(r"^(```|~~~)")

    result: list[str] = []
    in_code_block = False

    for original_line in body_lines:
        stripped = original_line.strip()

        if fence_re.match(stripped):
            in_code_block = not in_code_block
            result.append(original_line)
            continue

        if in_code_block:
            result.append(original_line)
            continue

        # Process image links first: ![alt](path)
        # Keep external images, convert local to text description
        def replace_image(match: re.Match[str]) -> str:
            alt = match.group(1)
            url = match.group(2)
            if url.startswith(("http://", "https://")):
                return match.group(0)  # Keep external images
            # Convert local image to text description
            return f"*[Image: {alt}]*" if alt else "*[Image]*"

        processed = image_re.sub(replace_image, original_line)

        # Process MkDocs cross-references: [text][ref] -> text
        processed = crossref_re.sub(r"\1", processed)

        # Process standard markdown links
        def replace_link(match: re.Match[str]) -> str:
            return _resolve_link(match.group(1), match.group(2), valid_anchors)

        processed = link_re.sub(replace_link, processed)
        result.append(processed)

    return result


def _build_toc_lines(headings: list[Heading], offset: int) -> list[str]:
    """Build table of contents lines with line number references.

    Parameters
    ----------
    headings
        List of headings to include in TOC.
    offset
        Line offset to add for final line numbers.

    Returns
    -------
    list[str]
        TOC bullet lines with anchors and line numbers.
    """
    toc_lines: list[str] = []
    for heading in headings:
        final_line_no = heading.body_line_index + 1 + offset
        indent = "  " * (heading.level - 1)
        bullet = f"{indent}- [{heading.title}](#{heading.anchor}) (line {final_line_no})"
        toc_lines.append(bullet)
    return toc_lines


def build_combined_markdown(
    docs_root: Path | None = None,
    input_files: Sequence[str] | None = None,
) -> str:
    """Build the combined markdown string with title, TOC, and body.

    Parameters
    ----------
    docs_root
        Root directory containing documentation files. Defaults to
        ``mkdocs-build/docs/`` relative to repo root.
    input_files
        Sequence of file paths relative to docs_root. Defaults to
        the standard architecture documentation files.

    Returns
    -------
    str
        Complete markdown document with TOC and line numbers.

    Raises
    ------
    ValueError
        If no input content is found.
    """
    if docs_root is None:
        docs_root = DEFAULT_DOCS_ROOT
    if input_files is None:
        input_files = DEFAULT_INPUT_FILES

    body_lines = read_combined_body(docs_root, input_files)

    if not body_lines:
        msg = "No input content found; check input_files and docs_root."
        raise ValueError(msg)

    # Extract headings and fix links
    headings = extract_headings(body_lines)
    valid_anchors = {h.anchor for h in headings}
    body_lines = fix_links(body_lines, valid_anchors)

    # Build header
    header_lines: list[str] = [
        "# CodeIntel - Combined architecture overview",
        "",
        "## Table of contents",
        "",
    ]

    # Calculate offset for line numbers (header + TOC + blank line)
    toc_entry_count = len(headings)
    offset = len(header_lines) + toc_entry_count + 1

    # Build TOC and assemble final document
    toc_lines = _build_toc_lines(headings, offset)

    return "\n".join(header_lines + toc_lines + [""] + body_lines) + "\n"


def main(
    output_path: Path | None = None,
    docs_root: Path | None = None,
    input_files: Sequence[str] | None = None,
) -> None:
    """Generate the combined architecture overview document.

    Parameters
    ----------
    output_path
        Path to write the output file. Defaults to
        ``CodeIntel_architecture_overview.md`` at repo root.
    docs_root
        Root directory containing documentation files.
    input_files
        Sequence of file paths relative to docs_root.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    output_text = build_combined_markdown(docs_root, input_files)
    output_path.write_text(output_text, encoding="utf-8")
    log.info("Wrote combined overview to %s", output_path)


# Called when script is executed directly or via mkdocs-gen-files
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    main()
