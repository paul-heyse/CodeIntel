Love this—this is exactly the kind of “one artifact to feed the beast” that’s useful.

Below is a self-contained Python script you can drop into `mkdocs_gen/` (e.g. `mkdocs_gen/build_single_markdown.py`) that:

* Takes a fixed set of “summary/architecture” Markdown files from `mkdocs-build/docs/`
* Concatenates them into one big Markdown document
* Adds a **title + Table of Contents** at the top
* The TOC includes:

  * Section/subsection nesting based on `#`, `##`, `###`, etc.
  * Markdown navlinks (`[Title](#anchor)`)
  * **Final line numbers** for each heading in the *combined* file
* Skips headings inside fenced code blocks so we don’t pick up `#` inside code.

You can customize which files are included by editing the `INPUT_FILES` list near the top.

---

## Script: `mkdocs_gen/build_single_markdown.py`

````python
#!/usr/bin/env python
"""
Build a single combined Markdown overview document for CodeIntel.

- Reads selected Markdown files from `mkdocs-build/docs/`
- Concatenates them into one document, separated by horizontal rules
- Scans all headings (#, ##, ###, ...) outside code fences
- Prepends a Table of Contents with:
    - Nested bullets according to heading level
    - Markdown anchors like [Title](#title)
    - Final line numbers: (line N) where N is in the final combined file
- Writes the result to `CodeIntel_architecture_overview.md` at repo root.

Intended usage:
    python mkdocs_gen/build_single_markdown.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List


# --- Configuration ---------------------------------------------------------

# Repo root: mkdocs_gen/ lives at top-level of the repo
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where your MkDocs source docs live
DOCS_ROOT = REPO_ROOT / "mkdocs-build" / "docs"

# Output markdown file you can upload / paste into LLM chat
OUTPUT_PATH = REPO_ROOT / "CodeIntel_architecture_overview.md"

# List of Markdown files (relative to DOCS_ROOT) to merge, in order.
# Adjust this as your docs evolve.
INPUT_FILES = [
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
]


# --- Data structures ------------------------------------------------------


@dataclass
class Heading:
    level: int          # 1 for '#', 2 for '##', ...
    title: str          # Cleaned heading text
    anchor: str         # slugified anchor
    body_line_index: int  # 0-based index in the body (before TOC is prepended)


# --- Helpers --------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert heading text to a GitHub-style slug for anchors.

    Lowercase, strip non-alphanumerics except spaces/hyphens,
    collapse spaces to single '-', and strip leading/trailing '-'.
    """
    text = text.strip().lower()
    # Remove characters that are not alphanumeric, space, or hyphen
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    # Collapse whitespace to single dash
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def read_combined_body() -> List[str]:
    """Read and concatenate all input markdown files into one list of lines."""
    combined: List[str] = []

    for i, rel_path in enumerate(INPUT_FILES):
        path = DOCS_ROOT / rel_path
        if not path.exists():
            print(f"[build_single_markdown] WARNING: missing {path}, skipping")
            continue

        if combined:
            # Separate documents with a horizontal rule for clarity
            combined.append("")
            combined.append("---")
            combined.append("")

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        combined.extend(lines)

    return combined


def extract_headings(body_lines: List[str]) -> List[Heading]:
    """Scan body_lines for Markdown headings outside of fenced code blocks."""
    headings: List[Heading] = []
    in_code_block = False

    fence_re = re.compile(r"^(```|~~~)")
    heading_re = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

    for idx, line in enumerate(body_lines):
        stripped = line.strip()

        # Track fenced code blocks so we ignore headings inside them
        if fence_re.match(stripped):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        m = heading_re.match(line)
        if not m:
            continue

        hashes = m.group(1)
        raw_title = m.group(2)
        # Strip trailing '#' characters and whitespace from the title
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


# --- Main builder ---------------------------------------------------------


def build_combined_markdown() -> str:
    """Build the combined markdown string with title, TOC, and body."""
    body_lines = read_combined_body()

    if not body_lines:
        raise SystemExit("No input content found; check INPUT_FILES and DOCS_ROOT.")

    headings = extract_headings(body_lines)

    # Header + Table-of-contents preamble
    header_lines: List[str] = [
        "# CodeIntel – Combined architecture overview",
        "",
        "## Table of contents",
        "",
        # TOC bullet lines go here
    ]

    # First, compute TOC bullets without line numbers to know how many lines TOC uses
    toc_bullets_no_numbers: List[str] = []
    for h in headings:
        indent = "  " * (h.level - 1)  # indent by level
        bullet = f"{indent}- [{h.title}](#{h.anchor})"
        toc_bullets_no_numbers.append(bullet)

    # We'll also insert a blank line after the TOC bullets
    extra_blank_after_toc = 1

    # Offset = how many lines we prepend before the body starts
    offset = len(header_lines) + len(toc_bullets_no_numbers) + extra_blank_after_toc

    # Now rebuild the TOC bullets including final line numbers
    toc_lines: List[str] = []
    for h in headings:
        body_line_no = h.body_line_index + 1  # 1-based in the body only
        final_line_no = body_line_no + offset  # shifted by header + toc lines

        indent = "  " * (h.level - 1)
        bullet = f"{indent}- [{h.title}](#{h.anchor}) (line {final_line_no})"
        toc_lines.append(bullet)

    # Compose final document
    final_lines: List[str] = []
    final_lines.extend(header_lines)
    final_lines.extend(toc_lines)
    final_lines.append("")  # blank line after TOC
    final_lines.extend(body_lines)

    return "\n".join(final_lines) + "\n"


def main() -> None:
    output_text = build_combined_markdown()
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")
    print(f"[build_single_markdown] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
````

---

## How you’ll use it

1. Save that file as:

```text
mkdocs_gen/build_single_markdown.py
```

2. From repo root, run:

```bash
python mkdocs_gen/build_single_markdown.py
```

3. You’ll get:

```text
CodeIntel_architecture_overview.md
```

at the repo root, containing:

* A top-level title
* A Table of Contents with nested bullets and `(line N)` for each section/subsection
* All of your high-level docs concatenated in order

That file is what you can upload to ChatGPT / other LLMs as a single, rich “architecture brain dump” for the repo.

If you later add more overview docs, just append their paths to `INPUT_FILES` and rerun the script.
