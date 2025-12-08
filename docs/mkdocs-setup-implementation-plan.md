Nice, you’ve already done the hard work (strict typing + NumPy docstrings + linting). Let’s wire up the “best in class” MkDocs stack in a way that’s super friendly for LLM programmer agents and keeps everything cleanly separated into:

* `mkdocs_gen/` – generator scripts (Python)
* `mkdocs-build/` – config + Markdown sources
* `mkdocs-output/` – built static site

I’ll give you:

1. Target layout and assumptions
2. Exact `mkdocs-build/mkdocs.yml`
3. Docs skeleton under `mkdocs-build/docs/`
4. `mkdocs_gen/gen_ref_pages.py` (auto API docs + nav)
5. Optional `mkdocs_gen/gen_arch_diagrams.py` (pydeps + pyreverse)
6. How to build / serve

You can hand this straight to a coding agent as a to-do list.

---

## 0. Target layout & assumptions

**Repo layout (after this change):**

```text
repo-root/
  mkdocs_gen/              # new – Python helper scripts for docs
    gen_ref_pages.py
    gen_arch_diagrams.py   # (optional, for diagrams)
  mkdocs-build/            # new – mkdocs config + markdown docs
    mkdocs.yml
    docs/
      index.md
      architecture/
        overview.md
        layering.md
        ingestion.md
        analytics.md
        graphs.md
        storage.md
        serving.md
        pipeline.md
        runtime.md
        datasets-and-snapshots.md
      # reference/ will be generated at build time
  mkdocs-output/           # new – mkdocs site_dir (build output)
  src/
    codeintel/
      analytics/
      config/
      core/
      graphs/
      ingestion/
      pipeline/
      runtime/
      serving/
      storage/
  tests/
  pyproject.toml
  ...
```

**Assumptions:**

* Your Python package is in `src/codeintel` (per the zips you attached).
* The mkdocs/docs stack is already installed in your environment (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-gen-files`, `mkdocs-literate-nav`, `mkdocs-section-index`, `mkdocs-autorefs`, `pymdown-extensions`, `markdown-exec`). ([PyPI][1])
* You installed `pydeps`, `pylint` (for `pyreverse`) and system `graphviz`. ([PyPI][2])

---

## 1. (Optional) pyproject extras for docs

If you want a clean “docs” extra (nice for tooling/CI), add this to `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
  "mkdocs",
  "mkdocs-material",
  "mkdocstrings[python]",
  "mkdocs-gen-files",
  "mkdocs-literate-nav",
  "mkdocs-section-index",
  "mkdocs-autorefs",
  "pymdown-extensions",
  "markdown-exec",
  "pydeps",
  "pylint",
  "graphviz",
]
```

Then:

```bash
uv pip install ".[docs]"
# or
pip install ".[docs]"
```

---

## 2. Create `mkdocs-build/mkdocs.yml`

> **Task for agent:** create file `mkdocs-build/mkdocs.yml` with the following content.

This config:

* Uses Material theme with good navigation. ([squidfunk.github.io][3])
* Sets `docs_dir` to `mkdocs-build/docs`. ([MkDocs][4])
* Sets `site_dir` to `mkdocs-output`.
* Auto-generates reference pages for everything under `src/` via mkdocs-gen-files + mkdocstrings recipes. ([mkdocstrings][5])
* Uses NumPy docstring parsing in mkdocstrings. ([mkdocstrings][6])

```yaml
site_name: CodeIntel
site_url: https://example.com/codeintel-docs
repo_url: https://github.com/your-org/codeintel
repo_name: your-org/codeintel

# docs live inside mkdocs-build/docs
docs_dir: docs

# built site goes to mkdocs-output at repo root
site_dir: ../mkdocs-output

theme:
  name: material
  language: en
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.tabs
    - navigation.indexes
    - content.code.copy
    - search.suggest
    - search.highlight
    - search.share

markdown_extensions:
  - toc:
      permalink: "#"
  - admonition
  - attr_list
  - def_list
  - footnotes

  # Pymdown extensions (for tabs, details, fenced code, etc.) :contentReference[oaicite:6]{index=6}
  - pymdownx.superfences
  - pymdownx.highlight:
      default_lang: python
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg

nav:
  - Overview: index.md
  - Architecture:
      - architecture/overview.md
      - architecture/layering.md
      - architecture/datasets-and-snapshots.md
      - architecture/ingestion.md
      - architecture/analytics.md
      - architecture/graphs.md
      - architecture/storage.md
      - architecture/serving.md
      - architecture/pipeline.md
      - architecture/runtime.md

  # Code reference nav is generated automatically in docs/reference/SUMMARY.md
  - Code Reference: reference/   # note trailing slash 

plugins:
  - search    # keep search enabled :contentReference[oaicite:8]{index=8}

  # Generate API reference pages + literate nav from src/ at build time :contentReference[oaicite:9]{index=9}
  - gen-files:
      scripts:
        # path is relative to this mkdocs.yml file
        - ../mkdocs_gen/gen_ref_pages.py

  # Use SUMMARY.md files for nav in subtrees (e.g. docs/reference/) :contentReference[oaicite:10]{index=10}
  - literate-nav:
      nav_file: SUMMARY.md

  # Make section headings clickable and show __init__ docs as section index :contentReference[oaicite:11]{index=11}
  - section-index

  # Cross-page links by heading title like [Overview][], [Datasets and snapshots][] :contentReference[oaicite:12]{index=12}
  - autorefs

  # Auto API docs from Python code :contentReference[oaicite:13]{index=13}
  - mkdocstrings:
      watch:
        - ../src                          # rebuild when code changes :contentReference[oaicite:14]{index=14}
      default_handler: python
      handlers:
        python:
          # where to look for modules; config is in mkdocs-build, code in ../src :contentReference[oaicite:15]{index=15}
          paths:
            - ../src

          # cross-ref external docs (stdlib as baseline; add others as needed) :contentReference[oaicite:16]{index=16}
          inventories:
            - https://docs.python.org/3/objects.inv

          options:
            # docstring / type handling (you already use NumPy style) :contentReference[oaicite:17]{index=17}
            docstring_style: numpy
            docstring_section_style: table
            show_signature: true
            show_signature_annotations: true
            modernize_annotations: true
            unwrap_annotated: true

            # head / TOC behavior
            show_root_heading: true
            show_root_full_path: true
            show_root_toc_entry: true
            show_symbol_type_heading: true
            heading_level: 2

            # member selection & order
            members_order: source
            inherited_members: true
            filters:
              - "!^_"

            # docstring sections – keep all, good for LLMs
            show_docstring_description: true
            show_docstring_parameters: true
            show_docstring_returns: true
            show_docstring_raises: true
            show_docstring_attributes: true
            show_docstring_examples: true

            show_if_no_docstring: false

  # Optional: run small scripts in code blocks (e.g. CLI --help) :contentReference[oaicite:18]{index=18}
  - markdown-exec
```

---

## 3. Create docs skeleton under `mkdocs-build/docs`

> **Task for agent:** create the following directory structure and minimal Markdown files.

### 3.1 Directories

```bash
mkdir -p mkdocs-build/docs/architecture
mkdir -p mkdocs-build/docs/reference   # empty; will be populated virtually
```

### 3.2 `mkdocs-build/docs/index.md`

```markdown
# CodeIntel

High-level description of CodeIntel.

- **Purpose**: code intelligence / codebase analytics for Python repositories.
- **Key subsystems**:
  - `codeintel.ingestion` – build snapshots & datasets
  - `codeintel.analytics` – analytics runtime, metrics, profiles
  - `codeintel.graphs` – graph engines (call graph, import graph, CFG/DFG, etc.)
  - `codeintel.storage` – DuckDB, Parquet, dataset contracts
  - `codeintel.serving` – backend services & APIs
  - `codeintel.pipeline` – unified orchestration
  - `codeintel.runtime` – runtime / environment glue
  - `codeintel.core` and `codeintel.config` – primitives, contracts, base abstractions

See [Architecture overview][] for the big picture, and [Code Reference][] for the full API.
```

(Those last two links are resolved by `mkdocs-autorefs` via headings named “Architecture overview” and “Code Reference”. ([PyPI][7]))

### 3.3 Architecture pages (stubs)

For each of these, create the file under `mkdocs-build/docs/architecture/` with a minimal skeleton (the content is just a starting point):

#### `overview.md`

````markdown
# Architecture overview

> **Goal for this page (for humans + LLMs)**  
> 1. Show the major subsystems (ingestion, analytics, graphs, storage, serving, pipeline, runtime, core, config).  
> 2. Show the primary dataflow and control-flow.  
> 3. Link to subsystem-specific pages for details.

## High-level dataflow

```mermaid
graph LR
  Ingestion["Ingestion (codeintel.ingestion)"] --> Storage["Storage (codeintel.storage)"]
  Storage --> Analytics["Analytics (codeintel.analytics)"]
  Analytics --> Graphs["Graphs (codeintel.graphs)"]
  Analytics --> Serving["Serving (codeintel.serving)"]
  Graphs --> Serving
  Serving --> Pipeline["Pipeline (codeintel.pipeline)"]
  Pipeline --> Clients["Clients / Agents / Tools"]
````

## Subsystems

* [Layering rules][Layering and boundaries]
* [Ingestion](ingestion.md)
* [Analytics](analytics.md)
* [Graphs](graphs.md)
* [Storage](storage.md)
* [Serving](serving.md)
* [Pipeline](pipeline.md)
* [Runtime](runtime.md)
* [Datasets and snapshots](datasets-and-snapshots.md)

## Where to look for code

The code lives under `src/codeintel`:

* `src/codeintel/ingestion`
* `src/codeintel/analytics`
* `src/codeintel/graphs`
* `src/codeintel/storage`
* `src/codeintel/serving`
* `src/codeintel/pipeline`
* `src/codeintel/runtime`
* `src/codeintel/core`
* `src/codeintel/config`

````

#### `layering.md`

```markdown
# Layering and boundaries

Define your explicit layering rules here for humans and LLM agents, e.g.:

- **Core / config**  
  - `codeintel.core`, `codeintel.config`  
  - No imports from engines, serving, or pipeline.

- **Engines (analytics, graphs, ingestion, storage)**  
  - Can depend on core/config.
  - Must not import `serving` or `pipeline`.

- **Serving**  
  - Implements backend services and HTTP/MCP layers.
  - Depends on storage/analytics/graphs but not on pipeline.

- **Pipeline**  
  - Orchestration layer that can depend on everything else except UI/clients.

When an agent modifies code, it should check this page first to avoid introducing layering violations.
````

#### `datasets-and-snapshots.md`

```markdown
# Datasets and snapshots

Document:

- Snapshot model (e.g. `SnapshotRef`, snapshot context, build paths).
- Dataset contracts (schema, ownership, lifecycle).
- How ingestion, analytics, graphs, and serving use snapshot context.

Link from here into the API docs for dataset contracts and snapshot-related models
(e.g. `codeintel.config.datasets.*`, `codeintel.storage.*`, etc.).
```

For each of the subsystem-specific pages (`ingestion.md`, `analytics.md`, `graphs.md`, `storage.md`, `serving.md`, `pipeline.md`, `runtime.md`), follow a consistent template:

```markdown
# Ingestion

## Responsibility

What `codeintel.ingestion` owns: e.g. building DB snapshots, extracting code metadata, etc.

## Key entrypoints

- `codeintel.ingestion.cli.*`
- `codeintel.ingestion.recipes.*`
- ...

## Dependencies

- Reads from: repo on disk, VCS metadata, etc.
- Writes to: `codeintel.storage` tables, snapshot directories.
- Called by: [Pipeline](pipeline.md), tests, etc.

## Extension points

Guidance for LLMs (and humans) on how to safely add/modify ingestion behaviors.
```

Repeat with appropriate wording for analytics, graphs, storage, serving, pipeline, runtime.

You don’t need to write all the prose now; these can be “TODO” stubs that you (or agents) fill in later. The important part is the headings and the cross-links.

---

## 4. Add `mkdocs_gen/gen_ref_pages.py`

> **Task for agent:** create file `mkdocs_gen/gen_ref_pages.py` with this content.

This is adapted from the official mkdocstrings recipe for “Automatic code reference pages” with literate-nav and section-index. ([mkdocstrings][8])

```python
"""Generate API reference pages and navigation for codeintel.*

This script is executed by the mkdocs-gen-files plugin during `mkdocs build`.
It:

1. Walks all Python modules under `src/`.
2. For each module, generates a virtual Markdown file under `reference/`.
3. Builds a literate nav at `reference/SUMMARY.md` that mkdocs-literate-nav uses.
4. Uses mkdocs-section-index so package `__init__.py` docs show on the section index.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files


def main() -> None:
    # repo root: mkdocs_gen/ is at repo root, so parent.parent is root
    root = Path(__file__).resolve().parent.parent

    # All source packages live under src/ (codeintel, etc.)
    src_root = root / "src"

    nav = mkdocs_gen_files.Nav()

    # Iterate over all .py files under src_root
    for path in sorted(src_root.rglob("*.py")):
        # E.g. src/codeintel/analytics/foo.py -> codeintel/analytics/foo
        module_path = path.relative_to(src_root).with_suffix("")
        doc_path = path.relative_to(src_root).with_suffix(".md")  # same but .md
        full_doc_path = Path("reference", doc_path)               # docs/reference/...

        parts = tuple(module_path.parts)

        # Handle __init__.py specially: use index.md and bind to section (section-index) :contentReference[oaicite:21]{index=21}
        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                # This would be a top-level __init__ (unlikely, but guard anyway)
                continue
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            # Skip script entrypoints
            continue

        # Build the literate nav structure: e.g. codeintel -> analytics -> foo
        nav[parts] = doc_path.as_posix()

        # Emit the virtual Markdown file with an mkdocstrings directive
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)  # dotted module path, e.g. codeintel.analytics.foo
            fd.write(f"# `{ident}`\n\n")
            fd.write(f"::: {ident}\n")
            fd.write("    options:\n")
            fd.write("      heading_level: 2\n")
            fd.write("      show_root_heading: false\n")
            fd.write("      show_source: true\n")

        # Map "Edit this page" back to the real source file in the repo
        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    # Write literate nav file for the reference section
    # mkdocs-literate-nav will consume this when nav includes `reference/`. :contentReference[oaicite:22]{index=22}
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


if __name__ == "__main__":
    main()
```

This gives you:

* One page per module under `src/`, under `reference/...`.
* Clickable sections for packages with docs bound to `index.md` (thanks to `section-index`). ([mkdocstrings][8])
* Navigation for the entire API generated automatically, anchored under “Code Reference: reference/” in `mkdocs.yml`. ([mkdocstrings][8])

If you ever want to restrict this to only `codeintel.*`, you can change:

```python
src_root = root / "src"
```

to:

```python
src_root = root / "src" / "codeintel"
```

and adjust the relative path logic accordingly (e.g. prepend `"codeintel"` to `parts`), but for your repo `src/` probably just contains `codeintel` anyway.

---

## 5. (Optional but recommended) Architecture diagrams script

You can integrate `pydeps` and `pyreverse` via a small helper script. This doesn’t need to be a MkDocs plugin; just run it occasionally to refresh diagrams.

> **Task for agent:** create `mkdocs_gen/gen_arch_diagrams.py`:

```python
"""Generate architecture diagrams (pydeps + pyreverse) for CodeIntel docs.

- pydeps: module dependency graph for the codeintel package.
- pyreverse: UML-style class and package diagrams for codeintel.

Outputs:
  mkdocs-build/docs/architecture/codeintel-imports.svg
  mkdocs-build/docs/architecture/codeintel-packages.svg
  mkdocs-build/docs/architecture/codeintel-classes.svg
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[gen_arch_diagrams] Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src_root = root / "src"
    docs_arch = root / "mkdocs-build" / "docs" / "architecture"
    docs_arch.mkdir(parents=True, exist_ok=True)

    # Ensure PYTHONPATH includes src so pydeps/pyreverse can import codeintel :contentReference[oaicite:25]{index=25}
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(src_root))

    # 1) pydeps: import graph for codeintel as a package
    pydeps_output = docs_arch / "codeintel-imports.svg"
    run(
        [
            "pydeps",
            "codeintel",
            "--max-bacon",
            "2",
            "--cluster",
            "--noshow",
            "-T",
            "svg",
            "-o",
            str(pydeps_output),
        ],
        cwd=src_root,
    )

    # 2) pyreverse: package and class diagrams for codeintel 
    # Generate UML diagrams into src_root
    run(
        [
            "pyreverse",
            "-o",
            "svg",
            "-p",
            "codeintel",
            "codeintel",
        ],
        cwd=src_root,
    )

    # Move the generated SVGs into the docs/architecture folder
    packages_svg = src_root / "packages_codeintel.svg"
    classes_svg = src_root / "classes_codeintel.svg"

    if packages_svg.exists():
        packages_svg.replace(docs_arch / "codeintel-packages.svg")
    if classes_svg.exists():
        classes_svg.replace(docs_arch / "codeintel-classes.svg")

    print("[gen_arch_diagrams] Done.")


if __name__ == "__main__":
    main()
```

> **Task for agent:** wire these diagrams into `architecture/overview.md` (just add image references):

```markdown
## Module dependency graph

![CodeIntel import graph](codeintel-imports.svg)

## Package structure

![CodeIntel packages](codeintel-packages.svg)

## Class structure (selected core types)

![CodeIntel classes](codeintel-classes.svg)
```

To regenerate diagrams:

```bash
python mkdocs_gen/gen_arch_diagrams.py
```

(pydeps and pyreverse will use Graphviz’s `dot` binary under the hood. ([pythonhosted.org][9]))

---

## 6. How to build & serve the docs

From repo root:

```bash
# Build docs into mkdocs-output/
mkdocs build -f mkdocs-build/mkdocs.yml

# Live-reload server (good while editing docs or code)
mkdocs serve -f mkdocs-build/mkdocs.yml -a localhost:8000
```

Mkdocstrings is configured with `watch: ["../src"]`, so `mkdocs serve` will rebuild when you change code under `src/`. ([PyPI][10])

---

## 7. Summary of tasks for your AI programming agent

If you want a checklist to hand to an agent:

1. **Create directories**

   * `mkdocs_gen/`
   * `mkdocs-build/`
   * `mkdocs-build/docs/`
   * `mkdocs-build/docs/architecture/`
   * (Leave `mkdocs-output/` empty; MkDocs will create it.)

2. **Add MkDocs config**

   * Create `mkdocs-build/mkdocs.yml` exactly as specified above.

3. **Create docs skeleton**

   * `mkdocs-build/docs/index.md`
   * `mkdocs-build/docs/architecture/overview.md`
   * `mkdocs-build/docs/architecture/layering.md`
   * `mkdocs-build/docs/architecture/datasets-and-snapshots.md`
   * `mkdocs-build/docs/architecture/{ingestion,analytics,graphs,storage,serving,pipeline,runtime}.md` as stubs.

4. **Implement auto API generation**

   * Create `mkdocs_gen/gen_ref_pages.py` with the provided script.
   * Ensure `mkdocs-build/mkdocs.yml` plugin config references `../mkdocs_gen/gen_ref_pages.py` in `gen-files.scripts`.

5. **(Optional) Implement architecture diagrams**

   * Create `mkdocs_gen/gen_arch_diagrams.py` with the provided script.
   * Add image links into `architecture/overview.md`.
   * Run `python mkdocs_gen/gen_arch_diagrams.py` to generate SVGs.

6. **Verify build**

   * Run `mkdocs build -f mkdocs-build/mkdocs.yml` and confirm that:

     * `mkdocs-output/` is created.
     * The navigation shows Overview, Architecture, and Code Reference.
     * Code Reference expands into `codeintel` and its modules.
   * Run `mkdocs serve -f mkdocs-build/mkdocs.yml` and sanity check locally.

Once this is in place, you’ll have:

* A fully auto-generated, per-module API reference (with NumPy-parsed docstrings and types). ([mkdocstrings][8])
* Architecture docs and diagrams that LLMs (and humans) can use to reason about CodeIntel’s structure.
* A clean separation of *generation code* (`mkdocs_gen`), *build config and content* (`mkdocs-build`), and *output* (`mkdocs-output`), which makes it easy to automate and reason about in your tooling.

[1]: https://pypi.org/project/mkdocs-gen-files/?utm_source=chatgpt.com "mkdocs-gen-files"
[2]: https://pypi.org/project/pydeps/1.2.8/?utm_source=chatgpt.com "pydeps 1.2.8"
[3]: https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/?utm_source=chatgpt.com "Setting up navigation - Material for MkDocs - GitHub Pages"
[4]: https://www.mkdocs.org/user-guide/configuration/?utm_source=chatgpt.com "Configuration"
[5]: https://mkdocstrings.github.io/recipes/?utm_source=chatgpt.com "Recipes - mkdocstrings"
[6]: https://mkdocstrings.github.io/usage/?utm_source=chatgpt.com "Usage - mkdocstrings"
[7]: https://pypi.org/project/mkdocs-autorefs/?utm_source=chatgpt.com "mkdocs-autorefs"
[8]: https://mkdocstrings.github.io/recipes/ "Recipes - mkdocstrings"
[9]: https://pythonhosted.org/pydeps/?utm_source=chatgpt.com "pydeps 1.3.6 documentation"
[10]: https://pypi.org/project/mkdocstrings/0.9.0/?utm_source=chatgpt.com "mkdocstrings · PyPI"
