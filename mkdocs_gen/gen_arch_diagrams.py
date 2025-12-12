"""Generate architecture diagrams for CodeIntel documentation.

This script produces SVG diagrams using pydeps and pyreverse to visualize:
- Module dependency graphs (pydeps)
- UML package and class diagrams (pyreverse)

Two types of diagrams are generated:
- **Overview diagrams**: Consolidated view showing only top-level packages
- **Detailed diagrams**: Full granularity (optional, for deep dives)

The generated diagrams are placed in the architecture documentation folder
for inclusion in the MkDocs site.

Outputs
-------
mkdocs-build/docs/architecture/codeintel-imports.svg
    Consolidated module import graph (top-level packages only).
mkdocs-build/docs/architecture/codeintel-packages.svg
    UML package diagram (top-level only).
mkdocs-build/docs/architecture/codeintel-classes.svg
    UML class diagram (key classes only, no attributes/methods).

Notes
-----
Requires Graphviz to be installed for SVG output from pyreverse.
Install with: ``sudo apt install graphviz`` (Ubuntu/Debian)
or ``brew install graphviz`` (macOS).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from mkdocs_gen.command_runner import CommandError, run_command_sync

log = logging.getLogger(__name__)


class DiagramResult(NamedTuple):
    """Result of a diagram generation task.

    Attributes
    ----------
    name
        Human-readable name of the diagram.
    success
        Whether generation succeeded.
    output_path
        Path to the generated file (if successful).
    error
        Error message (if failed).
    """

    name: str
    success: bool
    output_path: Path | None = None
    error: str | None = None


def check_graphviz() -> bool:
    """Check if Graphviz is installed and available.

    Returns
    -------
    bool
        True if Graphviz (dot) is available on PATH.
    """
    return shutil.which("dot") is not None


def run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_stderr: bool = False,
) -> str:
    """Execute a command with logging and error propagation.

    Returns
    -------
    str
        Combined output from the command.

    Raises
    ------
    CommandError
        If the command exits with a non-zero status.
    FileNotFoundError
        If the command executable cannot be resolved.
    """
    cwd_str = str(cwd) if cwd is not None else None
    log.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd_str)
    try:
        return run_command_sync(
            cmd,
            cwd=cwd,
            env=env,
            merge_stderr=capture_stderr,
        )
    except CommandError as exc:
        raise CommandError(exc.cmd, exc.returncode, exc.output) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(exc.filename or cmd[0]) from exc


TOP_LEVEL_PACKAGES = (
    "codeintel.analytics",
    "codeintel.config",
    "codeintel.core",
    "codeintel.graphs",
    "codeintel.ingestion",
    "codeintel.pipeline",
    "codeintel.runtime",
    "codeintel.serving",
    "codeintel.storage",
)


def generate_pydeps_diagram(
    src_root: Path,
    output_path: Path,
    env: dict[str, str],
) -> DiagramResult:
    """Generate consolidated module dependency diagram using pydeps.

    Create an overview diagram showing only top-level package dependencies,
    not the full detailed import graph. Uses --max-module-depth to coalesce
    all submodules into their parent packages.

    Parameters
    ----------
    src_root
        Path to the source root containing codeintel.
    output_path
        Path where the SVG will be written.
    env
        Environment variables including PYTHONPATH.

    Returns
    -------
    DiagramResult
        Result indicating success/failure and output path.
    """
    name = "pydeps import graph (overview)"
    log.info("[1/3] Generating %s...", name)

    try:
        cmd = [
            "pydeps",
            "codeintel",
            "--max-module-depth",
            "3",
            "--max-bacon",
            "0",
            "--only",
            "codeintel",
            "--cluster",
            "--rankdir",
            "TB",
            "--noshow",
            "-T",
            "svg",
            "-o",
            str(output_path),
        ]

        run_checked(cmd, cwd=src_root, env=env)
        log.info("[1/3] %s complete: %s", name, output_path.name)
        return DiagramResult(name=name, success=True, output_path=output_path)
    except CommandError as exc:
        log.exception("[1/3] %s failed", name)
        return DiagramResult(name=name, success=False, error=str(exc))
    except FileNotFoundError:
        msg = "pydeps not found - install with: pip install pydeps"
        log.exception("[1/3] %s failed: %s", name, msg)
        return DiagramResult(name=name, success=False, error=msg)


def generate_pyreverse_diagrams(
    src_root: Path,
    docs_arch: Path,
    env: dict[str, str],
) -> list[DiagramResult]:
    """Generate consolidated UML diagrams using pyreverse.

    Create overview diagrams that show only top-level packages and key classes,
    without the full detail of attributes, methods, and deep hierarchies.

    Parameters
    ----------
    src_root
        Path to the source root containing codeintel.
    docs_arch
        Path to the architecture docs folder for output.
    env
        Environment variables including PYTHONPATH.

    Returns
    -------
    list[DiagramResult]
        Results for packages and classes diagrams.
    """
    results: list[DiagramResult] = []

    log.info("[2/3] Generating pyreverse UML diagrams (overview)...")

    try:
        run_checked(
            [
                "pyreverse",
                "-o",
                "svg",
                "-p",
                "codeintel",
                "-k",
                "-a",
                "0",
                "-s",
                "0",
                *TOP_LEVEL_PACKAGES,
            ],
            cwd=src_root,
            env=env,
            capture_stderr=True,
        )

        packages_svg = src_root / "packages_codeintel.svg"
        classes_svg = src_root / "classes_codeintel.svg"

        if packages_svg.exists():
            dest = docs_arch / "codeintel-packages.svg"
            packages_svg.replace(dest)
            log.info("[2/3] Package diagram complete: %s", dest.name)
            results.append(
                DiagramResult(
                    name="pyreverse packages (overview)",
                    success=True,
                    output_path=dest,
                )
            )
        else:
            results.append(
                DiagramResult(
                    name="pyreverse packages (overview)",
                    success=False,
                    error="packages_codeintel.svg not generated",
                )
            )

        if classes_svg.exists():
            dest = docs_arch / "codeintel-classes.svg"
            classes_svg.replace(dest)
            log.info("[3/3] Class diagram complete: %s", dest.name)
            results.append(
                DiagramResult(
                    name="pyreverse classes (overview)",
                    success=True,
                    output_path=dest,
                )
            )
        else:
            results.append(
                DiagramResult(
                    name="pyreverse classes (overview)",
                    success=False,
                    error="classes_codeintel.svg not generated",
                )
            )

    except CommandError as exc:
        log.exception("[2/3] pyreverse failed")
        results.append(
            DiagramResult(name="pyreverse packages (overview)", success=False, error=str(exc))
        )
        results.append(
            DiagramResult(name="pyreverse classes (overview)", success=False, error=str(exc))
        )
    except FileNotFoundError:
        msg = "pyreverse not found - install with: pip install pylint"
        log.exception("[2/3] pyreverse failed: %s", msg)
        results.append(
            DiagramResult(name="pyreverse packages (overview)", success=False, error=msg)
        )
        results.append(DiagramResult(name="pyreverse classes (overview)", success=False, error=msg))

    return results


def generate_diagrams(*, parallel: bool = True) -> list[DiagramResult]:
    """Generate architecture diagrams using pydeps and pyreverse.

    Create SVG diagrams showing module dependencies and UML structure
    for the codeintel package. Diagrams are written to the architecture
    documentation folder.

    Parameters
    ----------
    parallel
        If True, run pydeps and pyreverse in parallel.

    Returns
    -------
    list[DiagramResult]
        Results for all diagram generation tasks.
    """
    root = Path(__file__).resolve().parent.parent
    src_root = root / "src"
    docs_arch = root / "mkdocs-build" / "docs" / "architecture"
    docs_arch.mkdir(parents=True, exist_ok=True)

    if not check_graphviz():
        log.warning(
            "Graphviz not found. Install with: sudo apt install graphviz (Ubuntu) "
            "or brew install graphviz (macOS)"
        )

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    if python_path:
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{python_path}"
    else:
        env["PYTHONPATH"] = str(src_root)

    pydeps_output = docs_arch / "codeintel-imports.svg"

    def generate_pydeps_batch() -> list[DiagramResult]:
        return [generate_pydeps_diagram(src_root, pydeps_output, env)]

    log.info("=" * 60)
    log.info("Architecture Diagram Generation")
    log.info("=" * 60)
    log.info("Output directory: %s", docs_arch)
    log.info("Parallel mode: %s", "enabled" if parallel else "disabled")
    log.info("-" * 60)

    results: list[DiagramResult] = []

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(generate_pydeps_batch),
                executor.submit(generate_pyreverse_diagrams, src_root, docs_arch, env),
            ]

            for future in as_completed(futures):
                results.extend(future.result())
    else:
        results.extend(generate_pydeps_batch())
        results.extend(generate_pyreverse_diagrams(src_root, docs_arch, env))

    log.info("-" * 60)
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    log.info("Diagram generation complete: %d succeeded, %d failed", succeeded, failed)

    for result in results:
        if result.success:
            log.info("  [OK] %s -> %s", result.name, result.output_path)
        else:
            log.error("  [FAIL] %s: %s", result.name, result.error)

    return results


def main() -> int:
    """CLI entrypoint for diagram generation.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any diagram failed).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results = generate_diagrams(parallel=True)
    failed = [r for r in results if not r.success]

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
