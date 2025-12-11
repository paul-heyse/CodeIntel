"""Orchestrate MkDocs documentation build with progress tracking.

This script provides a unified entry point for building CodeIntel documentation
with detailed progress reporting, parallelization where possible, and clear
status messages.

Features
--------
- Parallel diagram generation (pydeps + pyreverse run concurrently)
- Progress tracking with tqdm
- Detailed logging of each build phase
- Summary report at completion

Usage
-----
From repo root::

    python mkdocs_gen/build_docs.py

Or via Makefile::

    make docs
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from tqdm import tqdm

from mkdocs_gen.command_runner import CommandError, run_command_sync, stream_command_sync

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# Root paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_CONFIG = REPO_ROOT / "mkdocs-build" / "mkdocs.yml"
OUTPUT_DIR = REPO_ROOT / "mkdocs-output"


class ProgressBar(Protocol):
    """Minimal progress bar contract used by the build pipeline."""

    def set_description(self, desc: str) -> None:
        """Set the progress bar description."""

    def refresh(self) -> None:
        """Refresh the progress bar display."""

    def update(self, n: float | None = 1) -> bool | None:
        """Advance the progress bar by ``n`` units."""

    def close(self) -> None:
        """Close and clean up the progress bar."""


@dataclass
class BuildPhase:
    """Represents a build phase with timing and status.

    Attributes
    ----------
    name
        Human-readable name of the phase.
    status
        Current status (pending, running, success, failed, skipped).
    duration
        Time taken in seconds.
    error
        Error message if failed.
    """

    name: str
    status: str = "pending"
    duration: float = 0.0
    error: str | None = None


@dataclass
class BuildContext:
    """Context for the documentation build.

    Attributes
    ----------
    phases
        List of build phases.
    start_time
        Build start timestamp.
    parallel
        Whether to use parallel execution.
    """

    phases: list[BuildPhase] = field(default_factory=list)
    start_time: float = 0.0
    parallel: bool = True


def run_phase(
    phase: BuildPhase,
    func: Callable[[], None],
    pbar: ProgressBar,
) -> None:
    """Execute a build phase with timing and status tracking.

    Parameters
    ----------
    phase
        The phase being executed.
    func
        Function to execute for this phase.
    pbar
        Progress bar to update.
    """
    phase.status = "running"
    pbar.set_description(f"  {phase.name}")
    pbar.refresh()

    start = time.perf_counter()
    try:
        func()
        phase.status = "success"
    except FileNotFoundError as e:
        phase.status = "failed"
        phase.error = f"Command not found: {e.filename}"
        log.exception("%s failed", phase.name)
    except (OSError, RuntimeError) as e:
        phase.status = "failed"
        phase.error = str(e)
        log.exception("%s failed", phase.name)
    finally:
        phase.duration = time.perf_counter() - start
        pbar.update(1)


def generate_pydeps_diagram() -> None:
    """Generate the pydeps import graph diagram."""
    log.info("Generating pydeps import graph...")
    src_root = REPO_ROOT / "src"
    output = REPO_ROOT / "mkdocs-build" / "docs" / "architecture" / "codeintel-imports.svg"
    output.parent.mkdir(parents=True, exist_ok=True)

    run_command_sync(
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
            str(output),
        ],
        cwd=src_root,
    )
    log.info("  -> %s", output.name)


def generate_pyreverse_diagrams() -> None:
    """Generate pyreverse UML diagrams."""
    log.info("Generating pyreverse UML diagrams...")
    src_root = REPO_ROOT / "src"
    docs_arch = REPO_ROOT / "mkdocs-build" / "docs" / "architecture"
    docs_arch.mkdir(parents=True, exist_ok=True)

    # Run pyreverse (capture stderr to suppress "Format svg not supported natively" message)
    run_command_sync(
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

    # Move generated files
    for src_name, dest_name in [
        ("packages_codeintel.svg", "codeintel-packages.svg"),
        ("classes_codeintel.svg", "codeintel-classes.svg"),
    ]:
        src_path = src_root / src_name
        if src_path.exists():
            dest_path = docs_arch / dest_name
            src_path.replace(dest_path)
            log.info("  -> %s", dest_name)


def run_mkdocs_build() -> None:
    """Run the MkDocs build process.

    Raises
    ------
    CommandError
        If mkdocs build fails.
    """
    log.info("Running MkDocs build...")
    log.info("  Config: %s", MKDOCS_CONFIG.relative_to(REPO_ROOT))
    log.info("  Output: %s", OUTPUT_DIR.relative_to(REPO_ROOT))

    module_count = 0
    try:
        lines = stream_command_sync(
            ["mkdocs", "build", "-f", str(MKDOCS_CONFIG)],
            cwd=REPO_ROOT,
        )
    except CommandError as exc:
        raise CommandError(exc.cmd, exc.returncode, exc.output) from exc

    for line in lines:
        lowered = line.lower()
        if "reference/" in lowered:
            module_count += 1
            if module_count % 50 == 0:
                log.info("  Processed %d modules...", module_count)
        elif lowered.startswith("info"):
            if "documentation built" in lowered or "building" in lowered:
                log.info("  %s", line.split(" - ", 1)[-1])
        elif lowered.startswith("error"):
            log.error("  %s", line)

    # Count output files
    html_files = list(OUTPUT_DIR.rglob("*.html"))
    log.info("  Generated %d HTML pages", len(html_files))


def count_source_modules() -> int:
    """Count Python modules in src/ for progress estimation.

    Returns
    -------
    int
        Number of .py files in src/.
    """
    src_root = REPO_ROOT / "src"
    return len(list(src_root.rglob("*.py")))


def build_docs(*, parallel: bool = True, skip_diagrams: bool = False) -> BuildContext:
    """Build the complete documentation.

    Parameters
    ----------
    parallel
        If True, run diagram generation in parallel.
    skip_diagrams
        If True, skip diagram generation entirely.

    Returns
    -------
    BuildContext
        Build context with phase results.
    """
    ctx = BuildContext(parallel=parallel)
    ctx.start_time = time.perf_counter()

    # Define phases
    pydeps_phase: BuildPhase | None = None
    pyreverse_phase: BuildPhase | None = None
    if not skip_diagrams:
        pydeps_phase = BuildPhase("Pydeps import graph")
        pyreverse_phase = BuildPhase("Pyreverse UML diagrams")
        ctx.phases.extend([pydeps_phase, pyreverse_phase])

    mkdocs_phase = BuildPhase("MkDocs build")
    ctx.phases.append(mkdocs_phase)

    # Header
    module_count = count_source_modules()
    log.info("")
    log.info("=" * 70)
    log.info("CodeIntel Documentation Build")
    log.info("=" * 70)
    log.info("  Source modules: %d", module_count)
    log.info("  Parallel mode:  %s", "enabled" if parallel else "disabled")
    log.info("  Output:         %s/", OUTPUT_DIR.relative_to(REPO_ROOT))
    log.info("-" * 70)
    log.info("")

    # Progress bar for all phases
    pbar = cast(
        "ProgressBar",
        tqdm(
            total=len(ctx.phases),
            desc="Building docs",
            unit="phase",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {bar}",
        ),
    )
    try:
        if not skip_diagrams and parallel and pydeps_phase and pyreverse_phase:
            # Run diagram generation in parallel
            pbar.set_description("Generating diagrams (parallel)")

            def timed_task(
                task_func: Callable[[], None],
                phase: BuildPhase,
            ) -> None:
                """Run a task and record timing in the phase."""
                start = time.perf_counter()
                try:
                    task_func()
                    phase.status = "success"
                except (FileNotFoundError, OSError, RuntimeError) as e:
                    phase.status = "failed"
                    phase.error = str(e)
                phase.duration = time.perf_counter() - start

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(timed_task, generate_pydeps_diagram, pydeps_phase),
                    executor.submit(timed_task, generate_pyreverse_diagrams, pyreverse_phase),
                ]

                for future in as_completed(futures):
                    future.result()  # Re-raises any exception
                    pbar.update(1)

        elif not skip_diagrams and pydeps_phase and pyreverse_phase:
            # Sequential diagram generation
            run_phase(pydeps_phase, generate_pydeps_diagram, pbar)
            run_phase(pyreverse_phase, generate_pyreverse_diagrams, pbar)

        # MkDocs build (always sequential)
        run_phase(mkdocs_phase, run_mkdocs_build, pbar)
    finally:
        pbar.close()

    return ctx


def print_summary(ctx: BuildContext) -> None:
    """Print build summary.

    Parameters
    ----------
    ctx
        Build context with results.
    """
    total_time = time.perf_counter() - ctx.start_time

    log.info("")
    log.info("-" * 70)
    log.info("Build Summary")
    log.info("-" * 70)

    # Phase results
    for phase in ctx.phases:
        status_icon = "OK" if phase.status == "success" else "FAIL"
        log.info("  [%s] %-30s (%.1fs)", status_icon.ljust(4), phase.name, phase.duration)
        if phase.error:
            log.error("         Error: %s", phase.error)

    log.info("-" * 70)

    succeeded = sum(1 for p in ctx.phases if p.status == "success")
    failed = len(ctx.phases) - succeeded

    if failed == 0:
        log.info("Build completed successfully in %.1fs", total_time)
        log.info("  Output: %s/", OUTPUT_DIR)
        log.info("")
        log.info("To view locally:")
        log.info("  make docs-serve")
        log.info("  open http://localhost:8000")
    else:
        log.error("Build completed with %d failure(s) in %.1fs", failed, total_time)

    log.info("=" * 70)
    log.info("")


def main() -> int:
    """CLI entrypoint.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any phase failed).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Parse simple arguments
    args = sys.argv[1:]
    parallel = "--no-parallel" not in args
    skip_diagrams = "--skip-diagrams" in args

    if "--help" in args or "-h" in args:
        log.info(__doc__)
        log.info("")
        log.info("Options:")
        log.info("  --no-parallel    Disable parallel diagram generation")
        log.info("  --skip-diagrams  Skip diagram generation entirely")
        log.info("  --help, -h       Show this help message")
        return 0

    try:
        ctx = build_docs(parallel=parallel, skip_diagrams=skip_diagrams)
    except KeyboardInterrupt:
        log.warning("Build interrupted by user")
        return 130

    print_summary(ctx)

    failed = [p for p in ctx.phases if p.status != "success"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
