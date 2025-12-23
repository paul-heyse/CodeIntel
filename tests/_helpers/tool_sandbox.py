"""Tool sandbox helpers for deterministic CLI execution in tests."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from tests._helpers.tool_payloads import (
    coverage_json_payload,
    pytest_report_payload,
    scip_json_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class ToolStubSpec:
    """Specification for a stub executable."""

    stdout: str = ""
    returncode: int = 0
    creates: str | None = None
    creates_payload: str | None = None
    writes: str | None = None
    writes_payload: str | None = None


@dataclass
class ToolSandbox:
    """Create a sandboxed bin directory with stub tool executables."""

    root: Path
    bin_dir: Path
    _installed: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def create(cls, tmp_path: Path) -> ToolSandbox:
        """Create a tool sandbox rooted under a temp path.

        Returns
        -------
        ToolSandbox
            Newly created sandbox instance.
        """
        root = tmp_path / "tool_sandbox"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        return cls(root=root, bin_dir=bin_dir)

    def install_stub(
        self,
        name: str,
        *,
        spec: ToolStubSpec | None = None,
    ) -> Path:
        """Install a stub executable for a named tool.

        Returns
        -------
        Path
            Path to the created stub executable.
        """
        stub_path = self.bin_dir / name
        resolved = spec or ToolStubSpec()
        content = _render_stub_script(resolved)
        stub_path.write_text(content, encoding="utf-8")
        stub_path.chmod(0o755)
        self._installed[name] = stub_path
        return stub_path

    def install_default_stubs(self) -> None:
        """Install a default set of stub executables for common tools."""
        pytest_payload = pytest_report_payload(
            tests=[], summary={"passed": 0, "failed": 0, "skipped": 0}
        )
        coverage_payload = coverage_json_payload()
        scip_payload = scip_json_payload()
        self.install_stub(
            "pytest",
            spec=ToolStubSpec(
                writes="--json-report-file",
                writes_payload=json.dumps(pytest_payload),
            ),
        )
        self.install_stub(
            "coverage",
            spec=ToolStubSpec(
                writes="-o",
                writes_payload=json.dumps(coverage_payload),
            ),
        )
        self.install_stub(
            "scip-python",
            spec=ToolStubSpec(
                creates="--output",
                creates_payload="scip-binary",
            ),
        )
        self.install_stub(
            "scip",
            spec=ToolStubSpec(
                stdout=json.dumps(scip_payload),
            ),
        )
        self.install_stub("pyright", spec=ToolStubSpec(stdout="{}"))
        self.install_stub("pyrefly", spec=ToolStubSpec(stdout="{}"))
        self.install_stub("ruff", spec=ToolStubSpec(stdout="{}"))
        self.install_stub("git", spec=ToolStubSpec(stdout=""))

    def tools_config(self) -> ToolsConfig:
        """Return a ToolsConfig that points to stub binaries.

        Returns
        -------
        ToolsConfig
            Configuration with tool paths set to the stub binaries.
        """
        return ToolsConfig.with_overrides(
            scip_python_bin=str(self._installed.get("scip-python", "scip-python")),
            scip_bin=str(self._installed.get("scip", "scip")),
            pyright_bin=str(self._installed.get("pyright", "pyright")),
            pyrefly_bin=str(self._installed.get("pyrefly", "pyrefly")),
            ruff_bin=str(self._installed.get("ruff", "ruff")),
            coverage_bin=str(self._installed.get("coverage", "coverage")),
            pytest_bin=str(self._installed.get("pytest", "pytest")),
            git_bin=str(self._installed.get("git", "git")),
        )

    @contextmanager
    def prepend_path(self) -> Iterator[None]:
        """Temporarily prepend the sandbox bin directory to PATH.

        Yields
        ------
        None
            Control while the PATH override is active.
        """
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}{original}"
        try:
            yield None
        finally:
            os.environ["PATH"] = original


def _render_stub_script(spec: ToolStubSpec) -> str:
    """Render a stub executable script.

    Returns
    -------
    str
        Script contents for the stub executable.
    """
    return (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def _find_path(flag: str) -> Path | None:\n"
        "    args = sys.argv[1:]\n"
        "    if flag in args:\n"
        "        idx = args.index(flag)\n"
        "        if idx + 1 < len(args):\n"
        "            return Path(args[idx + 1])\n"
        "    for arg in args:\n"
        '        if arg.startswith(flag) and "=" in arg:\n'
        '            return Path(arg.split("=", 1)[1])\n'
        "    return None\n"
        "\n"
        "def _write(path: Path | None, payload: str) -> None:\n"
        "    if path is None:\n"
        "        return\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        '    path.write_text(payload, encoding="utf-8")\n'
        "\n"
        f"creates = {spec.creates!r}\n"
        f"writes = {spec.writes!r}\n"
        f"creates_payload = {spec.creates_payload or ''!r}\n"
        f"writes_payload = {spec.writes_payload or spec.stdout!r}\n"
        "if creates:\n"
        "    _write(_find_path(creates), creates_payload)\n"
        "if writes:\n"
        "    _write(_find_path(writes), writes_payload)\n"
        f"sys.stdout.write({spec.stdout!r})\n"
        f"sys.exit({spec.returncode})\n"
    )


__all__ = ["ToolSandbox", "ToolStubSpec"]
