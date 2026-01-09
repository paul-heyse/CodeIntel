"""Runtime helpers for Arrow threading and profile application."""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass

import pyarrow as pa

from codeintel.core.columnar.profiles import RuntimeProfile
from codeintel.core.config.settings import ArrowScanSettings
from codeintel.core.constants import (
    DEFAULT_ARROW_CPU_COUNT,
    DEFAULT_ARROW_IO_THREAD_COUNT,
    DEFAULT_ARROW_IO_THREAD_MULTIPLIER,
    DEFAULT_ARROW_MIN_IO_THREADS,
)
from codeintel.core.runtime.loader import load_runtime_settings

_ARROW_THREADING_CONFIGURED = threading.Event()


@dataclass(frozen=True, slots=True)
class ArrowThreadingSnapshot:
    """Snapshot of Arrow threading configuration for run manifests."""

    profile_name: str | None
    cpu_threads_requested: int | None
    io_threads_requested: int | None
    cpu_threads_before: int | None
    io_threads_before: int | None
    cpu_threads_after: int | None
    io_threads_after: int | None
    configured: bool

    def to_mapping(self) -> dict[str, int | bool | str | None]:
        """Return a mapping payload for manifest extras.

        Returns
        -------
        dict[str, int | bool | str | None]
            Mapping representation for JSON manifests.
        """
        return {
            "profile_name": self.profile_name,
            "cpu_threads_requested": self.cpu_threads_requested,
            "io_threads_requested": self.io_threads_requested,
            "cpu_threads_before": self.cpu_threads_before,
            "io_threads_before": self.io_threads_before,
            "cpu_threads_after": self.cpu_threads_after,
            "io_threads_after": self.io_threads_after,
            "configured": self.configured,
        }


def apply_arrow_threading(
    *,
    cpu_threads: int | None = DEFAULT_ARROW_CPU_COUNT,
    io_threads: int | None = DEFAULT_ARROW_IO_THREAD_COUNT,
    settings: ArrowScanSettings | None = None,
    profile_name: str | None = None,
) -> ArrowThreadingSnapshot:
    """Apply Arrow CPU and IO thread pool defaults.

    Returns
    -------
    ArrowThreadingSnapshot
        Snapshot capturing before/after thread counts.
    """
    resolved_settings = _resolve_arrow_scan_settings(settings)
    if cpu_threads == DEFAULT_ARROW_CPU_COUNT:
        cpu_threads = resolved_settings.cpu_count
    if io_threads == DEFAULT_ARROW_IO_THREAD_COUNT:
        io_threads = resolved_settings.io_thread_count
    requested_cpu = cpu_threads
    requested_io = io_threads
    before_cpu, before_io = _read_thread_counts()
    if _ARROW_THREADING_CONFIGURED.is_set():
        after_cpu, after_io = _read_thread_counts()
        return ArrowThreadingSnapshot(
            profile_name=profile_name,
            cpu_threads_requested=requested_cpu,
            io_threads_requested=requested_io,
            cpu_threads_before=before_cpu,
            io_threads_before=before_io,
            cpu_threads_after=after_cpu,
            io_threads_after=after_io,
            configured=False,
        )
    _ARROW_THREADING_CONFIGURED.set()
    resolved_cpu = _resolve_arrow_cpu_count(requested_cpu)
    resolved_io = _resolve_arrow_io_thread_count(requested_io, cpu_count=resolved_cpu)
    _apply_arrow_thread_counts(cpu_threads=resolved_cpu, io_threads=resolved_io)
    after_cpu, after_io = _read_thread_counts()
    return ArrowThreadingSnapshot(
        profile_name=profile_name,
        cpu_threads_requested=requested_cpu,
        io_threads_requested=requested_io,
        cpu_threads_before=before_cpu,
        io_threads_before=before_io,
        cpu_threads_after=after_cpu,
        io_threads_after=after_io,
        configured=True,
    )


def apply_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    settings: ArrowScanSettings | None = None,
) -> ArrowThreadingSnapshot:
    """Apply Arrow threading using runtime profile defaults.

    Returns
    -------
    ArrowThreadingSnapshot
        Snapshot capturing before/after thread counts.
    """
    resolved_settings = _resolve_arrow_scan_settings(settings)
    cpu_threads = resolved_settings.cpu_count
    io_threads = resolved_settings.io_thread_count
    if profile is not None:
        cpu_threads = profile.resolve_cpu_threads(default=cpu_threads)
        io_threads = profile.resolve_io_threads(default=io_threads)
    return apply_arrow_threading(
        cpu_threads=cpu_threads,
        io_threads=io_threads,
        settings=resolved_settings,
        profile_name=None if profile is None else profile.name,
    )


def _resolve_arrow_scan_settings(settings: ArrowScanSettings | None) -> ArrowScanSettings:
    if settings is not None:
        return settings
    return load_runtime_settings().build.arrow_scan


def _resolve_arrow_cpu_count(default_count: int | None) -> int:
    if default_count is not None and default_count > 0:
        return default_count
    detected = os.cpu_count() or 1
    return max(1, detected)


def _resolve_arrow_io_thread_count(
    default_count: int | None,
    *,
    cpu_count: int,
) -> int:
    if default_count is not None and default_count > 0:
        return default_count
    scaled = cpu_count * DEFAULT_ARROW_IO_THREAD_MULTIPLIER
    return max(DEFAULT_ARROW_MIN_IO_THREADS, scaled)


def _read_thread_counts() -> tuple[int | None, int | None]:
    before_cpu = None
    before_io = None
    with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
        before_cpu = pa.cpu_count()
    with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
        before_io = pa.io_thread_count()
    return before_cpu, before_io


def _apply_arrow_thread_counts(*, cpu_threads: int, io_threads: int) -> None:
    set_cpu = getattr(pa, "set_cpu_count", None)
    if callable(set_cpu):
        with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
            set_cpu(cpu_threads)
    set_io = getattr(pa, "set_io_thread_count", None)
    if callable(set_io):
        with contextlib.suppress(TypeError, ValueError, pa.ArrowInvalid):
            set_io(io_threads)


__all__ = [
    "ArrowThreadingSnapshot",
    "apply_arrow_threading",
    "apply_runtime_profile",
]
