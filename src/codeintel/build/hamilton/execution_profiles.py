"""Execution profile helpers for Hamilton runtime configuration."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton.adapters.parallel import create_parallel_adapter
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_options import BuildExecutionOptions

if TYPE_CHECKING:
    from hamilton.lifecycle import ResultBuilder


log = logging.getLogger(__name__)

_EXECUTOR_ALIASES: dict[str, str] = {
    "synchronous": "sync",
    "sync": "sync",
    "local": "sync",
    "thread": "thread",
    "threads": "thread",
    "threading": "thread",
    "process": "process",
    "processes": "process",
    "multiprocessing": "process",
    "mp": "process",
    "none": "none",
    "off": "none",
    "disabled": "none",
}

_EXECUTOR_CLASS_NAMES: dict[str, str] = {
    "sync": "SynchronousLocalTaskExecutor",
    "thread": "MultiThreadingExecutor",
    "process": "MultiProcessingExecutor",
}


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Resolved execution profile for a build run."""

    parallel_backend: str
    max_workers: int | None
    thread_name_prefix: str
    dynamic_enabled: bool
    dynamic_local_executor: str | None
    dynamic_remote_executor: str | None
    dynamic_remote_max_tasks: int | None


@dataclass(frozen=True, slots=True)
class DynamicExecutionConfig:
    """Resolved dynamic executor configuration for Hamilton."""

    enabled: bool
    local_executor: object | None
    remote_executor: object | None


def build_execution_profile(
    *,
    env: BuildEnv,
    options: BuildExecutionOptions,
    max_workers: int | None,
    thread_name_prefix: str,
) -> ExecutionProfile:
    """Build the execution profile from environment and runtime options.

    Returns
    -------
    ExecutionProfile
        Resolved execution profile for the build run.
    """
    remote_max_tasks = env.execution_settings.dynamic_remote_max_tasks
    if remote_max_tasks is None:
        remote_max_tasks = options.max_workers
    if remote_max_tasks is None:
        remote_max_tasks = env.execution_settings.max_workers
    return ExecutionProfile(
        parallel_backend=options.parallel_backend,
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
        dynamic_enabled=bool(env.execution_settings.dynamic_execution),
        dynamic_local_executor=env.execution_settings.dynamic_local_executor,
        dynamic_remote_executor=env.execution_settings.dynamic_remote_executor,
        dynamic_remote_max_tasks=remote_max_tasks,
    )


def build_parallel_adapter(
    profile: ExecutionProfile,
    *,
    result_builder: ResultBuilder | None,
    dynamic_enabled: bool,
) -> lifecycle_base.LifecycleAdapter | None:
    """Create a graph adapter for the configured parallel backend.

    Returns
    -------
    LifecycleAdapter | None
        Adapter instance, or None when dynamic execution is enabled.
    """
    if dynamic_enabled:
        return None
    return create_parallel_adapter(
        profile.parallel_backend,
        max_workers=profile.max_workers,
        thread_name_prefix=profile.thread_name_prefix,
        result_builder=result_builder,
    )


def apply_dynamic_execution_config(
    *,
    config: MutableMapping[str, object],
    profile: ExecutionProfile,
) -> DynamicExecutionConfig:
    """Resolve dynamic executors and update the Hamilton config.

    Returns
    -------
    DynamicExecutionConfig
        Resolved executor configuration for dynamic execution.
    """
    dynamic_config = resolve_dynamic_execution_config(profile)
    config["ci.dynamic_execution"] = dynamic_config.enabled
    config["ci_dynamic_module_records"] = dynamic_config.enabled
    if dynamic_config.enabled:
        if dynamic_config.local_executor is not None:
            config["ci.dynamic_execution.local_executor"] = dynamic_config.local_executor
        if dynamic_config.remote_executor is not None:
            config["ci.dynamic_execution.remote_executor"] = dynamic_config.remote_executor
    return dynamic_config


def resolve_dynamic_execution_config(profile: ExecutionProfile) -> DynamicExecutionConfig:
    """Resolve dynamic executors for the requested execution profile.

    Returns
    -------
    DynamicExecutionConfig
        Resolved executor configuration for dynamic execution.
    """
    if not profile.dynamic_enabled:
        return DynamicExecutionConfig(enabled=False, local_executor=None, remote_executor=None)
    local_executor = _resolve_task_executor(
        name=profile.dynamic_local_executor,
        default="sync",
        max_tasks=None,
    )
    remote_executor = _resolve_task_executor(
        name=profile.dynamic_remote_executor,
        default="thread",
        max_tasks=profile.dynamic_remote_max_tasks,
    )
    if local_executor is None and remote_executor is None:
        log.warning("Dynamic execution enabled but no executors resolved; disabling")
        return DynamicExecutionConfig(enabled=False, local_executor=None, remote_executor=None)
    if local_executor is None:
        local_executor = _resolve_task_executor(
            name="sync",
            default="sync",
            max_tasks=None,
        )
    return DynamicExecutionConfig(
        enabled=True,
        local_executor=local_executor,
        remote_executor=remote_executor,
    )


def _normalize_executor_name(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return _EXECUTOR_ALIASES.get(normalized, normalized)


def _executor_kwargs(executor_cls: type[object], *, max_tasks: int | None) -> dict[str, object]:
    if max_tasks is None:
        return {}
    params = inspect.signature(executor_cls).parameters
    if "max_tasks" in params:
        return {"max_tasks": max_tasks}
    if "max_workers" in params:
        return {"max_workers": max_tasks}
    if "max_concurrent_tasks" in params:
        return {"max_concurrent_tasks": max_tasks}
    return {}


def _instantiate_task_executor(
    executor_cls: type[object],
    *,
    max_tasks: int | None,
    label: str,
) -> object | None:
    kwargs = _executor_kwargs(executor_cls, max_tasks=max_tasks)
    try:
        return executor_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        log.warning("Failed to instantiate %s executor: %s", label, exc)
        return None


def _resolve_task_executor(
    *,
    name: str | None,
    default: str,
    max_tasks: int | None,
) -> object | None:
    normalized = _normalize_executor_name(name, default=default)
    if normalized == "none":
        return None
    class_name = _EXECUTOR_CLASS_NAMES.get(normalized)
    if class_name is None:
        log.warning("Unknown dynamic executor '%s', defaulting to %s", normalized, default)
        class_name = _EXECUTOR_CLASS_NAMES.get(default)
        if class_name is None:
            return None
    try:
        executors = importlib.import_module("hamilton.execution.executors")
    except ModuleNotFoundError as exc:
        log.warning("Dynamic execution requested but executors module missing: %s", exc)
        return None
    executor_cls = getattr(executors, class_name, None)
    if executor_cls is None:
        log.warning("Dynamic executor class %s is unavailable", class_name)
        return None
    return _instantiate_task_executor(
        executor_cls,
        max_tasks=max_tasks,
        label=normalized,
    )


__all__ = [
    "DynamicExecutionConfig",
    "ExecutionProfile",
    "apply_dynamic_execution_config",
    "build_execution_profile",
    "build_parallel_adapter",
    "resolve_dynamic_execution_config",
]
