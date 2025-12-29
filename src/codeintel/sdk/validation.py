"""Stable wrappers for Hamilton output validation modifiers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from hamilton.function_modifiers import check_output as h_check_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hamilton.data_quality.base import BaseDefaultValidator
    from hamilton.function_modifiers.base import TargetType

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


def check_output_warn(
    *,
    default_validator_candidates: Sequence[type[BaseDefaultValidator]] | None = None,
    target_: TargetType | None = None,
    **default_validator_kwargs: object,
) -> Decorator[P, R]:
    """Create a warning-level output check decorator.

    Returns
    -------
    Decorator[P, R]
        Decorator applying warning-level output checks.
    """
    return _check_output(
        "warn",
        default_validator_candidates=default_validator_candidates,
        target_=target_,
        **default_validator_kwargs,
    )


def check_output_fail(
    *,
    default_validator_candidates: Sequence[type[BaseDefaultValidator]] | None = None,
    target_: TargetType | None = None,
    **default_validator_kwargs: object,
) -> Decorator[P, R]:
    """Create a failure-level output check decorator.

    Returns
    -------
    Decorator[P, R]
        Decorator applying failure-level output checks.
    """
    return _check_output(
        "fail",
        default_validator_candidates=default_validator_candidates,
        target_=target_,
        **default_validator_kwargs,
    )


def _check_output(
    importance: str,
    *,
    default_validator_candidates: Sequence[type[BaseDefaultValidator]] | None = None,
    target_: TargetType | None = None,
    **default_validator_kwargs: object,
) -> Decorator[P, R]:
    decorator_factory = cast("Callable[..., object]", h_check_output)
    if default_validator_candidates is None:
        return cast(
            "Decorator[P, R]",
            decorator_factory(
                importance=importance,
                target_=target_,
                **default_validator_kwargs,
            ),
        )
    candidates = list(default_validator_candidates)
    return cast(
        "Decorator[P, R]",
        decorator_factory(
            importance=importance,
            default_validator_candidates=candidates,
            target_=target_,
            **default_validator_kwargs,
        ),
    )


__all__ = [
    "check_output_fail",
    "check_output_warn",
]
