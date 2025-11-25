"""Utility helpers for sample fixtures."""

from pkg.mod import hello


def loud(name: str) -> str:
    """Return an upper-cased greeting.

    Parameters
    ----------
    name:
        Person to greet loudly.

    Returns
    -------
    str
        Upper-case greeting string.
    """
    msg = hello(name)
    return msg.upper()
