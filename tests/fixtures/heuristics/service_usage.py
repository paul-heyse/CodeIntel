"""Service-like helpers that exercise models in heuristics fixtures."""

from __future__ import annotations

from sqlalchemy.orm import Session

from tests.fixtures.heuristics.models_pydantic import UserPayload
from tests.fixtures.heuristics.models_sqlalchemy import Post, User


def create_user(session: Session, name: str) -> User:
    """
    Insert a user into the session without committing.

    Parameters
    ----------
    session : Session
        SQLAlchemy session receiving the new user.
    name : str
        User name to persist.

    Returns
    -------
    User
        The transient User instance added to the session.
    """
    user = User(name=name)
    session.add(user)
    return user


def fetch_user(session: Session) -> User | None:
    """
    Retrieve the first user from the database.

    Parameters
    ----------
    session : Session
        SQLAlchemy session used for querying.

    Returns
    -------
    User | None
        The first User if present, otherwise None.
    """
    return session.query(User).first()


def serialize_post(post: Post) -> dict[str, object]:
    """
    Serialize a Post model into a plain dictionary.

    Parameters
    ----------
    post : Post
        Model instance to convert.

    Returns
    -------
    dict[str, object]
        Mapping containing the post identifier and title.
    """
    return post.to_dict()


def serialize_payload(payload: UserPayload) -> dict[str, object]:
    """
    Convert a Pydantic payload to a dictionary.

    Parameters
    ----------
    payload : UserPayload
        Payload to serialize.

    Returns
    -------
    dict[str, object]
        Dictionary representation of the payload fields.
    """
    return payload.dict()
