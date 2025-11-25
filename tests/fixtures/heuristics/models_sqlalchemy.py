"""SQLAlchemy fixture models backing heuristics tests."""

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """Minimal user table used to wire relationships."""

    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)


class Post(Base):
    """Post table with an owner relationship for fixture coverage."""

    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, nullable=True)
    user = relationship("User", back_populates="posts", uselist=False)

    def to_dict(self) -> dict[str, object]:
        """
        Return a serializable representation of the post.

        Returns
        -------
        dict[str, object]
            Dictionary containing the post identifier and title.
        """
        return {"id": self.id, "title": self.title}
