from sqlmodel import Field, Relationship, SQLModel
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from backend.models.comparison import ComparisonResult, ComparisonResultPublic


class UserBase(SQLModel):
    """
    Base model for the user. Contains the name and the list of comparisons
    assigned to the user.
    """

    name: str = Field(index=True)


class User(UserBase, table=True):
    """
    Database model for the user.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    comparisons: list["ComparisonResult"] = Relationship(back_populates="user")
    hashed_password: str


class UserCreate(UserBase):
    """
    Interface model for user creation
    """

    password: str


class UserPublic(UserBase):
    """
    Interface model for user in API
    """

    id: int


class UserPublicWithComparisons(UserPublic):
    comparisons: list["ComparisonResultPublic"]
