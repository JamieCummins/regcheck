from enum import Enum
from sqlmodel import Field, Relationship, SQLModel
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.models.user import User, UserPublic


class Divergence(Enum):
    NO = 0
    YES = 1
    UNSURE = -1


class DimensionComparisonResultBase(SQLModel):
    dimension_name: str
    dimension_description: Optional[str]
    divergence: Divergence
    divergence_summary: str
    preregistration_quotes: str
    paper_quotes: str


class DimensionComparisonResult(DimensionComparisonResultBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

    comparison_id: int = Field(default=None, foreign_key="comparisonresult.id")
    comparison: Optional["ComparisonResult"] = Relationship(
        back_populates="dimensions")


class DimensionComparisonResultPublic(DimensionComparisonResultBase):
    ...


class ComparisonResultBase(SQLModel):
    user_id: int | None = Field(default=None, foreign_key="user.id")
    # TODO: Datetime


class ComparisonResult(ComparisonResultBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user: "User" = Relationship(back_populates="comparisons")
    dimensions: list["DimensionComparisonResult"] = Relationship(
        back_populates="comparison"
    )


class ComparisonResultPublic(ComparisonResultBase):
    id: int


class ComparisonResultPublicWithDimensions(ComparisonResultPublic):
    user: "UserPublic"
    dimensions: list["DimensionComparisonResultPublic"]
