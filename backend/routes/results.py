from fastapi import APIRouter, HTTPException, status

from backend.core.database import SessionDep
from backend.models.comparison import (
    ComparisonResult,
    ComparisonResultPublicWithDimensions
)


router = APIRouter()


@router.get(
    "/result_2/{result_id}",
    response_model=ComparisonResultPublicWithDimensions
)
def get_result(result_id: int, session: SessionDep):
    result = session.get(ComparisonResult, result_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result with id {result_id} found.",
        )

    return result
