import logging
from typing import Annotated
from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import SQLModel, Session, create_engine as sqlmodel_create_engine


from backend.models import User, ComparisonResult, DimensionComparisonResult

_engine: Engine | None = None


def create_engine(url: str):
    global _engine
    _engine = sqlmodel_create_engine(
        url,
        connect_args={
            "check_same_thread": False,
        },
    )


def create_db_and_tables() -> None:
    if _engine is None:
        raise Exception("Uninitialized engine")
    SQLModel.metadata.create_all(_engine)


def get_session():
    with Session(_engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
