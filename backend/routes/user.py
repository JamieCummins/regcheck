from datetime import timedelta
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import select

from backend.core.auth import (
    Token,
    create_access_token,
    get_user_name_from_token,
    hash_password,
    verify_password,
    verify_password_dummy,
    oauth2_scheme,
)
from backend.core.database import SessionDep
from backend.core import config
from backend.models.user import User, UserCreate, UserPublic, UserPublicWithComparisons


router = APIRouter()


@router.post("/user", response_model=UserPublic)
def create_user(user_in: Annotated[UserCreate, Form()], session: SessionDep):
    hashed_password = hash_password(user_in.password)
    user = User(**user_in.model_dump(), hashed_password=hashed_password)

    session.add(user)
    session.commit()
    return user


def authenticate_user(
    username: str, password: str, session: SessionDep
) -> Optional[User]:
    user = session.exec(select(User).where(User.name == username)).first()
    if user:
        if verify_password(password, user.hashed_password):
            return user
    else:
        # Dummy password verification to not show the user does not exist
        verify_password_dummy(password)
    return None


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    settings: Annotated[config.Settings, Depends(config.get_settings)],
    session: SessionDep,
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    username = get_user_name_from_token(token, settings.session_secret)
    if username is None:
        raise credentials_exception

    user = session.exec(select(User).where(User.name == username)).first()
    if user is None:
        raise credentials_exception
    return user


@router.post("/token")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
    settings: Annotated[config.Settings, Depends(config.get_settings)],
) -> Token:
    user = authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.name},
        expires_delta=access_token_expires,
        secret_key=settings.session_secret,
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/user/me", response_model=UserPublicWithComparisons)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    return current_user
