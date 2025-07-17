from email.policy import HTTP
from typing import Optional
import uuid
from fastapi import APIRouter, Cookie, HTTPException, Request
from dotenv import load_dotenv
import os
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
from authlib.integrations.starlette_client import OAuthError
from fastapi.responses import JSONResponse, RedirectResponse
from datetime import datetime, timedelta, timezone
import jwt
import json

from app.user_store.store import User, UserStore


load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL is None:
    raise ValueError("FRONTEND_URL environment variable not set")

# OAuth settings
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")

if OAUTH_CLIENT_ID is None:
    raise ValueError("OAUTH_CLIENT_ID must be set")
if OAUTH_CLIENT_SECRET is None:
    raise ValueError("OAUTH_CLIENT_SECRET must be set")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if SECRET_KEY is None:
    raise ValueError("JWT_SECRET_KEY must be set")

ALGORITHM = "HS256"

PG_CONN_STR = os.getenv("PG_CONN_STR")
if PG_CONN_STR is None:
    raise ValueError("PG_CONN_STR must be set")


# Set up OAuth
config_data = {
    "GOOGLE_CLIENT_ID": OAUTH_CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": OAUTH_CLIENT_SECRET,
}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

router = APIRouter()

user_store = UserStore(PG_CONN_STR)


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        user (User): The user object to encode in the JWT.
        expires_delta (Optional[timedelta]): The expiration time for the JWT. Defaults to 30 minutes.
    """
    to_encode = user.model_dump()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=30))

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Cookie(None)) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload: dict = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload["email"]

        user = user_store.find_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login")
async def login(request: Request):
    request.session.clear()

    redirect_uri = request.url_for("auth")

    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth(request: Request) -> RedirectResponse:
    try:
        access_token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(url=f"{FRONTEND_URL}/chat")

    user = access_token["userinfo"]

    user_email = user["email"]

    # Create JWT token
    access_token_expires = timedelta(days=7)

    user = user_store.create(user_email)
    access_token = create_access_token(user, expires_delta=access_token_expires)

    response = RedirectResponse(url=f"{FRONTEND_URL}/chat")
    response.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        samesite="strict",
        secure=(os.getenv("ENVIRONMENT") != "development"),
    )
    return response


@router.post("/logout")
async def logout(request: Request):
    response = RedirectResponse(url=f"{FRONTEND_URL}/chat", status_code=303)
    response.set_cookie(
        "access_token",
        "",
        httponly=True,
        samesite="strict",
        secure=(os.getenv("ENVIRONMENT") != "development"),
    )
    return response
