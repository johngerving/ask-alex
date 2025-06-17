from fastapi import APIRouter
from dotenv import load_dotenv
import os
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
from authlib.integrations.starlette_client import OAuthError
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from app.session_store import SessionStore
import json


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

session_store = SessionStore(PG_CONN_STR, max_age=60 * 60 * 24 * 7)

router = APIRouter()


@router.post("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth(request: Request) -> RedirectResponse:
    try:
        access_token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(url=FRONTEND_URL)

    user = access_token["userinfo"]

    session_key = session_store.new(user)

    response = RedirectResponse(url=FRONTEND_URL)
    response.set_cookie(
        "session",
        session_key,
        httponly=True,
        samesite="strict",
        secure=(os.getenv("ENVIRONMENT") != "development"),
    )
    return response


@router.get("/user")
def public(request: Request) -> JSONResponse:
    session_key = request.cookies.get("session")
    if not session_key:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user = session_store.get(session_key)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    return JSONResponse(user)


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.pop("session", None)
    return RedirectResponse(url="/")
