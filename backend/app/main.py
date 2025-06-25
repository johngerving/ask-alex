import json
import os
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from app.auth import auth_router
from app.chat import chat_router

from pydantic import BaseModel

import logging

from dotenv import load_dotenv

from app.auth.auth import get_current_user
from app.user_store.store import User

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL is None:
    raise ValueError("FRONTEND_URL environment variable not set")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if SECRET_KEY is None:
    raise ValueError("JWT_SECRET_KEY environment variable not set")


app = FastAPI()
app.include_router(auth_router, prefix="/auth")
app.include_router(chat_router, prefix="/chat")


logger = logging.getLogger("ray.serve")


allow_origins = [FRONTEND_URL]

print("Allow origins: ", allow_origins)

if None in allow_origins:
    raise Exception("FRONTEND_URL environment variable is required")


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


@app.middleware("http")
async def authenticate_user(request: Request, call_next):
    if request.url.path.startswith("/auth"):
        return await call_next(request)

    access_token = request.cookies.get("access_token")
    if not access_token:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    try:
        user = get_current_user(access_token)
    except HTTPException as e:
        return JSONResponse({"detail": e.detail}, status_code=e.status_code)

    if not user:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    request.state.user = user

    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_credentials=True,
)


@app.get("/user")
def public(request: Request) -> JSONResponse:
    user: User = request.state.user

    return JSONResponse(user.model_dump())
