from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.chat_store import ChatStore
import os
from dotenv import load_dotenv
from llama_index.core.memory import Memory

from app.user_store.store import User

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if PG_CONN_STR is None:
    raise ValueError("PG_CONN_STR environment variable not set")

router = APIRouter()

chat_store = ChatStore(PG_CONN_STR)


@router.get("/")
async def get_chats(request: Request):
    """Returns a list of chat IDs for the current user."""
    user: User = request.state.user

    chat_ids = chat_store.get_chats(user)

    return chat_ids


@router.post("/")
async def post_chat(request: Request):
    """Creates a new chat for the current user and returns the chat ID."""
    user: User = request.state.user

    id = chat_store.create(user)

    return JSONResponse({"id": id}, status_code=201)


@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, request: Request):
    """Deletes a chat by ID."""

    user: User = request.state.user

    try:
        chat_store.delete(chat_id, user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{chat_id}/messages")
async def get_chat(chat_id: int, request: Request):
    """Returns the history of a chat by ID."""

    user: User = request.state.user

    try:
        chat = chat_store.find_by_id(chat_id, user)

        if not chat.context:
            chat_history = []
        else:
            memory: Memory = await chat.context.get("memory")
            chat_history = await memory.aget_all()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return chat_history
