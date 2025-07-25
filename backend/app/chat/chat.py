from typing import List, get_args
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.chat_store import ChatStore
import os
from dotenv import load_dotenv
from llama_index.core.memory import Memory
from llama_index.core.workflow.context import Context
from sse_starlette import EventSourceResponse
from pydantic import BaseModel
from app.agent.agents.agent import Agent, FinalAnswerEvent, StreamEvent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.workflow import JsonSerializer, JsonPickleSerializer
from llama_index.core.agent.workflow import AgentStream, ToolCall
from openai.types.chat import ChatCompletionMessageToolCall
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
import json
import logging
from app.user_store.store import User
from app.agent.utils import Source, generate_citations, message_to_tool_selections
from app.agent.memory.postgresql_chat_store import PostgreSQLChatStore
from app.agent.memory.custom_memory import CustomMemory

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
if PG_CONN_STR is None:
    raise ValueError("PG_CONN_STR environment variable not set")
SQLALCHEMY_CONN_STR = os.getenv("SQLALCHEMY_CONN_STR")
if SQLALCHEMY_CONN_STR is None:
    raise ValueError("SQLALCHEMY_CONN_STR environment variable not set")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
chat_store = ChatStore(PG_CONN_STR)
tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


class ChatInformation(BaseModel):
    """Model for chat information."""

    id: str
    title: str


@router.get("/")
async def get_chats(request: Request) -> List[ChatInformation]:
    """Returns a list of chat information for the current user."""
    user: User = request.state.user

    chats = chat_store.get_chats(user)

    chat_information: List[ChatInformation] = []

    for chat in chats:
        w = Agent(logger=None)

        if chat.context:
            ctx = Context.from_dict(w, chat.context, serializer=JsonSerializer())
        else:
            ctx = Context(w)

        chat_title = await ctx.get("chat_title", "New Chat")

        chat_information.append(
            ChatInformation(
                id=chat.id,
                title=chat_title,
            )
        )

    return chat_information


@router.post("/")
async def post_chat(request: Request):
    """Creates a new chat for the current user and returns the chat ID."""
    user: User = request.state.user

    id = chat_store.create(user)

    return JSONResponse({"id": id}, status_code=201)


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, request: Request):
    """Deletes a chat by ID."""

    user: User = request.state.user

    try:
        chat_store.delete(chat_id, user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{chat_id}/messages")
async def get_chat(chat_id: str, request: Request):
    """Returns the history of a chat by ID."""

    user: User = request.state.user

    memory = CustomMemory.from_defaults(
        token_limit=127000,
        session_id=f"{user.id}-{chat_id}",
        async_database_uri=SQLALCHEMY_CONN_STR,
    )

    chat_history = await memory.aget_all()

    chat = chat_store.find_by_id(chat_id, user)
    w = Agent(logger=None)

    if chat.context:
        ctx = Context.from_dict(w, chat.context, serializer=JsonSerializer())
    else:
        ctx = Context(w)

    retrieved_sources: List[Source] = await ctx.get("retrieved_sources", [])

    display_history: List[RequestMessage | RequestToolCall] = []

    for msg in chat_history:
        if msg.additional_kwargs.get("display", True):
            if msg.role == MessageRole.USER:
                display_history.append(
                    RequestMessage(content=msg.content or "", role="user")
                )
            elif msg.role == MessageRole.ASSISTANT:
                tool_calls = message_to_tool_selections(msg)

                for tool_call in tool_calls:
                    display_history.append(
                        RequestToolCall(
                            id=tool_call.tool_id,
                            name=tool_call.tool_name,
                            kwargs=tool_call.tool_kwargs or {},
                        )
                    )

                formatted_content = generate_citations(
                    retrieved_sources, msg.content or ""
                )

                # Only display the message if it has content and display is not set to False
                if msg.content:
                    display_history.append(
                        RequestMessage(content=formatted_content, role="assistant")
                    )

    return display_history


class RequestMessage(BaseModel):
    type: str = "message"
    content: str
    role: str


class RequestToolCall(BaseModel):
    type: str = "tool_call"
    id: str
    name: str
    kwargs: dict


class RAGBody(BaseModel):
    message: RequestMessage
    chatId: str


class RAGResponse(BaseModel):
    response: str = ""


@router.post("/messages")
async def run(request: Request) -> EventSourceResponse:
    user: User = request.state.user

    body = RAGBody(**await request.json())

    message: ChatMessage

    # Convert the request body to a LlamaIndex message
    if body.message.role == "assistant":
        message = ChatMessage(
            role="assistant",
            content=body.message.content,
        )
    elif body.message.role == "user":
        message = ChatMessage(
            role="user",
            content=body.message.content,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Message role must be either 'assistant' or 'user'. Got {body.message.role}",
        )

    async def event_generator():
        workflow = Agent(logger=logger, timeout=180, verbose=True)

        chat = chat_store.find_by_id(body.chatId, user)

        if chat.context:
            ctx = Context.from_dict(workflow, chat.context, serializer=JsonSerializer())
        else:
            ctx = Context(workflow)

        try:
            logger.info(f"Running workflow")

            memory = CustomMemory.from_defaults(
                session_id=f"{chat.user_id}-{chat.id}",
                async_database_uri=SQLALCHEMY_CONN_STR,
            )
            handler = workflow.run(message=message, ctx=ctx, memory=memory)

            # Read events from the workflow run
            async for ev in handler.stream_events():
                if await request.is_disconnected():
                    logger.info("Disconnected")
                    break

                # Stream events to the client
                if isinstance(ev, StreamEvent):
                    yield {
                        "event": "delta",
                        "data": format_event(ev.delta),
                    }
                elif isinstance(ev, ToolCall):
                    yield {
                        "event": "tool_call",
                        "data": json.dumps(
                            {
                                "id": ev.tool_id,
                                "name": ev.tool_name,
                                "kwargs": ev.tool_kwargs,
                            }
                        ),
                    }
                elif isinstance(ev, FinalAnswerEvent):
                    yield {
                        "event": "response",
                        "data": format_event(ev.content),
                    }
                else:
                    # logger.info(f"Event: {ev}")
                    pass

            logger.info(f"Got to end of event stream")

            # Get the final response from the workflow
            await handler

            try:
                chat_store.set_context(chat.id, ctx, user)
            except Exception as e:
                logger.error(f"Error setting context: {e}")

        except Exception as e:
            logger.info(f"Exception: {e}")
            print(f"Exception: {e}")
            yield {"event": "error", "data": format_event("Internal server error")}

    # Start event stream
    return EventSourceResponse(event_generator())


def format_event(event: str) -> str:
    """Format the event as JSON to be sent to the client"""
    return json.dumps({"v": event})
