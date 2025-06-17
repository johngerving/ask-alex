import json
import os
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.auth import auth_router

from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from agent.retrieval_agent import StreamEvent
from agent.agent import Agent
from llama_index.core.llms import ChatMessage
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from langfuse import get_client
import logging

from dotenv import load_dotenv

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL is None:
    raise ValueError("FRONTEND_URL environment variable not set")

COOKIE_SECRET = os.getenv("COOKIE_SECRET")
if COOKIE_SECRET is None:
    raise ValueError("COOKIE_SECRET environment variable not set")


app = FastAPI()
app.include_router(auth_router, prefix="/auth")


logger = logging.getLogger("ray.serve")


allow_origins = [FRONTEND_URL]

if None in allow_origins:
    raise Exception("FRONTEND_URL environment variable is required")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST"],
    allow_credentials=True,
)

app.add_middleware(SessionMiddleware, secret_key=COOKIE_SECRET)


class RequestMessage(BaseModel):
    content: str
    role: str


class RAGBody(BaseModel):
    messages: list[RequestMessage]


class RAGResponse(BaseModel):
    response: str = ""


logger = logging.getLogger("ray.serve")
# workflow = ChatFlow(logger=logger, timeout=120, verbose=True)
LlamaIndexInstrumentor().instrument()


@app.post("/")
async def run(request: Request) -> EventSourceResponse:
    body = RAGBody(**await request.json())

    # Run the pipeline with the user's query
    messages: List[ChatMessage] = []

    workflow = Agent(logger=logger, timeout=120, verbose=True)

    # Convert the request body to a list of LlamaIndex messages
    if len(body.messages) == 0:
        raise HTTPException(status_code=400, detail="Empty field 'messages'")
    for el in body.messages:
        if el.role == "assistant":
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=el.content,
                )
            )
        elif el.role == "user":
            messages.append(
                ChatMessage(
                    role="user",
                    content=el.content,
                )
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Message role must be either 'assistant' or 'user'. Got {el.role}",
            )

    history = messages[:-1]
    message = messages[-1]

    async def event_generator():
        try:
            langfuse = get_client()
            with langfuse.start_as_current_span(name="ask_alex_agent_trace"):
                logger.info(f"Running workflow")
                handler = workflow.run(message=message, history=history)

                # Read events from the workflow run
                async for ev in handler.stream_events():
                    if await request.is_disconnected():
                        logger.info("Disconnected")
                        break

                    # Stream events to the client
                    if isinstance(ev, StreamEvent):
                        yield {
                            "event": "delta",
                            "data": _format_event(ev.delta),
                        }
                    else:
                        # logger.info(f"Event: {ev}")
                        pass

                logger.info(f"Got to end of event stream")

                # Get the final response from the workflow
                result = await handler
                logger.info(f"Result: {result}")

                # Stream the final response to the client
                yield {"event": "response", "data": _format_event(str(result))}
            langfuse.flush()

        except Exception as e:
            logger.info(f"Exception: {e}")
            yield {"event": "error", "data": {"error": "Internal server error"}}

    # Start event stream
    return EventSourceResponse(event_generator())


def _format_event(event: str) -> str:
    """Format the event as JSON to be sent to the client"""
    return json.dumps({"v": event})
