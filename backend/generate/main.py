import os
import re
from typing import Generator, List

from fastapi.responses import StreamingResponse
from ray import serve
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from chat_flow import ChatFlow
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)

# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI()

import logging

logger = logging.getLogger("ray.serve")

logger.info(f"Frontend URL: {os.getenv('FRONTEND_URL')}")

allow_origins = [os.getenv("FRONTEND_URL")]

if None in allow_origins:
    raise Exception("FRONTEND_URL environment variable is required")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST"],
)


class RequestMessage(BaseModel):
    content: str
    type: str


class RAGBody(BaseModel):
    messages: list[RequestMessage]


class RAGResponse(BaseModel):
    response: str = ""


@serve.deployment
@serve.ingress(app)
class ChatQA:
    def __init__(self):
        self.workflow = ChatFlow(timeout=60, verbose=True)
        self.logger = logging.getLogger("ray.serve")

    @app.post("/")
    async def run(self, request: Request) -> RAGResponse:
        body = RAGBody(**await request.json())

        # Run the pipeline with the user's query
        messages: List[ChatMessage] = []

        if len(body.messages) == 0:
            raise HTTPException(status_code=400, detail="Empty field 'messages'")
        for el in body.messages:
            if el.type == "assistant":
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=el.content,
                    )
                )
            elif el.type == "user":
                messages.append(
                    ChatMessage(
                        role="user",
                        content=el.content,
                    )
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Message type must be either 'assistant' or 'user'. Got {el.type}",
                )

        history = messages[:-1]
        message = messages[-1]

        async def event_generator():
            try:
                self.logger.info(f"Running workflow")
                handler = self.workflow.run(message=message, history=history)

                async for ev in handler.stream_events():
                    if await request.is_disconnected():
                        self.logger.info("Disconnected")
                        break

                    self.logger.info(f"Event: {ev}")

                    yield {"event": "message", "data": str(ev)}

                    # if isinstance(ev, StopEvent):
                    #     self.logger.info(f"StopEvent: {ev.result}")
                    #     result = ev.result
                    #     result = result.strip()
                    #     result = re.sub(r"\n\s*\n", "\n\n", result)

                    #     yield {"event": "message", "data": str(ev.result)}

                await handler
                await handler
                self.logger.info(f"Got to end of event stream")
                self.logger.info(f"Handler: {handler}")
                result = handler
                self.logger.info(f"Result: {result}")

                yield {"event": "message", "data": "test"}

            except Exception as e:
                self.logger.info(f"Exception: {e}")

        return EventSourceResponse(event_generator())


deployment = ChatQA.bind()
