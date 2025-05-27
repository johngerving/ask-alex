import json
import os
import re
from typing import Generator, List
from uuid import uuid4

from fastapi.responses import StreamingResponse
from ray import serve
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from chat_flow import ChatFlow, WorkflowReasoning, WorkflowResponse
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from langfuse.llama_index import LlamaIndexInstrumentor

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
    role: str


class RAGBody(BaseModel):
    messages: list[RequestMessage]


class RAGResponse(BaseModel):
    response: str = ""


@serve.deployment
@serve.ingress(app)
class ChatQA:
    def __init__(self):
        self.logger = logging.getLogger("ray.serve")
        self.workflow = ChatFlow(logger=self.logger, timeout=120, verbose=True)
        self.instrumentor = LlamaIndexInstrumentor(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

    @app.post("/")
    async def run(self, request: Request) -> EventSourceResponse:
        body = RAGBody(**await request.json())

        # Run the pipeline with the user's query
        messages: List[ChatMessage] = []

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
                with self.instrumentor.observe():
                    self.logger.info(f"Running workflow")
                    handler = self.workflow.run(message=message, history=history)

                    # Read events from the workflow run
                    async for ev in handler.stream_events():
                        if await request.is_disconnected():
                            self.logger.info("Disconnected")
                            break

                        # Stream events to the client
                        if isinstance(ev, WorkflowResponse):
                            yield {
                                "event": "delta",
                                "data": self._format_event(ev.delta),
                            }
                        elif isinstance(ev, WorkflowReasoning):
                            yield {
                                "event": "reasoning",
                                "data": self._format_event(ev.delta),
                            }
                        else:
                            self.logger.info(f"Event: {ev}")

                    self.logger.info(f"Got to end of event stream")

                    # Get the final response from the workflow
                    result = await handler
                    self.logger.info(f"Result: {result}")

                    # Stream the final response to the client
                    yield {"event": "response", "data": self._format_event(str(result))}

                self.instrumentor.flush()

            except Exception as e:
                self.logger.info(f"Exception: {e}")

        # Start event stream
        return EventSourceResponse(event_generator())

    def _format_event(self, event: str) -> str:
        """Format the event as JSON to be sent to the client"""
        return json.dumps({"v": event})


deployment = ChatQA.bind()
