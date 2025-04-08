import os
import re
from typing import List

from ray import serve
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

from chat_flow import ChatFlow
from llama_index.core.llms import ChatMessage

from agno.storage.postgres import PostgresStorage

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

    @app.post("/")
    async def run(self, body: RAGBody) -> RAGResponse:

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
        response = str(await self.workflow.run(message=message, history=history))

        if not isinstance(response, str):
            raise Exception(f"Invalid response type: {type(response)}. Expected str.")

        logger.info(response)

        response = response.strip()
        response = re.sub(r"\n\s*\n", "\n\n", response)

        return RAGResponse(response=response)


deployment = ChatQA.bind()
