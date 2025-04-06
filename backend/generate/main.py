import os
import re

from ray import serve
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

from chat_agent import ChatWorkflow
from agno.storage.postgres import PostgresStorage
from agno.run.response import RunResponse

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
class HaystackQA:
    def __init__(self):
        pass

    @app.post("/")
    async def run(self, body: RAGBody) -> RAGResponse:
        from agno.models.message import Message

        # Run the pipeline with the user's query
        messages = []

        if len(body.messages) == 0:
            raise HTTPException(status_code=400, detail="Empty field 'messages'")
        for el in body.messages:
            if el.type == "assistant":
                messages.append(
                    Message(
                        role="assistant",
                        content=el.content,
                    )
                )
            elif el.type == "user":
                messages.append(
                    Message(
                        role="user",
                        content=el.content,
                    )
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Message type must be either 'assistant' or 'user'. Got {el.type}",
                )

        chat_agent = ChatWorkflow(
            user_id="John",
            storage=PostgresStorage(
                table_name="chat_workflows", db_url=os.getenv("PG_CONN_STR")
            ),
        )

        history = messages[:-1]
        message = messages[-1]
        response = chat_agent.run(message=message, history=history).content
        if not isinstance(response, str):
            raise Exception(f"Invalid response type: {type(response)}. Expected str.")

        logger.info(response)

        response = response.strip()
        response = re.sub(r"\n\s*\n", "\n\n", response)

        return RAGResponse(response=response)


haystack_deployment = HaystackQA.bind()
# query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"
