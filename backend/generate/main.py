import os

from haystack.dataclasses import ChatMessage
from rag_pipeline import RagPipeline

from ray import serve
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI()

import logging
logger = logging.getLogger("ray.serve")

logger.info(os.getenv("FRONTEND_URL"))

allow_origins = [
    os.getenv("FRONTEND_URL")
]

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
        self.pipeline = RagPipeline()

    @app.post("/")
    async def run(self, body: RAGBody) -> RAGResponse:
        from haystack.dataclasses import ChatMessage

        # Run the pipeline with the user's query
        messages = []

        if len(body.messages) == 0:
            raise HTTPException(status_code=400, detail="Empty field 'messages'")
        for el in body.messages:
            if el.type == "assistant":
                messages.append(ChatMessage.from_assistant(el.content))
            elif el.type == "user":
                messages.append(ChatMessage.from_user(el.content))
            else:
                raise HTTPException(status_code=400, detail=f"Message type must be either 'assistant' or 'user'. Got {el.type}")
        res = self.pipeline.run(messages)

        # Return different reply based on whether chat route or RAG route was followed
        if "rag_llm" in res:
            replies = res["rag_llm"]["replies"]
        elif "chat_llm" in res:
            replies = res["chat_llm"]["replies"]
        else:
            raise Exception("No LLM output found")

        if replies:
            return RAGResponse(response=replies[0].text)

        return RAGResponse()

haystack_deployment = HaystackQA.bind()
# query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"
