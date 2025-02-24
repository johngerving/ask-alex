import os

from rag_pipeline import RagPipeline

from ray import serve
import requests
from fastapi import FastAPI
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

if "" in allow_origins:
    raise Exception("FRONTEND_URL environment variable is required")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["GET", "POST"],
)

class RAGBody(BaseModel):
    query: str

class RAGResponse(BaseModel):
    response: str = ""

@serve.deployment
@serve.ingress(app)
class HaystackQA:
    def __init__(self):
        self.pipeline = RagPipeline()

    @app.post("/")
    async def run(self, body: RAGBody) -> RAGResponse:
        # Run the pipeline with the user's query
        res = self.pipeline.run(body.query)

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
