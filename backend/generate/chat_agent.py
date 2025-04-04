import json
import logging
import os
from textwrap import dedent
from typing import Iterator, Optional, Literal, List
from agno.agent import Agent, AgentMemory
from agno.run.response import RunEvent, RunResponse
from agno.utils.log import logger
from agno.models.google import Gemini
from agno.models.message import Message
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.workflow import Workflow
from agno.tools import tool
import haystack
from pydantic import BaseModel, Field

from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline


class Action(BaseModel):
    route: Literal["chat", "retrieval"] = Field(
        ...,
        description="The action to take. Choose chat if the user message does not require additional information to answer. Choose retrieval if you need information that cannot be found within the available context, or if the user specifically requests it.",
    )


def search_knowledge_base(query: str) -> str:
    """Use this function to search the knowledge base for information about a query.

    Args:
        query: The query to search for.

    Returns:
        str: A string containing the response from the knowledge base.
    """

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
        table_name="haystack_docs",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
    )

    # Retriever to get embedding vectors based on query
    retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=8)

    pipeline = Pipeline()

    pipeline.add_component(
        "embedder", SentenceTransformersTextEmbedder()
    )  # Get query vector embedding
    pipeline.add_component("retriever", retriever)

    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    res = pipeline.run({"embedder": {"text": query}})

    documents: List[haystack.Document] = res["retriever"]["documents"]
    documents = [
        {"index": i, "content": documents[i].content} for i in range(len(documents))
    ]
    logger = logging.getLogger("ray.serve")
    logger.info(documents)

    return json.dumps(documents, indent=2)


class ChatWorkflow(Workflow):
    def run(self, message: str, history: List[Message]) -> RunResponse:
        router = self.get_router_agent(history=history)

        response = router.run(message)
        if not isinstance(response.content, Action):
            raise Exception(f"Invalid Action response: {response.content}")
        action: Action = response.content

        response: RunResponse

        if action.route == "chat":
            chat_agent = self.get_chat_agent(history=history)
            response = chat_agent.run(message)
        elif action.route == "retrieval":
            retrieval_agent = self.get_retrieval_agent(history=history)
            response = retrieval_agent.run(message)

        return response

    def get_router_agent(self, history: List[Message]) -> Agent:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-lite"),
            instructions=dedent(
                """\
                You are a router agent designed to decide which action to take based on the current user message and the chat history.
                """
            ),
            memory=AgentMemory(messages=history),
            response_model=Action,
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

    def get_chat_agent(self, history: List[Message]) -> Agent:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.
                """
            ),
            memory=AgentMemory(messages=history),
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

    def get_retrieval_agent(self, history: List[Message]) -> Agent:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.

                Query the knowledge base, then formulate an answer to the query.

                Follow these steps:
                1. Think step-by-step about the query, expanding with more context if necessary.
                2. ALWAYS use the `search_knowledge_base(query)` tool to get relevant documents.
                3. Use the retrieved documents to write a comprehensive answer to the query, discarding irrelevant documents. Provide inline citations of each document you use.

                Finally, here are a set of rules that you MUST follow:
                <rules>
                - Use the `search_knowledge_base(query)` tool to retrieve documents from your knowledge base before answering the query.
                - Do not use phrases like "based on the information provided" or "from the knowledge base".
                - Always provide inline citations for any information you use to formulate your answer. These should be in the format [i], where [i] is the number of the document you are citing. Multiple citations should be written as [i][j].
                </rules>
                """
            ),
            memory=AgentMemory(messages=history),
            tools=[search_knowledge_base],
            markdown=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

    def retrieval_generation(self, query: str, history: List[Message]) -> RunResponse:
        agent = self.get_retrieval_agent(history=history)

        response = agent.run(query)

        return response
