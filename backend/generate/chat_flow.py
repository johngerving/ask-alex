import json
import logging
import os
from textwrap import dedent
from typing import Annotated, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
import haystack
from pydantic import BaseModel, Field

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import yaml


class WorkflowStartEvent(StartEvent):
    """Event to start the workflow."""

    message: ChatMessage
    history: List[ChatMessage]


class ChatRouteEvent(Event):
    pass


class RetrievalRouteEvent(Event):
    pass


class ChatOrRetrieval(BaseModel):
    """Data model for routing between chat and retrieval."""

    reasoning: str = Field(
        ..., description="The reasoning behind the routing decision."
    )
    route: Literal["chat", "retrieval"] = Field(
        ..., description="The route to take: either 'chat' or 'retrieval'."
    )


class ChatFlow(Workflow):
    """The main workflow for Ask Alex."""

    llm = GoogleGenAI(model="gemini-2.0-flash")

    small_llm = GoogleGenAI(model="gemini-2.0-flash-lite")

    # Get Postgres credentials from connection string
    pg_url = urlparse(os.getenv("PG_CONN_STR"))
    host = pg_url.hostname
    port = pg_url.port
    database = pg_url.path[1:]
    user = pg_url.username
    password = pg_url.password

    # Vector store to store chunks + embeddings in

    vector_store = PGVectorStore.from_params(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name="llamaindex_docs",
        schema_name="public",
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )

    retriever = index.as_retriever(similarity_top_k=8)

    @step
    async def route_chat_or_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> ChatRouteEvent | RetrievalRouteEvent:
        """Route the message to either chat or retrieval based on the message and history."""
        sllm = self.small_llm.as_structured_llm(output_cls=ChatOrRetrieval)

        # Set global context variables for use in the entire workflow
        await ctx.set("message", ev.message)
        await ctx.set("history", ev.history)

        # Call the LLM to determine the route
        json_obj: Dict[str, str] = sllm.chat(
            messages=[
                ChatMessage(
                    role="system",
                    content=dedent(
                        """\
                        You are a router agent tasked with deciding whether to route a user message to a chat agent or a retrieval agent.
                        You will receive a list of previous messages and the current user message.
                        Use the following steps:
                        1. Output a thought in which you reason through whether to route the message to the chat agent or the retrieval agent.
                        2. Output the route you have chosen: either "chat" or "retrieval".
                        """
                    ),
                ),
                ev.message,
            ]
        ).raw.dict()

        # Parse the JSON response from the LLM as a ChatOrRetrieval object
        response = ChatOrRetrieval(**json_obj)

        # Route query based on the LLM's response
        if response.route == "chat":
            return ChatRouteEvent()
        else:
            return RetrievalRouteEvent()

    @step
    async def chat(self, ctx: Context, ev: ChatRouteEvent) -> StopEvent:
        """Handle the chat route by generating a response using the LLM."""

        message = await ctx.get("message")
        history = await ctx.get("history")

        agent = FunctionAgent(
            llm=self.llm,
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses. Do not provide information beyond what you are given in the context.
                """
            ),
        )

        response = await agent.run(message, chat_history=history)

        return StopEvent(result=response)

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalRouteEvent) -> StopEvent:
        """Handle the retrieval route by searching the knowledge base for information."""

        message: ChatMessage = await ctx.get("message")
        history: List[ChatMessage] = await ctx.get("history")

        # Create a retriever tool to search the knowledge base
        retriever_tool = RetrieverTool.from_defaults(
            retriever=self.retriever,
            name="search_knowledge_base",
            description="Search the knowledge base for relevant documents.",
        )

        tools = [
            FunctionTool.from_defaults(
                self._search_knowledge_base,
                name="search_knowledge_base",
            )
        ]

        agent = FunctionAgent(
            llm=self.llm,
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Make your responses comprehensive, informative, and thorough, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.

                Formulate an answer to user queries.

                Follow these steps:
                1. Think step-by-step about the query, expanding with more context if necessary.
                2. ALWAYS use the `search_knowledge_base(query)` tool to get relevant documents.
                3. Search the knowledge base as many times as you need to obtain relevant documents.
                4. Use the retrieved documents to write a comprehensive answer to the query, discarding irrelevant documents. Provide inline citations of each document you use.

                Finally, here are a set of rules that you MUST follow:
                <rules>
                - Use the `search_knowledge_base(query)` tool to retrieve documents from your knowledge base before answering the query.
                - Do not use phrases like "based on the information provided" or "from the knowledge base".
                - Always provide inline citations for any information you use to formulate your answer. These should be in the format [doc_id], where doc_id is the "doc_id" field of the document you are citing. Multiple citations should be written as [id_1][id_2].
                </rules>
                """
            ),
            tools=tools,
        )

        response = await agent.run(message, chat_history=history)

        return StopEvent(result=response)

    def _search_knowledge_base(
        self, query: Annotated[str, "The query to search the knowledge base for"]
    ) -> str:
        """Search the knowledge base for relevant documents."""
        # Use the retriever to get relevant nodes
        nodes = self.retriever.retrieve(query)

        json_obj = [
            {"doc_id": node.node_id[:8], "content": node.text} for node in nodes
        ]

        print(json.dumps(json_obj, indent=2))

        return json.dumps(json_obj, indent=2)
