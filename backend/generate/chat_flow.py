import json
import logging
import os
from textwrap import dedent
from typing import Annotated, Dict, Iterator, Optional, Literal, List
import haystack
from pydantic import BaseModel, Field

from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline

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
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage

document_store = PgvectorDocumentStore(
    connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
    table_name="haystack_docs",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=False,
    search_strategy="hnsw",
)

embedder = SentenceTransformersTextEmbedder()
embedder.warm_up()

# Retriever to get embedding vectors based on query
retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=8)

pipeline = Pipeline()

pipeline.add_component("embedder", embedder)  # Get query vector embedding
pipeline.add_component("retriever", retriever)

pipeline.connect("embedder.embedding", "retriever.query_embedding")


def search_knowledge_base(query: Annotated[str, "The query to search for."]) -> str:
    """Use this function to search the knowledge base for information about a query."""

    res = pipeline.run({"embedder": {"text": query}})

    documents: List[haystack.Document] = res["retriever"]["documents"]
    documents = [
        {"doc_id": documents[i].id[:8], "content": documents[i].content}
        for i in range(len(documents))
    ]
    logger = logging.getLogger("ray.serve")
    logger.info(documents)

    return json.dumps(documents, indent=2)


tool = FunctionTool


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

        message = await ctx.get("message")
        history = await ctx.get("history")

        tool = FunctionTool.from_defaults(search_knowledge_base)

        agent = ReActAgent(
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses. Do not provide information beyond what you are given in the context.
                Formulate an answer to user queries.
                Follow these steps:
                1. Think step-by-step about the query, expanding with more context if necessary.
                2. ALWAYS use the `search_knowledge_base(query)` tool to get relevant documents.
                3. Use the retrieved documents to write a comprehensive answer to the query, discarding irrelevant documents. Provide inline citations of each document you use.
                Finally, here are a set of rules that you MUST follow:
                <rules>
                - Use the `search_knowledge_base(query)` tool to retrieve documents from your knowledge base before answering the query.
                - Do not use phrases like "based on the information provided" or "from the knowledge base".
                - Always provide inline citations for any information you use to formulate your answer. These should be in the format [doc_id], where doc_id is the id of the document you are citing. Multiple citations should be written as [id_1][id_2].
                </rules>
                """
            ),
            llm=self.llm,
            tools=[tool],
        )

        from llama_index.core.agent.workflow import AgentStream, ToolCallResult

        handler = agent.run(message, chat_history=history)

        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)

        response = await handler

        return StopEvent(result=response)
