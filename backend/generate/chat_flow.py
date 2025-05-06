import functools
import json
import os
from tabnanny import verbose
from textwrap import dedent
from typing import Annotated, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
import haystack
from pydantic import BaseModel, Field
from logging import Logger
import re

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
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.schema import TextNode
from llama_index.core import set_global_handler
from llama_index.core.schema import MetadataMode


from utils import validate_brackets


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


class WorkflowResponse(BaseModel):
    delta: str


class ChatFlow(Workflow):
    """The main workflow for Ask Alex."""

    def __init__(self, logger: Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    llm = GoogleGenAI(model="gemini-2.0-flash")

    small_llm = GoogleGenAI(model="gemini-2.0-flash-lite")

    pg_conn_str = os.getenv("PG_CONN_STR")
    if not pg_conn_str:
        raise ValueError("PG_CONN_STR environment variable not set")

    # Get Postgres credentials from connection string
    pg_url = urlparse(pg_conn_str)
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
        hybrid_search=True,
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

    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=50,
        verbose=True,
    )

    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse", similarity_top_k=50
    )

    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=10,
        llm=small_llm,
        num_queries=3,
        mode="relative_score",
    )

    @step
    async def route_chat_or_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> ChatRouteEvent | RetrievalRouteEvent:
        """Route the message to either chat or retrieval based on the message and history."""
        self.logger.info("Routing chat or retrieval")

        sllm = self.small_llm.as_structured_llm(output_cls=ChatOrRetrieval)

        # Only use the last n messages
        n = 3
        history: List[ChatMessage] = ev.history[max(0, len(ev.history) - n) :]
        for m in history:
            m.content = self._remove_citations(m.content)

        self.logger.info(f"History: {history}")

        # Set global context variables for use in the entire workflow
        await ctx.set("message", ev.message)
        await ctx.set("history", history)

        sources: List[TextNode] = []
        await ctx.set("sources", sources)

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
        self.logger.info("Running step chat")

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

        handler = agent.run(message, chat_history=history)

        response = ""

        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.response:
                ctx.write_event_to_stream(
                    WorkflowResponse(delta=ev.delta, response=ev.response)
                )

        response = handler

        return StopEvent(result=response)

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalRouteEvent) -> StopEvent:
        """Handle the retrieval route by searching the knowledge base for information."""
        self.logger.info("Running retrieval step")

        message: ChatMessage = await ctx.get("message")
        history: List[ChatMessage] = await ctx.get("history")

        tools = [
            FunctionTool.from_defaults(
                fn=self._think,
                name="think",
            ),
            self._make_search_tool(ctx),
        ]

        agent = FunctionAgent(
            llm=self.llm,
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Make your responses comprehensive, informative, and thorough, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.

                Formulate an answer to user queries.

                Follow these steps:
                1. Think step-by-step about the query with the `think(thought)` tool, expanding with more context if necessary.
                2. ALWAYS use the `search_knowledge_base(query)` tool to get relevant documents.
                3. Search the knowledge base as many times as you need to obtain relevant documents.
                4. Use the retrieved documents to write a comprehensive answer to the query, discarding irrelevant documents. Provide inline citations of each document you use.

                ## Using the think tool

                Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
                - Create a plan of action
                - List the specific rules that apply to the current request
                - Check if all required information is collected
                - Verify that the planned action complies with all policies
                - Iterate over tool results for correctness

                Here is an example of what to iterate over inside the think tool:
                <think_tool_example>
                I need to search the knowledge base for information about redwood trees.
                I should verify that the context is sufficient to answer a question about redwood trees comprehensively once I'm done.
                - Plan: collect missing info, verify relevancy, formulate answer
                </think_tool_example>

                Finally, here are a set of rules that you MUST follow:
                <rules>
                - Use the `think(thought)` tool to reason about the current request and develop a plan of action.
                - Use the `search_knowledge_base(query)` tool to retrieve document chunks from your knowledge base before answering the query.
                - Do not use phrases like "based on the information provided" or "from the knowledge base".
                - Do not show your internal planning to the user. Only use the `think(thought)` tool to do so.
                - Always provide inline citations for any information you use to formulate your answer, citing the id field of the chunk you used. DO NOT hallucinate a chunk id.
                    - Example 1: If you are citing a document with the id "asdfgh", you should write something like, "Apples fall to the ground in autum [asdfgh]."
                    - Example 2: If you are citing two documents with the ids "asdfgh" and "qwerty", you should write something like, "The sun rises in the east and sets in the west. [asdfgh][qwerty]."
                </rules>
                """
            ),
            tools=tools,
        )

        handler = agent.run(message, chat_history=history)

        full_raw_response = ""
        buffer = ""  # Buffer for potentially incomplete citations

        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.delta:
                # Prepend any leftover buffer from the previous delta
                current_chunk = buffer + ev.delta
                buffer = ""  # Clear buffer for now

                # Use regex to find citation patterns within chunk
                # The capture group not only splits the citation from the rest of the text, but also captures the citation
                citation_pattern = re.compile(r"(\[[^\]]*\])")
                # Pattern for incomplete citation start at the end of the chunk
                incomplete_citation_pattern = re.compile(r"(\[[^\]]*)$")

                parts = citation_pattern.split(current_chunk)
                delta_to_send = ""

                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is a complete citation found by a split
                        full_raw_response += part
                    else:  # This is text between citations or the final part
                        # Check if this non-citation part ends with an incomplete citation start
                        incomplete_match = incomplete_citation_pattern.search(part)
                        if incomplete_match:
                            # Hold back the incomplete part in the buffer
                            incomplete_part = incomplete_match.group(1)
                            text_part = part[: -len(incomplete_part)]
                            buffer = incomplete_part  # Store for next delta
                            full_raw_response += text_part  # Accumulate text part
                            delta_to_send += text_part
                        else:
                            # No incomplete citation found, send/accumulate normally
                            full_raw_response += part
                            delta_to_send += part

                # Send the processed delta (without incomplete citation ends)
                if delta_to_send:
                    ctx.write_event_to_stream(WorkflowResponse(delta=delta_to_send))

        # Process any remaining buffer content
        if buffer:
            self.logger.warning(f"Streaming ended with incomplete citation: {buffer}")
            full_raw_response += buffer

        final_formatted_response = await self._generate_citations(
            ctx, full_raw_response
        )
        self.logger.info(f"Final response: {final_formatted_response}")

        return StopEvent(result=final_formatted_response)

    def _make_search_tool(self, ctx: Context):
        async def search_knowledge_base(
            query: Annotated[str, "The query to search the knowledge base for"],
        ) -> str:
            """Search the knowledge base for relevant chunks from documents."""
            self.logger.info(f"Running search_knowledge_base with query: {query}")
            # Use the retriever to get relevant nodes
            try:
                nodes = await self.retriever.aretrieve(query)
                self.logger.info(f"Retrieved {len(nodes)} nodes")

                sources: List[TextNode] = await ctx.get("sources")
                sources = sources + nodes
                await ctx.set("sources", sources)

                content = ""

                for node in nodes:
                    content += (
                        "<chunk id={doc_id}>\n" "{content}\n" "</chunk>\n"
                    ).format(
                        doc_id=node.node_id[:8],
                        content=node.get_content(metadata_mode=MetadataMode.LLM),
                    )
                    self.logger.info(node.get_content(metadata_mode=MetadataMode.EMBED))

                print(content)

            except Exception as e:
                self.logger.error(e)
                raise

            return content

        return FunctionTool.from_defaults(
            async_fn=search_knowledge_base,
        )

    def _think(self, thought: Annotated[str, "A thought to think about."]):
        """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."""
        self.logger.info(f"Thought: {thought}")

    async def _generate_citations(self, ctx: Context, text: str) -> str:
        """Generate citations for a text response based on the sources used and replace them in the text.

        Args:
            ctx (Context): The context object containing the sources.
            text (str): The text response to generate citations for.

        Returns:
            str: The text response with citations generated.
        """
        sources: List[TextNode] = await ctx.get("sources")

        sources_used: List[TextNode] = []
        for source in sources:
            if source.node_id[:8] in text and not any(
                [s.node_id == source.node_id for s in sources_used]
            ):
                sources_used.append(source)

        citations = re.findall("\[[^\]]*\]", text)
        for citation in citations:
            try:
                matching_citation_idx = list(
                    s.node_id[:8] in citation for s in sources_used
                ).index(True)

                download_link = sources_used[matching_citation_idx].metadata.get(
                    "download_link"
                )
                if download_link is None:
                    raise ValueError("download_link not found")

                text = text.replace(
                    citation,
                    f"[[{matching_citation_idx+1}]]({sources_used[matching_citation_idx].metadata.get('download_link')})",
                )
            except ValueError as e:
                self.logger.info(
                    f"Citation '{citation}' had the following ValueError: {e}"
                )
                text = text.replace(citation, "")

        return text

    def _remove_citations(self, text: str) -> str:
        """Remove citations from a text response so as not to confuse the LLM."""
        # Match patterns of the form [citation](link)
        citations_removed = re.sub("\[(.*?)\]\((.*?)\)", "", text)
        return citations_removed
