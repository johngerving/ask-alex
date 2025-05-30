import functools
import json
import os
from typing import Annotated, Any, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
from llama_cloud import Llm
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
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.schema import TextNode
from llama_index.core import set_global_handler
from llama_index.core.schema import MetadataMode
from llama_index.core.llms import LLM


from tools.search_knowledge_base import make_retrieve_chunks_tool
from tools.search_documents import make_document_search_tool
from prompts import BASE_PROMPT, RETRIEVAL_AGENT_PROMPT, ROUTER_AGENT_PROMPT
from utils import generate_citations, remove_citations


class WorkflowStartEvent(StartEvent):
    """Event to start the workflow."""

    message: ChatMessage
    history: List[ChatMessage]


class ChatOrRetrievalRouteEvent(Event):
    pass


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


class WorkflowReasoning(BaseModel):
    delta: str


class ChatFlow(Workflow):
    """The main workflow for Ask Alex."""

    def __init__(self, logger: Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    @step
    async def setup(
        self, ctx: Context, ev: WorkflowStartEvent
    ) -> ChatOrRetrievalRouteEvent:
        self.logger.info("Running setup step")

        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            is_chat_model=True,
            is_function_calling_model=True,
        )
        # llm = OpenAILike(
        #     model="cognitivecomputations/Qwen3-235B-A22B-AWQ",
        #     api_key=os.getenv("LLM_API_KEY"),
        #     api_base=os.getenv("LLM_API_BASE"),
        #     is_chat_model=True,
        #     is_function_calling_model=True,
        # )

        small_llm = GoogleGenAI(
            model="gemini-2.0-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY"),
            is_chat_model=True,
            is_function_calling_model=True,
        )

        await ctx.set("llm", llm)
        await ctx.set("small_llm", small_llm)
        await ctx.set("logger", self.logger)

        # Only use the last n messages
        n = 3
        history: List[ChatMessage] = ev.history[max(0, len(ev.history) - n) :]
        for m in history:
            m.content = remove_citations(m.content)

        self.logger.info(f"History: {history}")

        # Set global context variables for use in the entire workflow
        await ctx.set("message", ev.message)
        await ctx.set("history", history)

        sources: List[TextNode] = []
        await ctx.set("sources", sources)

        return ChatOrRetrievalRouteEvent()

    @step
    async def route_chat_or_retrieval(
        self, ctx: Context, ev: ChatOrRetrievalRouteEvent
    ) -> ChatRouteEvent | RetrievalRouteEvent:
        """Route the message to either chat or retrieval based on the message and history."""
        self.logger.info("Routing chat or retrieval")

        small_llm: LLM = await ctx.get("small_llm")

        sllm = small_llm.as_structured_llm(output_cls=ChatOrRetrieval)

        message: ChatMessage = await ctx.get("message")

        # Call the LLM to determine the route
        json_obj: Dict[str, str] = sllm.chat(
            messages=[
                ChatMessage(role="system", content=ROUTER_AGENT_PROMPT),
                message,
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

        message: ChatMessage = await ctx.get("message")
        history: List[ChatMessage] = await ctx.get("history")

        llm: LLM = await ctx.get("llm")

        agent = FunctionAgent(
            llm=llm,
            system_prompt=BASE_PROMPT,
        )

        handler = agent.run(message, chat_history=history)

        response = ""

        # Stream response from agent
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

        message = ChatMessage(role="user", content=message.content)

        llm: LLM = await ctx.get("llm")

        tools = [
            await make_retrieve_chunks_tool(ctx),
            await make_document_search_tool(ctx),
        ]

        # Create main agent
        agent = FunctionAgent(
            llm=llm,
            system_prompt=RETRIEVAL_AGENT_PROMPT,
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

        sources: List[TextNode] = await ctx.get("sources")
        final_formatted_response = generate_citations(sources, full_raw_response)

        return StopEvent(result=final_formatted_response)
