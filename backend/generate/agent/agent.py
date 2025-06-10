import functools
import json
from operator import is_
import os
from typing import Annotated, Any, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
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
from llama_index.llms.openrouter import OpenRouter
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
from llama_index.core.memory import Memory


from .prompts import (
    BASE_PROMPT,
    ROUTER_AGENT_PROMPT,
)
from .retrieval_agent import RetrievalAgent, RetrievalStopEvent, StreamEvent
from .utils import remove_citations


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


class Agent(Workflow):
    """The main workflow for Ask Alex."""

    def __init__(self, logger: Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

        self.writer_llm = OpenRouter(
            model="deepseek/deepseek-chat-v3-0324",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            context_window=41000,
            max_tokens=4000,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        self.tool_llm = OpenRouter(
            model="qwen/qwen3-235b-a22b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            context_window=41000,
            max_tokens=4000,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        self.small_llm = OpenRouter(
            model="qwen/qwen3-32b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            context_window=41000,
            max_tokens=4000,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @step
    async def setup(
        self, ctx: Context, ev: WorkflowStartEvent
    ) -> ChatOrRetrievalRouteEvent:
        self.logger.info("Running setup step")

        await ctx.set("logger", self.logger)

        # Only use the last n messages
        # n = 3
        # history: List[ChatMessage] = ev.history[max(0, len(ev.history) - n) :]
        history: List[ChatMessage] = ev.history
        for m in history:
            m.content = remove_citations(m.content)
        history: List[ChatMessage] = ev.history

        memory: Memory = Memory.from_defaults()

        for m in history:
            await memory.aput(m)

        await memory.aput(ev.message)

        await ctx.set("memory", memory)

        return ChatOrRetrievalRouteEvent()

    @step
    async def route_chat_or_retrieval(
        self, ctx: Context, ev: ChatOrRetrievalRouteEvent
    ) -> ChatRouteEvent | RetrievalRouteEvent:
        """Route the message to either chat or retrieval based on the message and history."""
        self.logger.info("Routing chat or retrieval")

        sllm = self.small_llm.as_structured_llm(output_cls=ChatOrRetrieval)

        memory: Memory = await ctx.get("memory")
        history = memory.get()

        # Call the LLM to determine the route
        json_obj: Dict[str, str] = sllm.chat(
            messages=[
                ChatMessage(role="system", content=ROUTER_AGENT_PROMPT),
                *history,
            ],
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

        memory: Memory = await ctx.get("memory")
        history = memory.get()

        agent = FunctionAgent(
            llm=self.small_llm,
            system_prompt=BASE_PROMPT,
        )

        handler = agent.run(chat_history=history)

        response = ""

        # Stream response from agent
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.response:
                ctx.write_event_to_stream(StreamEvent(delta=ev.delta))

        response = await handler

        memory.put(ChatMessage(role="assistant", content=response))
        await ctx.set("memory", memory)

        return StopEvent(result=response)

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalRouteEvent) -> StopEvent:
        """Handle the retrieval route by searching the knowledge base for information."""
        w = RetrievalAgent(
            writer_llm=self.writer_llm,
            tool_llm=self.tool_llm,
            timeout=self._timeout,
            verbose=self._verbose,
        )

        memory = await ctx.get("memory")

        # Use the current memory as the initial memory for the retrieval agent
        handler = w.run(memory=memory)

        # Pass events from retrieval agent along
        async for ev in handler.stream_events():
            if isinstance(ev, StreamEvent):
                ctx.write_event_to_stream(ev)

        result: RetrievalStopEvent = await handler

        # Update the context memory with the final response from the retrieval agent
        await ctx.set("memory", result.memory)

        return StopEvent(result=result.response)
