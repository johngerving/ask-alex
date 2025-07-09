import functools
import json
from operator import is_
import os
from textwrap import dedent
from typing import Annotated, Any, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
from pydantic import BaseModel, Field, model_serializer
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
    AgentWorkflow,
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.schema import TextNode
from llama_index.core import set_global_handler
from llama_index.core.schema import MetadataMode
from llama_index.core.llms import LLM, TextBlock
from llama_index.core.memory import Memory

from app.agent.tools import (
    handoff_to_writer,
    search_documents,
    analyze_documents,
    query_knowledge_base,
    call_metadata_agent,
)
from app.agent.retrieval_agent import RetrievalAgent

from .prompts import (
    CHAT_AGENT_PROMPT,
    FINAL_ANSWER_PROMPT,
    RETRIEVAL_AGENT_PROMPT,
    ROUTER_AGENT_PROMPT,
)
from .utils import (
    Source,
    filter_tool_calls,
    filter_tool_results,
    generate_citations,
    remove_citations,
)


class WorkflowStartEvent(StartEvent):
    """Event to start the workflow."""

    message: ChatMessage
    memory: Optional[Memory] = None

    @model_serializer
    def serialize_start_event(self) -> dict:
        return {"message": self.message}


class ChatOrRetrievalRouteEvent(Event):
    pass


class ChatRouteEvent(Event):
    pass


class RetrievalRouteEvent(Event):
    pass


class WriteFinalAnswerEvent(Event):
    pass


class GenerateTitleEvent(Event):
    pass


class ChatOrRetrieval(BaseModel):
    """Data model for routing between chat and retrieval."""

    reasoning: str = Field(
        ..., description="The reasoning behind the routing decision."
    )
    route: Literal["chat", "retrieval"] = Field(
        ..., description="The route to take: either 'chat' or 'retrieval'."
    )


class StreamEvent(BaseModel):
    delta: str


class FinalAnswerEvent(BaseModel):
    content: str


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

        # Only use the last n messages
        # n = 3
        # history: List[ChatMessage] = ev.history[max(0, len(ev.history) - n) :]
        memory = ev.memory

        if not memory:
            memory = Memory.from_defaults()

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
        history = await memory.aget()

        history = filter_tool_results(history)
        history = filter_tool_calls(history)

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
        history = await memory.aget()

        history = filter_tool_results(history)

        agent = FunctionAgent(
            llm=self.small_llm,
            system_prompt=CHAT_AGENT_PROMPT,
        )

        handler = agent.run(chat_history=history)

        response = ""

        # Stream response from agent
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.response:
                ctx.write_event_to_stream(StreamEvent(delta=ev.delta))

        response = str(await handler)

        await memory.aput(ChatMessage(role="assistant", content=response))
        await ctx.set("memory", memory)

        ctx.write_event_to_stream(FinalAnswerEvent(content=response))

        return StopEvent()

    @step
    async def retrieve(
        self, ctx: Context, ev: RetrievalRouteEvent
    ) -> RetrievalRouteEvent | WriteFinalAnswerEvent:
        """Handle the retrieval route by searching the knowledge base for information."""
        memory: Memory = await ctx.get("memory")
        history = await memory.aget()

        agent = RetrievalAgent(
            llm=self.tool_llm,
            system_prompt=RETRIEVAL_AGENT_PROMPT,
            tools=[
                query_knowledge_base,
                call_metadata_agent,
                search_documents,
                analyze_documents,
                handoff_to_writer,
            ],
        )

        workflow = AgentWorkflow(
            agents=[agent],
        )

        # Use the current memory as the initial memory for the retrieval agent
        handler = workflow.run(
            chat_history=history,
            memory=memory,
        )

        # Pass events from retrieval agent along
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(ev.delta, end="", flush=True)
            if isinstance(ev, ToolCall):
                print(f"Tool call: {ev.tool_name}{ev.tool_kwargs}")
                if ev.tool_name != "handoff_to_writer":
                    ctx.write_event_to_stream(ev)

        result = str(await handler)

        # agent = FunctionAgent(
        #     llm=self.tool_llm,
        #     system_prompt=RETRIEVAL_AGENT_PROMPT,
        #     tools=[
        #         query_knowledge_base,
        #         call_metadata_agent,
        #         search_documents,
        #         analyze_document,
        #         handoff_to_writer,
        #     ],
        # )

        # workflow = AgentWorkflow(
        #     agents=[agent],
        # )

        # # Use the current memory as the initial memory for the retrieval agent
        # handler = workflow.run(
        #     chat_history=history,
        #     memory=memory,
        # )

        # # Pass events from retrieval agent along
        # async for ev in handler.stream_events():
        #     if isinstance(ev, AgentStream):
        #         print(ev.delta, end="", flush=True)
        #     if isinstance(ev, ToolCall):
        #         print(f"Tool call: {ev.tool_name}{ev.tool_kwargs}")
        #         if ev.tool_name != "handoff_to_writer":
        #             ctx.write_event_to_stream(ev)

        # result = str(await handler)

        print("Result:", result)

        agent_ctx = handler.ctx

        agent_retrieved_sources: List[Source] = await agent_ctx.get(
            "retrieved_sources", []
        )
        retrieved_sources: List[Source] = await ctx.get("retrieved_sources", [])
        retrieved_sources.extend(agent_retrieved_sources)
        await ctx.set("retrieved_sources", retrieved_sources)

        if result == "handoff_to_writer":
            await ctx.set("memory", memory)
            return WriteFinalAnswerEvent()
        else:
            msg = ChatMessage(
                role=MessageRole.USER,
                content="<agent_reminder>You must call at least one tool. You can either hand over control with the handoff_to_writer tool, or you can retrieve more information.</agent_reminder>",
                additional_kwargs={
                    "display": False,  # Do not display this message in the chat
                },
            )
            await memory.aput(msg)
            await ctx.set("memory", memory)
            return RetrievalRouteEvent()

    @step
    async def write_final_answer(
        self, ctx: Context, ev: WriteFinalAnswerEvent
    ) -> GenerateTitleEvent:
        """Write the final answer using the writer agent."""

        full_memory: Memory = await ctx.get("memory")
        full_history = await full_memory.aget_all()

        system_message = ChatMessage(
            role="system",
            content=FINAL_ANSWER_PROMPT,
        )

        chat_history: List[ChatMessage] = [system_message]

        memory_str = ""

        for message in full_history:
            if message.additional_kwargs.get("display", True):
                if message.role == MessageRole.ASSISTANT:
                    if message.additional_kwargs.get(
                        "tool_calls"
                    ) is not None and isinstance(
                        message.additional_kwargs["tool_calls"], list
                    ):
                        tool_calls = message.additional_kwargs.get("tool_calls", [])
                        try:
                            if not any(
                                tool_call["function"]["name"] == "handoff_to_writer"
                                for tool_call in tool_calls
                            ):
                                kwargs = {
                                    "tool_calls": message.additional_kwargs[
                                        "tool_calls"
                                    ]
                                }
                                memory_str += f"{kwargs}\n"
                        except Exception:
                            pass
                    else:
                        memory_str += f"<message role={message.role}>\n"
                        memory_str += f"{message.content}\n"
                        memory_str += "</message>\n"
                else:
                    if not "handoff_to_writer" in message.content:
                        memory_str += f"<message role={message.role}>\n"
                        memory_str += f"{message.content}\n"
                        memory_str += "</message>\n"
        chat_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content=memory_str,
            )
        )

        response_stream = await self.writer_llm.astream_chat(
            messages=chat_history,
        )

        # Parse citations in the response

        full_raw_response = ""
        buffer = ""  # Buffer for potentially incomplete citations

        async for response in response_stream:
            # Prepend any leftover buffer from the previous delta
            current_chunk = buffer + (response.delta or "")
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
                ctx.write_event_to_stream(StreamEvent(delta=delta_to_send))

        # Process any remaining buffer content
        if buffer:
            # self.logger.warning(f"Streaming ended with incomplete citation: {buffer}")
            full_raw_response += buffer

        if not response.message.content:
            raise Exception("No response from agent")

        await full_memory.aput(response.message)
        await ctx.set("memory", full_memory)

        retrieved_sources: List[Source] = await ctx.get("retrieved_sources", [])
        final_formatted_response = generate_citations(
            retrieved_sources, full_raw_response
        )

        ctx.write_event_to_stream(FinalAnswerEvent(content=final_formatted_response))

        return GenerateTitleEvent()

    @step
    async def generate_title(self, ctx: Context, ev: GenerateTitleEvent) -> StopEvent:
        """Generate a title for the chat."""
        title: str = await ctx.get("chat_title", None)

        if title:
            return StopEvent()

        memory: Memory = await ctx.get("memory")
        history = await memory.aget()

        # Use the small LLM to generate a title based on the chat history
        system_msg = ChatMessage(
            role="system",
            content="Given the following chat history, generate a concise and descriptive title for the chat. The title should be no more than 5 words long. Format your title in plain text; DO NOT use markdown. /no_think",
        )

        user_msg_content = ""

        for message in history:
            text_contents = "\n".join(
                block.text for block in message.blocks if isinstance(block, TextBlock)
            )

            user_msg_content += (
                f"<message role={message.role}>\n{text_contents}\n</message>\n"
            )

        user_msg = ChatMessage(role="user", content=user_msg_content)

        title = self.small_llm.chat(
            messages=[system_msg, user_msg],
        ).message.content

        await ctx.set("chat_title", title)

        return StopEvent()
