import json
from textwrap import dedent
from unittest.mock import Base
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, Dict, List, Literal
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field


class ReflectionInputEvent(Event):
    input: List[ChatMessage]


class CallToolRouteEvent(Event):
    pass


class FinalAnswerRouteEvent(Event):
    pass


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: List[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class Reflection(BaseModel):
    inner_thoughts: str
    route: Literal["final_answer", "call_tools"]


class ReflectionAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        small_llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or OpenAI()
        self.small_llm = llm or OpenAI()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> ReflectionInputEvent:
        # Clear sources
        await ctx.set("sources", [])

        # Check if memory is set up
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # Get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # Get chat history
        chat_history = memory.get()

        # Update context
        await ctx.set("memory", memory)

        return ReflectionInputEvent(input=chat_history)

    @step
    async def handle_reflection(
        self, ctx: Context, ev: ReflectionInputEvent
    ) -> ReflectionInputEvent | CallToolRouteEvent | FinalAnswerRouteEvent:
        chat_history = ev.input

        tool_schemas = json.dumps(
            [tool.metadata.description for tool in self.tools], indent=2
        )

        system_message = ChatMessage(
            role="system",
            content=dedent(
                """You are a helpful assistant. You have access to the following tools:
                %s 

                Given the following conversation, you have the following tasks:
                1. Think step-by-step to reason about the conversation. You should plan any actions that you will take, including the tools you will use or how you will respond to the user. You should reflect on the output of the tools you call, determining whether you have enough information to make a final answer.
                2. Decide whether to call tools or output a final answer.

                You do not need to call the tools. You only need to reason about the conversation.

                Output your answer in the following format:
                {
                    "inner_thoughts": <your thoughts>,
                    "route": <call_tools or final_answer>
                }
                """
                % tool_schemas
            ),
        )

        chat_history.insert(0, system_message)

        reflectionWorkflow = ReflectionValidation(
            llm=self.small_llm, timeout=self._timeout
        )
        result: ReflectionStopEvent = await reflectionWorkflow.run(
            chat_history=chat_history
        )

        # Save the reasoning content to the memory
        memory: ChatMemoryBuffer = await ctx.get("memory")
        memory.put(
            ChatMessage(role="assistant", content=result.reflection.inner_thoughts)
        )
        await ctx.set("memory", memory)

        if result.reflection.route == "call_tools":
            return CallToolRouteEvent()
        else:
            return FinalAnswerRouteEvent()

    @step
    async def handle_tools_route(
        self, ctx: Context, ev: CallToolRouteEvent
    ) -> ReflectionInputEvent | ToolCallEvent:
        memory: ChatMemoryBuffer = await ctx.get("memory")
        chat_history = memory.get()

        system_message = ChatMessage(
            role="system",
            content=dedent(
                """You are a helpful assistant. Given the following conversation, call the appropriate tools based on your previous thoughts."""
            ),
        )

        chat_history.insert(0, system_message)

        # Stream response
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )

        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # Save the final response
        memory: ChatMemoryBuffer = await ctx.get("memory")
        memory.put(response.message)
        await ctx.set("memory", memory)

        # Get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        print(f"Tool calls: {tool_calls}")

        if not tool_calls:
            return ReflectionInputEvent(input=memory.get())
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> ReflectionInputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources: List[ToolOutput] = await ctx.get("sources", default=[])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # Update memory
        memory: ChatMemoryBuffer = await ctx.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.set("sources", sources)
        await ctx.set("memory", memory)

        chat_history = memory.get()
        return ReflectionInputEvent(input=chat_history)

    @step
    async def handle_final_answer(
        self, ctx: Context, ev: FinalAnswerRouteEvent
    ) -> StopEvent:
        # Get memory
        memory: ChatMemoryBuffer = await ctx.get("memory")
        chat_history = memory.get()

        sources: List[ToolOutput] = await ctx.get("sources", default=[])

        response_stream = await self.llm.astream_chat(
            messages=chat_history,
        )

        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # Save the final response
        memory.put(response.message)
        await ctx.set("memory", memory)

        return StopEvent(result={"response": response, "sources": [*sources]})


class ReflectionValidatedEvent(Event):
    reflection: Reflection


class ReflectionStartEvent(StartEvent):
    chat_history: List[ChatMessage]


class ReflectionStopEvent(StopEvent):
    reflection: Reflection


class ReflectionValidation(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.llm = llm or OpenAI()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: ReflectionStartEvent
    ) -> ReflectionInputEvent:
        await ctx.set("num_rounds", 0)

        return ReflectionInputEvent(input=ev.chat_history)

    @step
    async def handle_reflection(
        self, ctx: Context, ev: ReflectionInputEvent
    ) -> ReflectionValidatedEvent | ReflectionInputEvent:
        num_rounds = await ctx.get("num_rounds")
        if num_rounds >= 4:
            return ReflectionValidatedEvent(
                reflection=Reflection(inner_thoughts="", route="final_answer")
            )
        num_rounds += 1
        await ctx.set("num_rounds", num_rounds)

        chat_history = ev.input

        llm = self.llm.as_structured_llm(output_cls=Reflection)

        print(f"CHAT HISTORY: {chat_history}")

        # Stream the response
        response_stream = await llm.astream_chat(
            messages=chat_history,
        )

        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        print(response.raw)

        try:
            response_obj = Reflection(**response.raw.dict())
        except Exception as e:
            print(f"Error parsing response: {e}")
            error_msg = f"Error parsing response: {e}"
            chat_history.append(response.message)
            chat_history.append(ChatMessage(role="assistant", content=error_msg))
            return ReflectionInputEvent(input=chat_history)

        return ReflectionValidatedEvent(reflection=response_obj)

    @step
    async def return_reflection(
        self, ctx: Context, ev: ReflectionValidatedEvent
    ) -> ReflectionStopEvent:
        return ReflectionStopEvent(reflection=ev.reflection)
