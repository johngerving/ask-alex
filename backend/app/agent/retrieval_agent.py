from typing import List, Sequence

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.agent.workflow import FunctionAgent
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Sequence, Optional, Union, cast

from pydantic._internal._model_construction import ModelMetaclass
from llama_index.core.agent.workflow.prompts import DEFAULT_STATE_PROMPT
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentInput,
    AgentSetup,
    AgentWorkflowStartEvent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)
from llama_index.core.llms import ChatMessage, LLM, TextBlock
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType, PromptDictType
from llama_index.core.tools import (
    BaseTool,
    AsyncBaseTool,
    FunctionTool,
    ToolOutput,
    ToolSelection,
    adapt_to_async_tool,
)
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.workflow.checkpointer import CheckpointCallback
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StopEvent
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.workflow.workflow import Workflow, WorkflowMeta

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_AGENT_NAME = "Agent"
DEFAULT_AGENT_DESCRIPTION = "An agent that can perform a task"
WORKFLOW_KWARGS = (
    "timeout",
    "verbose",
    "service_manager",
    "resource_manager",
    "num_concurrent_runs",
)


class RetrievalAgent(FunctionAgent):

    tool_required: bool = Field(
        default=True,
        description="Whether the agent requires a tool call to proceed. If True, the agent will not respond until a tool call is made.",
    )

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )

        response = await self.llm.astream_chat_with_tools(  # type: ignore
            tools=tools,
            chat_history=current_llm_input,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
        )
        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=last_chat_response.delta or "",
                    response=last_chat_response.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=raw,
                    current_agent_name=self.name,
                )
            )

        tool_calls: ToolSelection = self.llm.get_tool_calls_from_response(  # type: ignore
            last_chat_response, error_on_no_tool_call=False
        )

        if not any(
            [tool_call.tool_name == "handoff_to_writer" for tool_call in tool_calls]
        ):
            # only add to scratchpad if we didn't select the handoff tool
            scratchpad.append(last_chat_response.message)
        else:
            scratchpad.append(ChatMessage(role=MessageRole.ASSISTANT, content=""))

        await ctx.store.set(self.scratchpad_key, scratchpad)

        last_chat_response.message.content = ""
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        return AgentOutput(
            response=last_chat_response.message,
            tool_calls=tool_calls or [],
            raw=raw,
            current_agent_name=self.name,
        )

    async def parse_agent_output(
        self, ctx: Context, ev: AgentOutput
    ) -> Union[StopEvent, ToolCall, AgentInput]:
        max_iterations = await ctx.get("max_iterations", default=DEFAULT_MAX_ITERATIONS)
        num_iterations = await ctx.get("num_iterations", default=0)
        num_iterations += 1
        await ctx.set("num_iterations", num_iterations)

        if num_iterations >= max_iterations:
            raise WorkflowRuntimeError(
                f"Max iterations of {max_iterations} reached! Either something went wrong, or you can "
                "increase the max iterations with `.run(.., max_iterations=...)`"
            )

        if any(
            tool_call.tool_name == "handoff_to_writer" for tool_call in ev.tool_calls
        ):
            memory: BaseMemory = await ctx.store.get("memory")
            output = await self.finalize(ctx, ev, memory)

            cur_tool_calls: List[ToolCallResult] = await ctx.store.get(
                "current_tool_calls", default=[]
            )
            output.tool_calls = cur_tool_calls  # type: ignore
            await ctx.store.set("current_tool_calls", [])

            return StopEvent(result=output)

        await ctx.store.set("num_tool_calls", len(ev.tool_calls))

        for tool_call in ev.tool_calls:
            ctx.send_event(
                ToolCall(
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_id=tool_call.tool_id,
                )
            )

        user_msg_str = await ctx.store.get("user_msg_str")
        input_messages = await memory.aget(input=user_msg_str)

        return AgentInput(input=input_messages, current_agent_name=self.name)

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        for tool_call_result in results:
            if not tool_call_result.tool_name == "handoff_to_writer":
                scratchpad.append(
                    ChatMessage(
                        role="tool",
                        blocks=tool_call_result.tool_output.blocks,
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )

                if (
                    tool_call_result.return_direct
                    and tool_call_result.tool_name != "handoff"
                ):
                    scratchpad.append(
                        ChatMessage(
                            role="assistant",
                            content=str(tool_call_result.tool_output.content),
                            additional_kwargs={
                                "tool_call_id": tool_call_result.tool_id
                            },
                        )
                    )
                    break

        await ctx.store.set(self.scratchpad_key, scratchpad)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """
        Finalize the function calling agent.

        Adds all in-progress messages to memory.
        """
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        if scratchpad:
            await memory.aput_messages(scratchpad)

        # reset scratchpad
        await ctx.store.set(self.scratchpad_key, [])

        return output
