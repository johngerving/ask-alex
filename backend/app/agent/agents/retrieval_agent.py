from typing import List, Sequence

from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from llama_index.core.agent.workflow.base_agent import (
    BaseWorkflowAgent,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_DESCRIPTION,
    DEFAULT_MAX_ITERATIONS,
)
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.agent.workflow.prompts import (
    DEFAULT_HANDOFF_PROMPT,
    DEFAULT_HANDOFF_OUTPUT_PROMPT,
    DEFAULT_STATE_PROMPT,
)
from llama_index.core.agent.workflow.react_agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentSetup,
    AgentOutput,
    AgentWorkflowStartEvent,
)
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType, PromptDictType
from llama_index.core.tools import (
    AsyncBaseTool,
    ToolOutput,
    ToolSelection,
)
from llama_index.core.workflow import (
    Context,
    StopEvent,
    step,
    WorkflowRuntimeError,
)
from llama_index.core.agent.workflow import AgentWorkflow
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
        reasoning = ""
        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            try:
                reasoning_delta = last_chat_response.raw.choices[0].delta.reasoning
                if reasoning_delta is not None:
                    reasoning += reasoning_delta
            except Exception as e:
                pass

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

        print("Reasoning:", reasoning)

        last_chat_response.message.content = (
            (f"<think>\n{reasoning}</think>\n") if reasoning else ""
        )

        if tool_calls and not any(
            [tool_call.tool_name == "handoff_to_writer" for tool_call in tool_calls]
        ):
            # only add to scratchpad if we didn't select the handoff tool
            scratchpad.append(last_chat_response.message)
        else:
            scratchpad.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    additional_kwargs={
                        "display": False,
                    },
                )
            )

        await ctx.store.set(self.scratchpad_key, scratchpad)

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

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        for tool_call_result in results:
            scratchpad.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={
                        "tool_call_id": tool_call_result.tool_id,
                        "tool_call_name": tool_call_result.tool_name,
                    },
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
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break

        scratchpad.append(
            ChatMessage(
                role=MessageRole.USER,
                content="<reminder>Do not answer the user directly. If you are done calling tools, use the handoff_to_writer tool to hand off the task to a writer.</reminder>",
                additional_kwargs={"display": False},
            )
        )

        await ctx.store.set(self.scratchpad_key, scratchpad)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """
        Finalize the function calling agent.

        Adds all in-progress messages to memory.
        """

        # Reset the count for number of analyze_documents calls
        await ctx.store.set("num_analyses", 0)

        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        if scratchpad:
            # Filter out handoff_to_writer messages when adding to memory
            scratchpad = [
                msg for msg in scratchpad if msg.additional_kwargs.get("display", True)
            ]
            for msg in scratchpad:
                if msg.role == MessageRole.ASSISTANT:
                    msg.content = ""

            await memory.aput_messages(scratchpad)

        # reset scratchpad
        await ctx.store.set(self.scratchpad_key, [])

        return output

    @step
    async def init_run(self, ctx: Context, ev: AgentWorkflowStartEvent) -> AgentInput:
        """Sets up the workflow and validates inputs."""
        await self._init_context(ctx, ev)

        user_msg: Optional[Union[str, ChatMessage]] = ev.get("user_msg")
        chat_history: Optional[List[ChatMessage]] = ev.get("chat_history", [])

        # Convert string user_msg to ChatMessage
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role="user", content=user_msg)

        # Add messages to memory
        memory: BaseMemory = await ctx.store.get("memory")

        # First set chat history if it exists
        if chat_history:
            await memory.aset(chat_history)

        # Then add user message if it exists
        if user_msg:
            await memory.aput(user_msg)
            content_str = "\n".join(
                [
                    block.text
                    for block in user_msg.blocks
                    if isinstance(block, TextBlock)
                ]
            )
            await ctx.store.set("user_msg_str", content_str)
        elif chat_history:
            # If no user message, use the last message from chat history as user_msg_str
            content_str = "\n".join(
                [
                    block.text
                    for block in chat_history[-1].blocks
                    if isinstance(block, TextBlock)
                ]
            )
            await ctx.store.set("user_msg_str", content_str)
        else:
            raise ValueError("Must provide either user_msg or chat_history")

        # Get all messages from memory
        input_messages = await memory.aget()

        # send to the current agent
        return AgentInput(input=input_messages, current_agent_name=self.name)

    @step
    async def setup_agent(self, ctx: Context, ev: AgentInput) -> AgentSetup:
        """Main agent handling logic."""
        llm_input = [*ev.input]

        if self.system_prompt:
            llm_input = [
                ChatMessage(role="system", content=self.system_prompt),
                *llm_input,
            ]

        state = await ctx.store.get("state", default=None)
        formatted_input_with_state = await ctx.store.get(
            "formatted_input_with_state", default=False
        )
        if state and not formatted_input_with_state:
            # update last message with current state
            for block in llm_input[-1].blocks[::-1]:
                if isinstance(block, TextBlock):
                    block.text = self.state_prompt.format(state=state, msg=block.text)
                    break
            await ctx.store.set("formatted_input_with_state", True)

        return AgentSetup(
            input=llm_input,
            current_agent_name=ev.current_agent_name,
        )

    @step
    async def run_agent_step(self, ctx: Context, ev: AgentSetup) -> AgentOutput:
        """Run the agent."""
        memory: BaseMemory = await ctx.store.get("memory")
        user_msg_str = await ctx.store.get("user_msg_str")
        tools = await self.get_tools(user_msg_str or "")

        agent_output = await self.take_step(
            ctx,
            ev.input,
            tools,
            memory,
        )

        ctx.write_event_to_stream(agent_output)
        return agent_output

    @step
    async def parse_agent_output(
        self, ctx: Context, ev: AgentOutput
    ) -> Union[StopEvent, AgentSetup, ToolCall, None]:
        max_iterations = await ctx.store.get(
            "max_iterations", default=DEFAULT_MAX_ITERATIONS
        )
        num_iterations = await ctx.store.get("num_iterations", default=0)
        num_iterations += 1
        await ctx.store.set("num_iterations", num_iterations)

        if num_iterations >= max_iterations:
            raise WorkflowRuntimeError(
                f"Max iterations of {max_iterations} reached! Either something went wrong, or you can "
                "increase the max iterations with `.run(.., max_iterations=...)`"
            )

        if any(
            [tool_call.tool_name == "handoff_to_writer" for tool_call in ev.tool_calls]
        ):
            memory: BaseMemory = await ctx.store.get("memory")
            output = await self.finalize(ctx, ev, memory)

            return StopEvent(result=output)

        if not ev.tool_calls:
            scratchpad: List[ChatMessage] = await ctx.store.get(
                self.scratchpad_key, default=[]
            )
            scratchpad.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content="<agent_reminder>You must call at least one tool. Call handoff_to_writer to hand off the task to a writer.</agent_reminder>",
                    additional_kwargs={"display": False},
                )
            )
            await ctx.store.set(self.scratchpad_key, scratchpad)

            return AgentSetup(input=[], current_agent_name=self.name)

        await ctx.store.set("num_tool_calls", len(ev.tool_calls))

        for tool_call in ev.tool_calls:
            ctx.send_event(
                ToolCall(
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_id=tool_call.tool_id,
                )
            )

        return None

    @step
    async def call_tool(self, ctx: Context, ev: ToolCall) -> ToolCallResult:
        """Calls the tool and handles the result."""
        ctx.write_event_to_stream(
            ToolCall(
                tool_name=ev.tool_name,
                tool_kwargs=ev.tool_kwargs,
                tool_id=ev.tool_id,
            )
        )

        tools = await self.get_tools(ev.tool_name)
        tools_by_name = {tool.metadata.name: tool for tool in tools}
        if ev.tool_name not in tools_by_name:
            tool = None
            result = ToolOutput(
                content=f"Tool {ev.tool_name} not found. Please select a tool that is available.",
                tool_name=ev.tool_name,
                raw_input=ev.tool_kwargs,
                raw_output=None,
                is_error=True,
            )
        else:
            tool = tools_by_name[ev.tool_name]
            result = await self._call_tool(ctx, tool, ev.tool_kwargs)

        result_ev = ToolCallResult(
            tool_name=ev.tool_name,
            tool_kwargs=ev.tool_kwargs,
            tool_id=ev.tool_id,
            tool_output=result,
            return_direct=tool.metadata.return_direct if tool else False,
        )

        ctx.write_event_to_stream(result_ev)
        return result_ev

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResult
    ) -> Union[AgentInput, StopEvent, None]:
        """Aggregate tool results and return the next agent input."""
        num_tool_calls = await ctx.store.get("num_tool_calls", default=0)
        if num_tool_calls == 0:
            raise ValueError("No tool calls found, cannot aggregate results.")

        tool_call_results: list[ToolCallResult] = ctx.collect_events(  # type: ignore
            ev, expected=[ToolCallResult] * num_tool_calls
        )
        if not tool_call_results:
            return None

        memory: BaseMemory = await ctx.store.get("memory")

        # track tool calls made during a .run() call
        cur_tool_calls: List[ToolCallResult] = await ctx.store.get(
            "current_tool_calls", default=[]
        )
        cur_tool_calls.extend(tool_call_results)
        await ctx.store.set("current_tool_calls", cur_tool_calls)

        await self.handle_tool_call_results(ctx, tool_call_results, memory)

        if any(
            tool_call_result.return_direct for tool_call_result in tool_call_results
        ):
            # if any tool calls return directly, take the first one
            return_direct_tool = next(
                tool_call_result
                for tool_call_result in tool_call_results
                if tool_call_result.return_direct
            )

            # always finalize the agent, even if we're just handing off
            result = AgentOutput(
                response=ChatMessage(
                    role="assistant",
                    content=return_direct_tool.tool_output.content or "",
                ),
                tool_calls=[
                    ToolSelection(
                        tool_id=t.tool_id,
                        tool_name=t.tool_name,
                        tool_kwargs=t.tool_kwargs,
                    )
                    for t in cur_tool_calls
                ],
                raw=return_direct_tool.tool_output.raw_output,
                current_agent_name=self.name,
            )
            result = await self.finalize(ctx, result, memory)

        user_msg_str = await ctx.store.get("user_msg_str")
        input_messages = await memory.aget(input=user_msg_str)

        return AgentInput(input=input_messages, current_agent_name=self.name)
