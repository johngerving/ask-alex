from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
from llama_index.llms.openrouter import OpenRouter
from typing import List
from app.agent.prompts import FINAL_ANSWER_PROMPT
from app.agent.utils import message_to_tool_selections

# Initialize LLM to parse tool calls
llm = OpenRouter(
    model="qwen/qwen3-30b-a3b",
    is_function_calling_model=True,
)


def construct_writer_context(history: List[ChatMessage]) -> str:
    """Construct the context for the writer agent from the chat history."""

    system_message = ChatMessage(
        role="system",
        content=FINAL_ANSWER_PROMPT,
    )

    # Don't include tool call results from more than two turns ago
    # This is to avoid overwhelming the writer with too much context
    # and to keep the context relevant to the current conversation.
    tool_omit_cutoff_idx = -1
    user_msg_count = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i].role == MessageRole.USER:
            user_msg_count += 1
            if user_msg_count >= 2:
                tool_omit_cutoff_idx = i
                break

    final_history = [system_message]

    for i, message in enumerate(history):
        if message.additional_kwargs.get("display", True):
            if message.role == MessageRole.TOOL and not i > tool_omit_cutoff_idx:
                # If the tool result is from a previous user message, we don't include it in the context
                message.content = "Tool result omitted for brevity.\n"

            final_history.append(message)

    return final_history
