from llama_index.core.llms import ChatMessage, MessageRole
from app.agent.prompts import ROUTER_AGENT_PROMPT
from app.agent.utils import filter_tool_results
from typing import List


def construct_router_context(history: List[ChatMessage]) -> List[ChatMessage]:
    history = filter_tool_results(history)

    # Filter out messages containing tool calls
    history = [
        msg for msg in history if msg.additional_kwargs.get("tool_calls", None) is None
    ]

    # Truncate to only include up to 3 turns
    if len(history) > 7:
        history = history[-7:]

    # Construct context string
    context_str = "Here is what has happened in the conversation so far (older messages may be truncated for brevity):"
    for msg in history:
        if msg.role == MessageRole.USER:
            context_str += "<user>\n"
            context_str += f"{msg.content}\n"
            context_str += "</user>\n\n"
        elif msg.role == MessageRole.ASSISTANT:
            agent = msg.additional_kwargs.get("agent", "writer")
            context_str += (
                f"<{'writer_agent' if agent == 'writer' else 'chat_agent'}>\n"
            )
            context_str += f"{msg.content}\n"
            context_str += (
                f"</{'writer_agent' if agent == 'writer' else 'chat_agent'}>\n\n"
            )

    context_str += "Now, determine whether to route to the chat or retrieval agent."

    final_history = [
        ChatMessage(role="system", content=ROUTER_AGENT_PROMPT),
        ChatMessage(role="user", content=context_str),
    ]

    return final_history
