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

    memory_str = ""

    last_user_msg_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].role == MessageRole.USER:
            last_user_msg_idx = i
            break

    for i, message in enumerate(history):
        if message.additional_kwargs.get("display", True):
            if message.role == MessageRole.ASSISTANT:
                if message.additional_kwargs.get(
                    "tool_calls"
                ) is not None and isinstance(
                    message.additional_kwargs["tool_calls"], list
                ):
                    tool_calls = message_to_tool_selections(message)

                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call.tool_name
                            tool_args = tool_call.tool_kwargs or {}

                            memory_str += f"<{tool_name}>\n"

                            for key, value in tool_args.items():
                                memory_str += f"\t{key}: {value}\n"
                            memory_str += f"</{tool_name}>\n\n"
                        except Exception as e:
                            print(e)
                else:
                    memory_str += f"<assistant>\n"
                    memory_str += f"{message.content}\n"
                    memory_str += "</assistant>\n\n"
            elif message.role == MessageRole.TOOL:
                try:
                    tool_name = message.additional_kwargs["tool_call_name"]

                    memory_str += f"<{tool_name}_result>\n"
                    if i > last_user_msg_idx:
                        memory_str += f"{message.content}\n"
                    else:
                        # If the tool result is from a previous user message, we don't include it in the context
                        memory_str += f"Tool result omitted for brevity.\n"
                    memory_str += f"</{tool_name}_result>\n\n"
                except Exception as e:
                    print(e)

            elif message.role == MessageRole.USER:
                memory_str += f"<user>\n"
                memory_str += f"{message.content}\n"
                memory_str += "</user>\n\n"

    memory_str += "Now, write an answer to the user."

    final_history = [
        system_message,
        ChatMessage(
            role=MessageRole.USER,
            content=memory_str.strip(),
        ),
    ]

    return final_history
