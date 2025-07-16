import re
from typing import List
from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.agent.workflow import AgentStream, ToolCall
from llama_index.llms.openrouter import OpenRouter
from openai.types.chat import ChatCompletionMessageToolCall
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Source(BaseModel):
    """Model for a citation source."""

    id: str
    link: str


def generate_citations(sources: List[Source], text: str) -> str:
    """Generate citations for a text response based on the sources used and replace them in the text.

    Args:
        sources (List[Source]): A list of sources used in the response
        text (str): The text response to generate citations for.

    Returns:
        str: The text response with citations generated.
    """
    citation_num = 1

    # Find citations in the response
    citations = re.findall("(\[([^\]]*)\])", text)
    for citation in citations:
        try:
            matching_citation_idx = list(citation[1] in s.id for s in sources).index(
                True
            )

            num_occurrences = text.count(citation[0])
            if num_occurrences > 0:
                # Replace the citation with a link to the document
                text = text.replace(
                    citation[0],
                    f"[[{citation_num}]]({sources[matching_citation_idx].link})",
                )
                citation_num += 1
        except Exception as e:
            logger.error(f"Error generating citation for {citation[0]}: {e}")
            text = text.replace(citation[0], "")

    return text


def remove_citations(text: str) -> str:
    """Remove citations from a text response so as not to confuse the LLM."""
    # Match patterns of the form [citation](link)
    citations_removed = re.sub("\[(.*?)\]\((.*?)\)", "", text)
    return citations_removed


def filter_tool_results(chat_history: List[ChatMessage]) -> List[ChatMessage]:
    """Filter to exclude tool results from the chat history."""
    return [msg for msg in chat_history if msg.role != MessageRole.TOOL]


def filter_tool_calls(chat_history: List[ChatMessage]) -> List[ChatMessage]:
    """Filter to exclude tool calls from the chat history."""
    new_history = chat_history

    for msg in new_history:
        msg.additional_kwargs.pop("tool_calls", None)

    return new_history


def message_to_tool_selections(msg: ChatMessage) -> List[ToolSelection]:
    llm = OpenRouter(
        model="qwen/qwen3-30b-a3b",
        is_function_calling_model=True,
    )
    tool_calls = msg.additional_kwargs.get("tool_calls", [])

    # Convert tool call dicts to ChatCompletionMessageToolCall objects
    # This is a workaround for a bug in LlamaIndex
    tool_calls = [
        ChatCompletionMessageToolCall(**tool_call) for tool_call in tool_calls
    ]
    msg.additional_kwargs["tool_calls"] = tool_calls

    chat_response = ChatResponse(message=msg)

    tool_calls = llm.get_tool_calls_from_response(
        chat_response, error_on_no_tool_call=False
    )

    return tool_calls
