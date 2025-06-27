import re
from typing import List
from llama_index.core.llms import ChatMessage, MessageRole
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
    # Find citations in the response
    citations = re.findall("(\[([^\]]*)\])", text)
    for citation in citations:
        try:
            for source in sources:
                matching_citation_idx = list(
                    citation[1] in s.id for s in sources
                ).index(True)

            # Replace the citation with a link to the document
            text = text.replace(
                citation[0],
                f"[[{matching_citation_idx+1}]]({sources[matching_citation_idx].link})",
            )
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


def filter_writer_handoff(chat_history: List[ChatMessage]) -> List[ChatMessage]:
    """Filter to exclude writer handoff messages from the chat history."""

    new_history = []

    for msg in chat_history:
        if msg.content == "handoff_to_writer":
            continue

        if msg.role == MessageRole.ASSISTANT:
            tool_calls = msg.additional_kwargs.get("tool_calls", [])

            try:
                if any(
                    tool_call["function"]["name"] == "handoff_to_writer"
                    for tool_call in tool_calls
                ):
                    continue
            except Exception as e:
                print(f"Error checking tool calls: {e}")

        new_history.append(msg)

    return new_history
