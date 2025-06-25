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
    # Compile a list of sources used in the response
    sources_used: List[Source] = []

    for source in sources:
        if source.id in text and not any([s.id == source.id for s in sources_used]):
            # Add source to list if it's used in the response
            sources_used.append(source)

    # Find citations in the response
    citations = re.findall("\[[^\]]*\]", text)
    for citation in citations:
        try:
            for source in sources_used:
                matching_citation_idx = list(
                    s.id in citation for s in sources_used
                ).index(True)

            # Replace the citation with a link to the document
            text = text.replace(
                citation,
                f"[[{matching_citation_idx+1}]]({sources_used[matching_citation_idx].link})",
            )
        except Exception as e:
            logger.error(f"Error generating citation for {citation}: {e}")
            text = text.replace(citation, "")

    return text


def remove_citations(text: str) -> str:
    """Remove citations from a text response so as not to confuse the LLM."""
    # Match patterns of the form [citation](link)
    citations_removed = re.sub("\[(.*?)\]\((.*?)\)", "", text)
    return citations_removed


def filter_tool_results(chat_history: List[ChatMessage]):
    """Filter to exclude tool results from the chat history."""
    return [msg for msg in chat_history if msg.role != MessageRole.TOOL]


def filter_tool_calls(chat_history: List[ChatMessage]):
    """Filter to exclude tool calls from the chat history."""
    new_history = chat_history

    for msg in new_history:
        msg.additional_kwargs.pop("tool_calls", None)

    return new_history
