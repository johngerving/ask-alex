import re
from typing import List
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.workflow import Context
from llama_index.core.schema import Document
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
