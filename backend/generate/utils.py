import re
from typing import List
from llama_index.core.schema import TextNode
from llama_index.core.workflow import Context


def generate_citations(sources: List[TextNode], text: str) -> str:
    """Generate citations for a text response based on the sources used and replace them in the text.

    Args:
        sources (list[TextNode]): The total sources to generate citations for.
        text (str): The text response to generate citations for.

    Returns:
        str: The text response with citations generated.
    """
    # Compile a list of sources used in the response
    sources_used: List[TextNode] = []
    for source in sources:
        if source.node_id[:8] in text and not any(
            [s.node_id == source.node_id for s in sources_used]
        ):
            # Add source to list if it's used in the response
            sources_used.append(source)

    # Find citations in the response
    citations = re.findall("\[[^\]]*\]", text)
    for citation in citations:
        try:
            matching_citation_idx = list(
                s.node_id[:8] in citation for s in sources_used
            ).index(True)

            # Get the download link for a given citation
            download_link = sources_used[matching_citation_idx].metadata.get(
                "download_link"
            )
            if download_link is None:
                raise ValueError("download_link not found")

            # Replace the citation with a link to the document
            text = text.replace(
                citation,
                f"[[{matching_citation_idx+1}]]({sources_used[matching_citation_idx].metadata.get('download_link')})",
            )
        except ValueError as e:
            text = text.replace(citation, "")

    return text


def remove_citations(text: str) -> str:
    """Remove citations from a text response so as not to confuse the LLM."""
    # Match patterns of the form [citation](link)
    citations_removed = re.sub("\[(.*?)\]\((.*?)\)", "", text)
    return citations_removed
