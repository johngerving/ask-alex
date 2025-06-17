import re
from typing import List
from llama_index.core.schema import TextNode
from llama_index.core.workflow import Context
from llama_index.core.schema import Document


def generate_citations(
    source_nodes: List[TextNode], source_docs: List[Document], text: str
) -> str:
    """Generate citations for a text response based on the sources used and replace them in the text.

    Args:
        source_nodes (List[TextNode]): The total TextNodes cited in the response.
        source_docs (List[Document]): The total Documents cited in the response.
        text (str): The text response to generate citations for.

    Returns:
        str: The text response with citations generated.
    """
    # Compile a list of sources used in the response
    sources_used: List[TextNode | Document] = []

    for source in source_docs:
        if source.doc_id[:8] in text and not any(
            [s.doc_id == source.doc_id for s in sources_used]
        ):
            # Add source to list if it's used in the response
            sources_used.append(source)

    for source in source_nodes:
        if source.node_id[:8] in text and not any(
            [s.node_id == source.node_id for s in sources_used]
        ):
            # Add source to list if it's used in the response
            sources_used.append(source)

    # Find citations in the response
    citations = re.findall("\[[^\]]*\]", text)
    for citation in citations:
        try:
            for source in sources_used:
                if hasattr(source, "doc_id"):
                    matching_citation_idx = list(
                        s.doc_id[:8] in citation for s in sources_used
                    ).index(True)
                elif hasattr(source, "node_id"):
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
