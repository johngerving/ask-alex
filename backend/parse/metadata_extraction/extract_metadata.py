import json
import os
from typing import Dict, List
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

from docling_core.types.doc.document import DoclingDocument

from llama_index.core import Document as LIDocument
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.text_splitter import TokenTextSplitter

load_dotenv()


def extract_metadata():
    """Extracts metadata from each document in the database and modifies the document to include the metadata."""
    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    # Set up two connections: one for reading and one for writing
    with psycopg.connect(conn_str) as read_conn, psycopg.connect(
        conn_str
    ) as write_conn:
        # Count the number of documents that need to be processed
        with read_conn.cursor("count_documents") as cur:
            batch_size = 32
            num_rows = cur.execute(
                "SELECT COUNT(*) FILTER(WHERE document->'metadata'->'summary' IS NULL) FROM documents"
            ).fetchone()[0]
            total_batches = (
                num_rows + batch_size - 1
            ) // batch_size  # Calculate total batches
            curr_batch = 1

        with read_conn.cursor(
            "fetch_documents"
        ) as fetch_cur, write_conn.cursor() as update_cur:
            # Fetch documents that do not have a summary in their metadata
            fetch_cur.execute(
                "SELECT link, document FROM documents WHERE document->'metadata'->'summary' IS NULL"
            )

            while True:
                # Fetch a batch of documents
                records = fetch_cur.fetchmany(size=batch_size)

                if not records:
                    break

                print(f"Processing batch {curr_batch}/{total_batches}")

                links: List[str] = [record[0] for record in records]
                if not isinstance(links, list) or not all(
                    isinstance(link, str) for link in links
                ):
                    raise TypeError(
                        f"Expected links to be a list of strings, got {type(links)}"
                    )

                batch = [record[1] for record in records]

                summaries = summarize_batch(batch)

                # Update the database with the summaries
                update_cur.executemany(
                    "UPDATE documents SET document = jsonb_set(document, %s, %s, true) WHERE link = %s",
                    [
                        (["metadata", "summary"], Jsonb(summary), link)
                        for link, summary in zip(links, summaries)
                    ],
                )
                # Commit the changes to the database
                write_conn.commit()

                curr_batch += 1


def summarize_document(document_obj: Dict[any, any]) -> str:
    """Summarizes a single document.

    Args:
        document_obj: A dictionary representing an exported LlamaIndex document, whose text is a JSON string representation of a Docling document.
    """

    # Create a LlamaIndex document and convert it to a Docling document to get the content
    li_doc = LIDocument.from_dict(document_obj)
    dl_doc = DoclingDocument.model_validate(json.loads(li_doc.text))

    content = dl_doc.export_to_markdown()

    max_tokens = 31000

    # Check if the document has already been summarized
    cached_summary = li_doc.metadata.get("summary")
    if cached_summary is not None:
        return cached_summary

    splitter = TokenTextSplitter(
        chunk_size=max_tokens,
    )

    llm = OpenAILike(
        model="meta-llama/Llama-3.2-3B-Instruct",
        api_key=os.getenv("SUMMARIZER_LLM_API_KEY"),
        api_base=os.getenv("SUMMARIZER_LLM_API_BASE"),
        context_window=32000,
        max_tokens=1000,
    )

    query_str = "Summarize this text in a few sentences, capturing the main points and core meaning of the text. Be concise and clear in your summary."

    # Split the content into manageable chunks
    chunks = splitter.split_text(content)

    # Summarize the chunks recursively
    summarizer = TreeSummarize(
        llm=llm,
    )

    try:
        # Summarize the document content
        summary = summarizer.get_response(query_str, chunks)
    except Exception as e:
        print(f"Error summarizing document: {e}")
        print("Document content:", chunks)
        raise

    return summary


def summarize_batch(batch: List[Dict[any, any]]) -> List[str]:
    """Summarizes a batch of documents and updates the database to include the summary."""

    summaries = [summarize_document(doc) for doc in batch]

    return summaries
