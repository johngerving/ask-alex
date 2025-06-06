import json
from logging import Logger
import os
import re
from textwrap import dedent
from typing import Annotated, List
from urllib.parse import urlparse

from llama_index.core.workflow import (
    Context,
)
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.core.schema import MetadataMode
from llama_index.core.llms import LLM
from llama_index.core.schema import Document
import psycopg

# SELECT DISTINCT elem FROM documents CROSS JOIN LATERAL jsonb_array_elements_text(document->'metadata'->'discipline') AS t(elem) WHERE jsonb_typeof(document->'metadata'->'discipline') = 'array';


async def make_document_search_tool(ctx: Context) -> FunctionTool:
    pg_conn_str = os.getenv("PG_CONN_STR")
    if not pg_conn_str:
        raise ValueError("PG_CONN_STR environment variable not set")

    async def search_documents(
        search_terms: Annotated[
            List[str], "A list of key words or terms to search for"
        ],
        page: Annotated[int, "The page number for pagination, starting from 1"] = 1,
    ) -> str:
        """Search the knowledge base for relevant documents. Best used for information about documents themselves and summaries of specific documents. Do NOT use if answer can be found in a specific chunk of a given document. Use the retrieve_chunks tool instead for that purpose.

        Example user messages to use this tool for:
        - "What documents talk about ...?"
        - "Summarize documents about ..."
        """

        query = " ".join(search_terms)
        try:
            with psycopg.connect(pg_conn_str) as conn:
                with conn.cursor() as cur:
                    count = cur.execute(
                        """\
                        SELECT COUNT(DISTINCT d.document)
                        FROM   data_llamaindex_docs AS l
                        JOIN   documents            AS d ON d.id = l.document_id
                        WHERE  l.text_search_tsv @@ plainto_tsquery(%s);
                        """,
                        (query,),
                    ).fetchone()[0]
                    if count == 0:
                        return "No results found."

                    results = cur.execute(
                        """\
                        SELECT *
                        FROM (
                            SELECT DISTINCT ON (d.document)
                                d.document,
                                ts_rank(l.text_search_tsv, plainto_tsquery(%s)) AS rank
                            FROM   data_llamaindex_docs AS l
                            JOIN   documents            AS d ON d.id = l.document_id
                            WHERE  l.text_search_tsv @@ plainto_tsquery(%s)
                            ORDER  BY d.document, rank DESC
                        ) AS t
                        ORDER BY rank DESC LIMIT 10 OFFSET %s;
                        """,
                        (query, query, page * 10 - 10),
                    ).fetchall()

            docs = [Document.from_dict(obj[0]) for obj in results]

            display_object = {
                "total_results": count,
                "page": page,
            }

            display_docs = []

            for doc in docs:
                display_docs.append(
                    {
                        "title": doc.metadata.get("title", "Untitled document"),
                        "summary": doc.metadata.get("summary", None),
                    }
                )

            display_object["results"] = display_docs

            return json.dumps(display_object, indent=2)

        except Exception as e:
            raise

    return FunctionTool.from_defaults(
        async_fn=search_documents,
    )
