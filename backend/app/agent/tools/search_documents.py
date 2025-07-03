import json
from logging import Logger
import os
import re
from textwrap import dedent
from typing import Annotated, List, Optional
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
from llama_index.core.tools import FunctionTool
import psycopg
from psycopg import connect, sql
import logging

from app.agent.utils import Source

logger = logging.getLogger(__name__)


pg_conn_str = os.getenv("PG_CONN_STR")
if not pg_conn_str:
    raise ValueError("PG_CONN_STR environment variable not set")


async def search_documents(
    ctx: Context,
    query: Optional[
        Annotated[str, "A PostgreSQL tsquery string to search for in the document text"]
    ] = None,
    title: Optional[Annotated[str, "The title of the document to search for"]] = None,
    author: Optional[Annotated[str, "The author of the document to search for"]] = None,
    department: Optional[
        Annotated[str, "The department that published the document"]
    ] = None,
    collection: Optional[
        Annotated[str, "The collection that the document belongs to"]
    ] = None,
    start_year: Optional[
        Annotated[int, "The start of the date range to search for documents"]
    ] = None,
    end_year: Optional[
        Annotated[int, "The end of the date range to search for documents"]
    ] = None,
    page: Optional[
        Annotated[int, "The page number for pagination, starting from 1"]
    ] = 1,
) -> str:
    """Search the knowledge base for relevant documents. Best used for information about documents themselves and summaries of specific documents. Do NOT use if answer can be found in a specific chunk of a given document. Use the retrieve_chunks tool instead for that purpose.

    Example user messages to use this tool for:
    - "What documents talk about ...?"
    - "Summarize documents about ..."
    """

    where_parts: List[sql.Composable] = []
    params: List[object] = []

    # Full-text keywords over tsvector column
    if query:
        where_parts.append(sql.SQL("l.text_search_tsv @@ to_tsquery(%s)"))
        params.append(query)

    # Metadata collection field
    if collection:
        where_parts.append(sql.SQL("(d.document->'metadata'->>'collection') = %s"))
        params.append(collection)

    # Metadata title field
    if title:
        where_parts.append(
            sql.SQL(
                "to_tsvector('english', d.document->'metadata'->>'title') @@ to_tsquery('english', %s)"
            )
        )
        params.append(title)

    # Metadata author field
    if author:
        where_parts.append(
            sql.SQL(
                "to_tsvector('english', jsonb_to_text(d.document->'metadata'->'author')) @@ to_tsquery('english', %s)"
            )
        )
        params.append(author)

    # Metadata department field
    if department:
        # `?` tests existence of a key in the JSONB array of authors
        where_parts.append(sql.SQL("d.document->'metadata'->'department' ? %s"))
        params.append(department)

    # Metadata start_year field
    if start_year is not None:
        where_parts.append(
            sql.SQL("(d.document->'metadata'->>'publication_date')::int >= %s")
        )
        params.append(start_year)

    if end_year is not None:
        where_parts.append(
            sql.SQL("(d.document->'metadata'->>'publication_date')::int <= %s")
        )
        params.append(end_year)

    # Concatenate the WHERE clause
    where_sql = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
        if where_parts
        else sql.SQL("")
    )

    # Ranking expression (only if keywords supplied)
    rank_sql: sql.Composable
    rank_params: List[object]
    if query:
        rank_sql = sql.SQL("ts_rank(l.text_search_tsv, to_tsquery(%s))")
        # rank needs the same query text once more
        rank_params = [query]
    else:
        # If there's no keyword search, give every row the same rank (zero)
        rank_sql = sql.SQL("0")
        rank_params = []

    # Compose the COUNT and DATA queries
    count_query = sql.SQL(
        """
        SELECT COUNT(DISTINCT d.id)
        FROM   data_llamaindex_docs AS l
        JOIN   documents            AS d ON d.id = l.document_id
        {}
        """
    ).format(where_sql)

    data_query = sql.SQL(
        """
        SELECT *
        FROM (
            SELECT DISTINCT ON (d.id)
                d.document,
                {} AS rank
            FROM   data_llamaindex_docs AS l
            JOIN   documents            AS d ON d.id = l.document_id
            {}
            ORDER  BY d.id, rank DESC
        ) AS sub
        ORDER BY rank DESC
        LIMIT  10
        OFFSET %s;
        """
    ).format(rank_sql, where_sql)

    # Combine params
    count_params = params.copy()
    data_params: List[object] = []
    if query:
        data_params.append(query)

    data_params.extend(params)
    data_params.append(page * 10 - 10)

    # Run the queries
    try:
        with connect(pg_conn_str) as conn, conn.cursor() as cur:
            total = cur.execute(count_query, count_params).fetchone()[0]
            if total == 0:
                return "No results found."

            rows = cur.execute(data_query, data_params).fetchall()
    except Exception as e:
        logger.error(f"Error executing search query: {e}")
        raise Exception(f"Error executing search query: {e}")

    docs = [Document.from_dict(obj[0]) for obj in rows]

    sources_used = [
        Source(
            id=doc.doc_id,
            link=doc.metadata.get("download_link", None),
        )
        for doc in docs
    ]

    sources: List[Source] = await ctx.get("retrieved_sources", [])
    sources.extend(sources_used)
    await ctx.set("retrieved_sources", sources)

    # Format results
    out = [f"Total results: {total}", f"Page {page} of {total // 10 + 1}"]
    out.append("<results>")
    for doc in docs:
        out.append("<result>")
        out.append(
            f"Document ID: {doc.doc_id}\n"
            f"Title: {doc.metadata.get('title', 'Untitled document')}\n"
            f"Authors: {doc.metadata.get('author', 'Unknown author')}\n"
            f"Publication Date: {doc.metadata.get('publication_date', 'Unknown date')}\n"
            f"Department: {doc.metadata.get('department', 'Unknown department')}\n"
            f"Collection: {doc.metadata.get('collection', 'Unknown collection')}"
        )
        out.append("</result>")
    out.append("</results>")

    return "\n".join(out)


tool = FunctionTool.from_defaults(
    async_fn=search_documents,
)
