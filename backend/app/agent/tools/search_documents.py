import os
from typing import Annotated, List, Literal, Optional

from llama_index.core.workflow import (
    Context,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from psycopg import sql
import psycopg
import logging

from app.agent.utils import Source

logger = logging.getLogger(__name__)


pg_conn_str = os.getenv("PG_CONN_STR")
if not pg_conn_str:
    raise ValueError("PG_CONN_STR environment variable not set")


async def search_documents(
    ctx: Context,
    query: Optional[
        Annotated[
            str,
            "A Tantivy query string that represents the key words or phrases to search for in the document.",
        ]
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
    sort_by: Optional[
        Annotated[
            Literal["relevance", "newest_first", "oldest_first"],
            "The field to sort the results by",
        ]
    ] = "relevance",
) -> str:
    """Search the knowledge base for relevant documents. Best used for information about documents themselves. Do NOT use if answer can be found in a specific chunk of a given document. Use the query_knowledge_base tool instead for that purpose.

    Before using this tool, you should call the call_metadata_agent tool to generate metadata for the search.

    Returns:
        str: A formatted string containing the document search results. If there are no results, it returns "No results found." If there was an error executing the search query, it raises an exception with the error message.

    Example user messages to use this tool for:
    - "What documents talk about ...?"
    - "Summarize documents about ..."
    - "Find documents that mention ..."
    """

    page_size = 5

    where_parts: List[sql.Composable] = []
    params: List[object] = []

    # Full-text keywords over tsvector column
    if query:
        where_parts.append(sql.SQL("text @@@ paradedb.parse_with_field('text', %s)"))
        params.append(query)

    # Metadata collection field
    if collection:
        where_parts.append(sql.SQL("(document->'metadata'->>'collection') = %s"))
        params.append(collection)

    # Metadata title field
    if title:
        where_parts.append(
            sql.SQL(
                "to_tsvector('english', document->'metadata'->>'title') @@ plainto_tsquery('english', %s)"
            )
        )
        params.append(title)

    # Metadata author field
    if author:
        where_parts.append(
            sql.SQL(
                "to_tsvector('english', jsonb_to_text(document->'metadata'->'author')) @@ plainto_tsquery('english', %s)"
            )
        )
        params.append(author)

    # Metadata department field
    if department:
        # `?` tests existence of a key in the JSONB array of authors
        where_parts.append(sql.SQL("document->'metadata'->'department' ? %s"))
        params.append(department)

    # Metadata start_year field
    if start_year is not None:
        where_parts.append(
            sql.SQL("(document->'metadata'->>'publication_date')::int >= %s")
        )
        params.append(start_year)

    if end_year is not None:
        where_parts.append(
            sql.SQL("(document->'metadata'->>'publication_date')::int <= %s")
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
    if sort_by == "relevance":
        if query:
            rank_sql = sql.SQL("paradedb.score(id)")
        else:
            rank_sql = sql.SQL("0")
    else:
        # Sort by publication date, descending
        rank_sql = sql.SQL("(document->'metadata'->>'publication_date')::int")

    order_by = sql.SQL("ASC" if sort_by == "oldest_first" else "DESC")

    # Compose the COUNT and DATA queries
    count_query = sql.SQL(
        """
        SELECT COUNT(id)
        FROM   documents
        {}
        """
    ).format(where_sql)

    data_query = sql.SQL(
        """
        SELECT document, id, {} AS rank
        FROM documents 
        {}
        ORDER BY rank {}
        LIMIT %s
        OFFSET %s
        """
    ).format(rank_sql, where_sql, order_by)

    # Combine params
    count_params = params.copy()

    data_params = params.copy()
    data_params.append(page_size)
    data_params.append(page * page_size - page_size)

    # Run the queries
    try:
        async with await psycopg.AsyncConnection.connect(pg_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(count_query, count_params)
                rows = await cur.fetchone()
                total = rows[0]
                if total == 0:
                    return "No results found."

                print("data query:", data_query.as_string(conn))
                await cur.execute(data_query, data_params)
                rows = await cur.fetchall()
    except Exception as e:
        logger.error(f"Error executing search query: {e}")
        raise Exception(
            f"There was an error executing the search query in the database: {e}"
        )

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
    out = [f"Total results: {total}", f"Page {page} of {total // page_size + 1}"]
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


tool = FunctionTool.from_defaults(async_fn=search_documents, return_direct=True)
