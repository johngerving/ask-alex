import json
from logging import Logger
import os
import re
from textwrap import dedent
from typing import Annotated, Dict, List, Optional

from llama_index.core.workflow import (
    Context,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from docling.chunking import HybridChunker
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import get_response_synthesizer
import psycopg
import logging


async def analyze_documents(ctx: Context, ids: List[str], query: str) -> Dict[str, str]:
    """Analyze a document by its ID and query it for relevant information. Expensive - Use only when the user specifically requests information from a document or documents.

    If the user is requesting information over multiple documents, you may pass up to 3 document or chunk IDs in.

    Args:
        ids (list[str]): A list of IDs of documents or chunks inside of documents to analyze. A maximum of 3 IDs can be provided at once. If more than 3 IDs are provided, an error will be raised.
        query (str): The query to run against the documents.

    Returns:
        Dict[str, str]: A dictionary containing the responses from the analysis for each document ID.
    """

    num_analyses: int = await ctx.store.get("num_analyses", default=0)
    if num_analyses >= 1:
        return "You have already performed another analysis. You are limited to one per user interaction. Wait until the next interaction."

    id_limit = 3

    if len(ids) > id_limit:
        raise ValueError(
            f"A maximum of {id_limit} document IDs can be provided at once. Please reduce the number of IDs and try again."
        )

    async with await psycopg.AsyncConnection.connect(os.getenv("PG_CONN_STR")) as conn:
        async with conn.cursor() as cur:
            # Fetch the document by its ID, or if it's a chunk, fetch the parent document
            await cur.execute(
                "SELECT DISTINCT ON (documents.id) documents.document, documents.id "
                "FROM documents "
                "RIGHT JOIN "
                "data_llamaindex_docs ON "
                "(documents.id = data_llamaindex_docs.document_id) "
                "WHERE documents.id = ANY(%s) OR data_llamaindex_docs.node_id = ANY(%s)",
                (ids, ids),
            )

            rows = await cur.fetchmany(id_limit)
            if not rows:
                raise ValueError(
                    "No documents found for the provided IDs. Please check the IDs and try again."
                )

            found_ids = [row[1] for row in rows]
            if any(id not in found_ids for id in ids):
                error_msg = ""
                for id in ids:
                    if id not in found_ids:
                        error_msg += f"Document or chunk with ID {id} not found.\n"

                raise ValueError(
                    "Some of the provided IDs do not correspond to any documents or chunks in the database:\n"
                    + error_msg
                )

            li_docs = [Document.from_dict(row[0]) for row in rows]

            parser = DoclingNodeParser(
                chunker=HybridChunker(
                    max_tokens=100000,
                )
            )

            analyses = ""

            for doc, id in zip(li_docs, ids):
                doc.excluded_llm_metadata_keys = doc.metadata.keys()
                nodes = await parser.aget_nodes_from_documents([doc])

                nodes_with_scores = [
                    NodeWithScore(
                        node=node,
                        score=1.0,
                    )
                    for node in nodes
                ]

                llm = OpenRouter(
                    model="meta-llama/llama-3.2-3b-instruct",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    context_window=100000,
                    max_tokens=4000,
                    is_chat_model=True,
                    is_function_calling_model=True,
                )

                synthesizer = get_response_synthesizer(
                    llm=llm,
                    response_mode=ResponseMode.TREE_SUMMARIZE,
                )

                response = await synthesizer.asynthesize(
                    query=query,
                    nodes=nodes_with_scores,
                )

                analyses += "<analysis>\n"
                analyses += f"ID: {id}\n"
                analyses += f"Title: {doc.metadata.get('title', 'No title')}\n"
                analyses += f"Response: {response}\n"
                analyses += "</analysis>\n\n"

            num_analyses: int = await ctx.store.get("num_analyses", default=0)
            num_analyses = num_analyses + 1

            if num_analyses > 0:
                analyses += "<note>You will not be able to perform another analysis until the next user interaction.</note>\n"

            await ctx.store.set("num_analyses", num_analyses)

            return analyses


tool = FunctionTool.from_defaults(
    async_fn=analyze_documents,
)
