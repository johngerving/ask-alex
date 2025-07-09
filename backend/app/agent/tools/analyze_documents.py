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


async def analyze_documents(ids: List[str], query: str) -> Dict[str, str]:
    """Analyze a document by its ID and query it for relevant information.

    Args:
        ids (list[str]): A list of IDs of documents or chunks inside of documents to analyze. A maximum of 5 IDs can be provided at once. If more than 5 IDs are provided, an error will be raised.
        query (str): The query to run against the documents.

    Returns:
        Dict[str, str]: A dictionary containing the responses from the analysis for each document ID.
    """

    id_limit = 5

    if len(ids) > id_limit:
        raise ValueError(
            f"A maximum of {id_limit} document IDs can be provided at once. Please reduce the number of IDs and try again."
        )

    with psycopg.connect(os.getenv("PG_CONN_STR")) as conn:
        with conn.cursor() as cur:
            # Fetch the document by its ID, or if it's a chunk, fetch the parent document
            cur.execute(
                "SELECT documents.document, documents.id "
                "FROM documents "
                "RIGHT JOIN "
                "data_llamaindex_docs ON "
                "(documents.id = data_llamaindex_docs.document_id) "
                "WHERE documents.id = ANY(%s) OR data_llamaindex_docs.node_id = ANY(%s)",
                (ids, ids),
            )

            rows = cur.fetchall()
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
                    max_tokens=50000,
                )
            )

            analyses = ""

            for doc, id in zip(li_docs, ids):
                nodes = await parser.aget_nodes_from_documents([doc])

                nodes_with_scores = [
                    NodeWithScore(
                        node=node,
                        score=1.0,
                    )
                    for node in nodes
                ]

                llm = OpenRouter(
                    model="meta-llama/llama-3.2-1b-instruct",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    context_window=128000,
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
                analyses += f"Response: {response}\n"
                analyses += "</analysis>\n"

            return analyses


tool = FunctionTool.from_defaults(
    async_fn=analyze_documents,
)
