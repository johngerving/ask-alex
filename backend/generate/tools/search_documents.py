from logging import Logger
import os
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

# SELECT DISTINCT elem FROM documents CROSS JOIN LATERAL jsonb_array_elements_text(document->'metadata'->'discipline') AS t(elem) WHERE jsonb_typeof(document->'metadata'->'discipline') = 'array';


async def make_document_search_tool(ctx: Context) -> FunctionTool:
    logger: Logger = await ctx.get("logger")

    pg_conn_str = os.getenv("PG_CONN_STR")
    if not pg_conn_str:
        raise ValueError("PG_CONN_STR environment variable not set")

    async def search_documents(
        query: Annotated[str, "The query to search the knowledge base for"],
    ) -> str:
        """Search the knowledge base for relevant documents."""
        logger.info(f"Running search_documents with query: {query}")
        try:
            pass
        except Exception as e:
            logger.error(e)
            raise

    return FunctionTool.from_defaults(
        async_fn=search_documents,
    )
