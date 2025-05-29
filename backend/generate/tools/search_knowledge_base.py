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


async def make_search_knowledge_base_tool(ctx: Context) -> FunctionTool:
    logger: Logger = await ctx.get("logger")
    small_llm: LLM = await ctx.get("small_llm")

    pg_conn_str = os.getenv("PG_CONN_STR")
    if not pg_conn_str:
        raise ValueError("PG_CONN_STR environment variable not set")

    # Get Postgres credentials from connection string
    pg_url = urlparse(pg_conn_str)
    host = pg_url.hostname
    port = pg_url.port
    database = pg_url.path[1:]
    user = pg_url.username
    password = pg_url.password

    # Vector store to store chunks + embeddings in
    vector_store = PGVectorStore.from_params(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name="llamaindex_docs",
        schema_name="public",
        hybrid_search=True,
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    # Index the chunks, using HuggingFace embedding model
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )

    # Vector retriever
    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=50,
        verbose=True,
    )

    # Keyword retriever
    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse", similarity_top_k=50
    )

    # Fuse results from both retrievers
    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=10,
        llm=small_llm,
        num_queries=3,  # LLM generates extra queries
        mode="relative_score",
    )

    async def search_knowledge_base(
        query: Annotated[str, "The query to search the knowledge base for"],
    ) -> str:
        """Search the knowledge base for relevant chunks from documents."""
        logger.info(f"Running search_knowledge_base with query: {query}")
        try:
            # Use the retriever to get relevant nodes
            nodes = await retriever.aretrieve(query)
            logger.info(f"Retrieved {len(nodes)} nodes")

            # Get sources set in tool
            sources: List[TextNode] = await ctx.get("sources")
            sources = sources + nodes
            await ctx.set("sources", sources)

            content = ""

            for node in nodes:
                # Format chunks to be returned to agent
                content += ("<chunk id={doc_id}>\n" "{content}\n" "</chunk>\n").format(
                    doc_id=node.node_id[:8],
                    content=node.get_content(metadata_mode=MetadataMode.LLM),
                )

        except Exception as e:
            logger.error(e)
            raise

        return content

    return FunctionTool.from_defaults(
        async_fn=search_knowledge_base,
    )
