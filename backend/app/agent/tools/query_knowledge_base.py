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
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.schema import MetadataMode
from llama_index.core.llms import LLM
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel

from app.agent.utils import Source
from app.agent.query_engines.chunk_query_engine import ChunkQueryEngine

small_llm = OpenRouter(
    model="qwen/qwen3-30b-a3b",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    context_window=128000,
    max_tokens=4000,
    is_chat_model=True,
    is_function_calling_model=True,
)

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


async def query_knowledge_base(
    ctx: Context,
    question: Annotated[
        str, "A standalone question to answer using chunks from the knowledge base."
    ],
) -> str:
    """Search the knowledge base for relevant chunks from documents and synthesize an answer. Best used for questions asking for information about particular topics, concepts, events, entities, etc. Do NOT use if the question asks seems to require a summary of any given document or set of documents. Use the search_documents tool instead for that purpose.

    When using this tool, separate distinct queries/questions into separate tool calls, rather than combining multiple questions into one single tool call. This will allow for more diverse information to be retrieved.

    Example user messages to use this tool for:
    - "What is ...?"
    - "Tell me about the history of ..."
    - "Write a report on ..."

    Example inputs to this tool:
    - "What is the history of Cal Poly Humboldt?"
    - "What is the location of Sequoia Park?"
    """
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
        similarity_top_k=7,
        llm=small_llm,
        num_queries=3,  # LLM generates extra queries
        mode="relative_score",
    )

    query_engine = ChunkQueryEngine(
        retriever=retriever,
        llm=small_llm,
    )

    try:
        # Use the retriever to get relevant nodes
        response = await query_engine.aquery(question)

        response_str = response.response
        nodes = response.source_nodes

        sources_used = [
            Source(id=node.node_id, link=node.metadata.get("download_link", None))
            for node in nodes
        ]

        # Get sources set in tool
        sources: List[Source] = await ctx.get("retrieved_sources", [])
        sources = sources + sources_used
        await ctx.set("retrieved_sources", sources)
    except Exception as e:
        raise

    return response_str


tool = FunctionTool.from_defaults(
    async_fn=query_knowledge_base,
)
