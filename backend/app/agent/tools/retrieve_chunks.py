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
from llama_index.llms.openrouter import OpenRouter


async def make_retrieve_chunks_tool(ctx: Context) -> FunctionTool:
    small_llm = OpenRouter(
        model="meta-llama/llama-4-scout",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=41000,
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

    async def retrieve_chunks(
        query: Annotated[str, "The query to retrieve relevant chunks"],
    ) -> str:
        """Search the knowledge base for relevant chunks from documents. Best used for questions asking for information about particular topics, concepts, events, entities, etc. Do NOT use if the question asks seems to require a summary of any given document or set of documents. Use the search_documents tool instead for that purpose.

        When using this tool, separate distinct queries/questions into separate tool calls, rather than combining multiple questions into one single tool call. This will allow for more diverse information to be retrieved.

        Example user messages to use this tool for:
        - "What is ...?"
        - "Tell me about the history of ..."
        - "Write a report on ..."

        Example inputs to this tool:
        - "Cal Poly Humboldt history"
        - "Sequoia Park location"
        """
        try:
            # Use the retriever to get relevant nodes
            nodes = await retriever.aretrieve(query)

            # Get sources set in tool
            sources: List[TextNode] = await ctx.get("retrieved_nodes", [])
            sources = sources + nodes
            await ctx.set("retrieved_nodes", sources)

            content = ""

            for node in nodes:
                # Format chunks to be returned to agent
                content += ("<chunk id={doc_id}>\n" "{content}\n" "</chunk>\n").format(
                    doc_id=node.node_id[:8],
                    content=node.get_content(metadata_mode=MetadataMode.LLM),
                )

        except Exception as e:
            raise

        return content

    return FunctionTool.from_defaults(
        async_fn=retrieve_chunks,
    )
