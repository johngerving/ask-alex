from typing import Dict
import numpy as np
import os
from urllib.parse import urlparse

from docling.chunking import HybridChunker

from llama_index.core import Document
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.ingestion import IngestionPipeline

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import psycopg

import ray.data
import ray


class DocumentIndexer:
    def __init__(self):
        import logging

        self.logger = logging.getLogger("ray.data")

        # Get Postgres credentials from connection string
        pg_url = urlparse(os.getenv("PG_CONN_STR"))
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
            embed_dim=768,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
            hybrid_search=True,
        )

        self.pipeline = IngestionPipeline(
            vector_store=vector_store,
            transformations=[
                DoclingNodeParser(
                    chunker=HybridChunker(
                        max_tokens=500,
                    )
                ),
                HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                ),
            ],
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Converts a batch of links to PDFs and runs them through an indexing pipeline.

        Args:
            batch: A dictionary with a column for links to PDFs.
        """
        import json

        # Get the text stored in each document, convert it to a dictionary, and convert each of those into a LlamaIndex Document
        documents = [Document.from_dict(json.loads(doc)) for doc in batch["document"]]

        for document in documents:
            # Exclude some metadata keys from being shown to the LLM and passed to the embedder so as to preserve the chunk size
            document.excluded_llm_metadata_keys = [
                "abstract",
                "url",
                "download_link",
                "dl_meta",
            ]
            document.excluded_embed_metadata_keys = [
                "abstract",
                "url",
                "download_link",
                "publication_date",
                "dl_meta",
                "summary",
            ]

            # Filter discipline list
            max_disciplines = 5
            discipline = document.metadata.get("discipline")
            if discipline is not None:
                discipline = list(set(discipline))  # Only unique values
                discipline = discipline[
                    0 : min(max_disciplines, len(discipline))
                ]  # Maximum of max_disciplines
                document.metadata["discipline"] = discipline

            # Limit number of authors shown in document
            max_authors = 5
            author = document.metadata.get("author")
            if author is not None:
                author = author[0 : min(max_authors, len(author))]
                document.metadata["author"] = author

        self.logger.info(f"Processing batch of size {len(documents)}")
        try:
            self.pipeline.run(documents=documents)
        except Exception as e:
            self.logger.error(e)
            self.logger.info([document.metadata for document in documents])
            raise

        return batch


def index_documents():
    """
    Indexes a ray.data.Dataset containing links to PDFs stored in S3.
    """
    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    # Drop the documents table if it exists
    with psycopg.connect(conn_str) as conn:
        conn.cursor().execute("DROP TABLE IF EXISTS data_llamaindex_docs")

    # Read full documents from Postgres database
    ds = ray.data.read_sql("SELECT * FROM documents", lambda: psycopg.connect(conn_str))

    # Run the indexing pipeline in parallel in batches
    ds = ds.map_batches(
        DocumentIndexer,
        batch_size=32,
        num_gpus=1,  # 1 GPU per worker
        concurrency=8,  # 8 workers
    )

    for _ in ds.iter_batches(batch_size=None):
        pass
