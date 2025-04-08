from typing import Dict
import numpy as np
import os
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat

from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from urllib.parse import urlparse

from document_cleaner import DocumentCleaner

import psycopg
import ray.data

import logging

import ray


class DocumentIndexer:
    def __init__(self):
        import logging

        self.logger = logging.getLogger("ray.data")

        pg_url = urlparse(os.getenv("PG_CONN_STR"))
        host = pg_url.hostname
        port = pg_url.port
        database = pg_url.path[1:]
        user = pg_url.username
        password = pg_url.password

        self.logger.info(f"{host} {port} {database} {user} {password}")

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
        )

        self.pipeline = IngestionPipeline(
            vector_store=vector_store,
            transformations=[
                SentenceSplitter(chunk_size=250, chunk_overlap=50),
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

        self.logger.info(f"Processing batch of size {len(documents)}")
        self.pipeline.run(documents=documents)

        return batch


@ray.remote
def index_documents():
    """
    Indexes a ray.data.Dataset containing links to PDFs stored in S3.
    """

    from pyarrow import fs
    import psycopg

    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    # Drop the documents table if it exists
    with psycopg.connect(conn_str) as conn:
        conn.cursor().execute("DROP TABLE IF EXISTS llamaindex_docs")

    # Read full documents from Postgres database
    ds = ray.data.read_sql("SELECT * FROM documents", lambda: psycopg.connect(conn_str))

    # Run the indexing pipeline in parallel in batches
    ds.map_batches(
        DocumentIndexer,
        batch_size=32,
        num_gpus=1,  # 1 GPU per worker
        concurrency=8,  # 8 workers
    ).materialize()
