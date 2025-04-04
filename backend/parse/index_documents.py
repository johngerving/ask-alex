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

from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.google import GeminiEmbedder
from document_cleaner import DocumentCleaner

import psycopg
import ray.data

import logging

import ray


class DocumentIndexer:
    def __init__(self):
        import logging

        from agno.vectordb.pgvector import PgVector, SearchType
        from agno.embedder.google import GeminiEmbedder
        from document_cleaner import DocumentCleaner

        self.logger = logging.getLogger("ray.data")

        # Define pgvector document store to store the documents and their embeddings in
        self.vector_db = PgVector(
            table_name="agno_docs",
            schema="public",
            db_url=os.getenv("PG_CONN_STR"),
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(),
        )
        self.document_cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_repeated_substrings=True,
            remove_regex="^\[\d+\]|\(\d+\)|\b[A-Z][a-z]+ et al\., \d{4}|\(\w+, \d{4}\)|doi:|http[s]?://",
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Converts a batch of links to PDFs and runs them through an indexing pipeline.

        Args:
            batch: A dictionary with a column for links to PDFs.
        """
        from agno.document import Document
        import json

        # Get the text stored in each document, convert it to a dictionary, and convert each of those into an Agno Document
        documents = [Document.from_dict(json.loads(doc)) for doc in batch["document"]]

        self.logger.info(f"Processing batch of size {len(documents)}")
        # Clean the documents
        documents = self.document_cleaner.clean(documents)
        self.vector_db.insert(
            documents=documents,
            batch_size=len(batch),
        )

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
        conn.cursor().execute("DROP TABLE IF EXISTS agno_docs")

    # Read full documents from Postgres database
    ds = ray.data.read_sql("SELECT * FROM documents", lambda: psycopg.connect(conn_str))

    # Run the indexing pipeline in parallel in batches
    ds.map_batches(
        DocumentIndexer,
        batch_size=32,
        num_gpus=1,  # 1 GPU per worker
        concurrency=4,  # 4 workers
    ).materialize()
