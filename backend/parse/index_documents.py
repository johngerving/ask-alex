from typing import Dict
import numpy as np
import os
import json

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat


from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
import ray.data
from haystack.utils import Secret
from haystack.utils import ComponentDevice, Device

import ray

class DocumentIndexer:
    def __init__(self):
        from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

        from haystack import Pipeline
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
        from haystack.components.writers import DocumentWriter
        from haystack.utils import Secret

        # Define pgvector document store to store the documents and their embeddings in
        document_store = PgvectorDocumentStore(
            connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
            table_name="haystack_docs",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=True,
            search_strategy="hnsw"
        )

        # Create a Haystack pipeline to index the documents
        self.pipeline = Pipeline()
        # Clean the documents
        self.pipeline.add_component("cleaner", DocumentCleaner(remove_empty_lines=True, remove_repeated_substrings=True, remove_regex="^\[\d+\]|\(\d+\)|\b[A-Z][a-z]+ et al\., \d{4}|\(\w+, \d{4}\)|doi:|http[s]?://"))
        # Split them by paragraph
        self.pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200, split_overlap=50))
        # Pipeline step to create document embeddings
        self.pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(device=ComponentDevice.from_single(Device.gpu())))
        # Pipeline step to write the documents to our PgvectorDocumentStore
        self.pipeline.add_component("writer", DocumentWriter(document_store))

        self.pipeline.connect("cleaner.documents", "splitter.documents")
        self.pipeline.connect("splitter.documents", "embedder.documents")
        self.pipeline.connect("embedder", "writer")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        '''
        Converts a batch of links to PDFs and runs them through an indexing pipeline.

        Args:
            batch: A dictionary with a column for links to PDFs.
            indexing_pipeline: A Haystack Pipeline object to run the batch through.
        '''
        from haystack import Document
        import json

        # Get the text stored in each document, convert it to a dictionary, and convert each of those into a Haystack Document
        documents = [Document.from_dict(json.loads(doc)) for doc in batch['document']]

        print(f"Processing batch of size {len(documents)}")
        try:
            # Run the pipeline on the batch received
            res = self.pipeline.run({"cleaner": {"documents": documents}}, include_outputs_from={"splitter"})
        except Exception as e:
            print("Error:", e)
        
        return batch

@ray.remote
def index_documents():
    '''
    Indexes a ray.data.Dataset containing links to PDFs stored in S3.
    '''

    from pyarrow import fs

    # Create a filesystem to store the converted documents in
    filesys = fs.S3FileSystem(endpoint_override=os.getenv("AWS_ENDPOINT_URL"))

    ds = ray.data.read_parquet("s3://documents/", filesystem=filesys)

    # Run the indexing pipeline in parallel in batches
    ds.map_batches(
        DocumentIndexer, 
        batch_size=32, 
        num_gpus=1,     # 1 GPU per worker
        concurrency=4   # 4 workers
    ).materialize()
