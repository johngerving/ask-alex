from typing import Dict
import numpy as np
import os

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat


from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
import ray.data
from haystack.utils import Secret
from docling_batch_converter import DoclingBatchConverter

import ray

def index_batch(batch: Dict[str, np.ndarray], indexing_pipeline: Pipeline) -> Dict[str, np.ndarray]:
    print(f"Processing batch of size {len(batch['data'])}")
    try:
        # Run the pipeline on the batch received
        indexing_pipeline.run(data={"paths": batch["data"]})
    except Exception as e:
        print("Error:", e)
    
    return batch

@ray.remote
def process_dataset(indexing_pipeline: Pipeline, ds: ray.data.Dataset):
    # Run the indexing pipeline in parallel in batches
    ds.map_batches(index_batch, fn_args=[indexing_pipeline], batch_size=2, num_gpus=1, concurrency=8).materialize()

@ray.remote
def index_documents(ds: ray.data.Dataset):
    '''
    Indexes a ray.data.Dataset containing links to PDFs.

    Args:
        ds: A ray.data.Dataset with a string field "data".
    '''
    from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.datamodel.base_models import InputFormat

    from haystack import Pipeline, Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack.utils import Secret

    import ray.data
    from docling_batch_converter import DoclingBatchConverter

    # Define pgvector document store to store the documents and their embeddings in
    document_store = PgvectorDocumentStore(
        connection_string=Secret(os.getenv("PG_CONN_STR")),
        table_name="haystack_docs",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="hnsw"
    )

    # Use the GPU to parse the PDFs
    accelerator_options = AcceleratorOptions(
        num_threads=16, device=AcceleratorDevice.CUDA
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options

    dl_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # Create a Haystack pipeline to index the documents
    indexing_pipeline = Pipeline()
    # Add a Docling converter to read the PDFs - have to pass in "User-Agent" header or else requests get blocked
    indexing_pipeline.add_component("converter", DoclingBatchConverter(converter=dl_converter, convert_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"}}))
    # Pipeline step to create document embeddings
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    # Pipeline step to write the documents to our PgvectorDocumentStore
    indexing_pipeline.add_component("writer", DocumentWriter(document_store))

    indexing_pipeline.connect("converter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    # Process the dataset
    result = process_dataset.remote(indexing_pipeline, ds)
    ray.get(result)