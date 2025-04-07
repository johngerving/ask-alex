import psycopg
import ray
import ray.data

from docling.datamodel.base_models import InputFormat, ConversionStatus
from agno.document import Document
from llama_index.core import Document as LIDocument
import uuid

from typing import Dict
import numpy as np

import os


class Converter:
    def __init__(self):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.datamodel.base_models import InputFormat, ConversionStatus

        from docling_haystack.converter import MetaExtractor

        # Use the GPU to parse the PDFs
        accelerator_options = AcceleratorOptions(
            num_threads=16, device=AcceleratorDevice.CUDA
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        self.meta_extractor = MetaExtractor()

        # Create a Docling converter to convert our PDFs
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import json

        from agno.document import Document
        import uuid

        from docling.datamodel.base_models import ConversionStatus

        print(f'Processing batch of size {len(input_batch["link"])}')
        # Convert all the documents in the input batch - need a User-Agent header to avoid getting blocked
        conversion_results = self.converter.convert_all(
            input_batch["link"],
            raises_on_error=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
            },
        )

        documents = []

        # Loop through the conversion results
        for link, result in zip(input_batch["link"], conversion_results):
            if result.status == ConversionStatus.FAILURE:
                print(f"Error processing document at {link} - {result.errors}")
                documents.append(None)
            else:
                # Get the document contnt and create a LlamaIndex document with it
                dl_doc = result.document
                text = dl_doc.export_to_markdown()
                li_doc = LIDocument(
                    doc_id=str(uuid.uuid4()),
                    text=text,
                    metadata=self.meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc),
                )
                # Convert our Agno document to a dictionary to store as text
                json_doc = json.dumps(li_doc.to_dict())
                documents.append(json_doc)
        return {"link": input_batch["link"], "document": np.array(documents)}


@ray.remote
def convert_documents(ds: ray.data.Dataset):
    """
    Converts a dataset containing links to PDFs to markdown documents.

    Args:
        ds: A ray.data.Dataset with a column containing links to PDFs.
    """

    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    # Drop the documents table if it exists
    with psycopg.connect(conn_str) as conn:
        conn.cursor().execute(
            "CREATE TABLE IF NOT EXISTS documents (link TEXT PRIMARY KEY, document TEXT)"
        )

    # Convert PDFs to markdown documents
    ds = ds.map_batches(
        Converter,
        concurrency=8,  # Run 8 workers
        batch_size=64,  # Send batches of 32 links
        num_gpus=1,  # Use 1 GPU per worker
    )
    # Filter out the documents that had errors in them
    ds = ds.filter(lambda row: row["document"] is not None)
    # Write dataset to Postgres
    ds.write_sql(
        "INSERT INTO documents VALUES(%s, %s)",
        lambda: psycopg.connect(conn_str),
    )
