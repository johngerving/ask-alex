import logging
import psycopg
from psycopg.types.json import Jsonb
import ray
import ray.data

from get_metadata import get_metadata
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat, ConversionStatus
from llama_index.core import Document as LIDocument

import json
import uuid

from typing import Any, Dict
import numpy as np

import os


class Converter:
    def __init__(self):

        # Use the GPU to parse the PDFs
        accelerator_options = AcceleratorOptions(
            num_threads=16, device=AcceleratorDevice.CUDA
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # Create a Docling converter to convert our PDFs
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.logger = logging.getLogger("ray.data")

    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
        for link, metadata, result in zip(
            input_batch["link"], input_batch["metadata"], conversion_results
        ):
            if result.status == ConversionStatus.FAILURE:
                print(f"Error processing document at {link} - {result.errors}")
                documents.append(None)
            else:
                # Get the document content and create a LlamaIndex document with it
                dl_doc = result.document
                text = json.dumps(dl_doc.export_to_dict())

                # Extract metadata from Digital Commons API
                metadata_keys = [
                    "title",
                    "url",
                    "download_link",
                    "author",
                    "publication_date",
                    "discipline",
                    "abstract",
                ]
                metadata: Dict[Any, Any] = json.loads(metadata)
                metadata = {key: metadata.get(key) for key in metadata_keys}

                li_doc = LIDocument(
                    doc_id=str(uuid.uuid4()),
                    text=text,
                    metadata=metadata,
                )
                # Convert our document to a dictionary to store as text
                json_doc = Jsonb(li_doc.to_dict())
                documents.append(json_doc)
        return {"link": input_batch["link"], "document": np.array(documents)}


def convert_documents():
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
            "CREATE TABLE IF NOT EXISTS documents (link TEXT, document JSONB)"
        )

    ds = get_metadata()

    # Convert PDFs to markdown documents
    ds = ds.map_batches(
        Converter,
        concurrency=8,  # Run 8 workers
        batch_size=64,  # Send batches of 32 links
        num_gpus=1,  # Use 1 GPU per worker
        num_cpus=2,  # Use 2 CPUs per worker
    )
    # Filter out the documents that had errors in them
    ds = ds.filter(lambda row: row["document"] is not None)
    # Write dataset to Postgres
    ds.write_sql(
        "INSERT INTO documents VALUES(%s, %s)",
        lambda: psycopg.connect(conn_str),
    )
