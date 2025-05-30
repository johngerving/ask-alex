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

        with psycopg.connect(os.getenv("PG_CONN_STR"), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (link, document) VALUES (%s, %s) ON CONFLICT (link) DO UPDATE SET document = EXCLUDED.document",
                    [(link, doc) for link, doc in zip(input_batch["link"], documents)],
                )
        return input_batch


def convert_documents():
    """
    Converts a dataset containing links to PDFs to markdown documents.

    Args:
        ds: A ray.data.Dataset with a column containing links to PDFs.
    """

    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    ds = get_metadata()

    # Convert PDFs to markdown documents
    ds = ds.map_batches(
        Converter,
        concurrency=8,  # Run 8 workers
        batch_size=128,  # Send batches of 32 links
        num_gpus=1,  # Use 1 GPU per worker
        num_cpus=1,  # Use 1 CPUs per worker
    )

    # Loop through the batches to force execution
    for _ in ds.iter_batches():
        pass
