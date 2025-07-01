import logging
import psycopg
from psycopg.types.json import Jsonb
import ray
import ray.data

from digitalcommons import dataset_from_digitalcommons
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling_core.types.doc.document import DoclingDocument
from llama_index.core import Document as LIDocument

import json
import uuid

from typing import Any, Dict, List
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

        documents: List[DoclingDocument] = []
        with psycopg.connect(os.getenv("PG_CONN_STR")) as conn:
            with conn.cursor() as cur:
                for link in input_batch["link"]:
                    # Check if document already exists in the database
                    cur.execute(
                        "SELECT document FROM documents WHERE link = %s", (link,)
                    )
                    row = cur.fetchone()

                    doc = None
                    if row is not None:
                        existing_doc = row[0]
                        li_doc = LIDocument.from_dict(existing_doc)
                        doc = DoclingDocument.model_validate_json(li_doc.get_content())
                    else:
                        # If the document does not exist, we will convert it
                        try:
                            result = self.converter.convert(
                                link,
                                raises_on_error=False,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
                                },
                            )

                            if result.status == ConversionStatus.FAILURE:
                                raise Exception(f"{result.errors}")

                            doc = result.document
                        except Exception as e:
                            self.logger.error(f"Error converting document {link}: {e}")

                    documents.append(doc)

        json_docs = []
        statuses = []

        # Loop through the conversion results
        for link, metadata, dl_doc in zip(
            input_batch["link"], input_batch["metadata"], documents
        ):
            json_doc = None
            if dl_doc is not None:
                # Get the document content and create a LlamaIndex document with it
                text = json.dumps(dl_doc.export_to_dict())

                metadata: Dict[Any, Any] = json.loads(metadata)

                li_doc = LIDocument(
                    doc_id=str(uuid.uuid4()),
                    text=text,
                    metadata=metadata,
                )
                # Convert our document to a dictionary to store as text
                json_doc = json.dumps(li_doc.to_dict())

            json_docs.append(json_doc)
            statuses.append("success" if json_doc is not None else "failure")

        return {
            "link": input_batch["link"],
            "status": statuses,
            "document": json_docs,
        }


def is_successful(row: Dict[str, Any]) -> bool:
    return row["status"] == "success"


class SaveDocuments:
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        documents_dict = [Jsonb(json.loads(doc)) for doc in batch["document"]]

        values = zip(batch["link"], documents_dict)
        with psycopg.connect(os.getenv("PG_CONN_STR"), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (link, document) VALUES (%s, %s) ON CONFLICT (link) DO UPDATE SET document = EXCLUDED.document",
                    [(link, doc) for link, doc in values],
                )
        return batch


def convert_documents():
    """
    Converts a dataset containing links to PDFs to markdown documents.

    Args:
        ds: A ray.data.Dataset with a column containing links to PDFs.
    """

    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    ds = dataset_from_digitalcommons()

    # Convert PDFs to markdown documents
    ds = ds.map_batches(
        Converter,
        concurrency=8,  # Run 8 workers
        batch_size=32,  # Send batches of 32 links
        num_gpus=1,  # Use 1 GPU per worker
        num_cpus=1,  # Use 1 CPUs per worker
    )

    ds = ds.filter(fn=is_successful)

    ds = ds.map_batches(
        SaveDocuments,
        batch_size=128,
        num_cpus=1,
        concurrency=8,
    )

    ds.count()
